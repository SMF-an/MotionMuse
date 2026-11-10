"""MotionMuse camera server with ArUco and markerless board modes."""
from __future__ import annotations

import argparse
from http import server
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlparse

import cv2
import numpy as np

from serving.aruco_tracking import ArucoBoardTracker
from serving.markerless_tracking import MarkerlessBoardTracker


ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web"


class CameraWorker:
    def __init__(self, camera_index: int) -> None:
        self.capture = cv2.VideoCapture(camera_index)
        if not self.capture.isOpened():
            raise RuntimeError(f"无法打开摄像头 {camera_index}")
        self.tracker = ArucoBoardTracker()
        self.markerless = MarkerlessBoardTracker()
        self.mode = "aruco"
        self.frozen_frame: np.ndarray | None = None
        self.frozen_at: float | None = None
        self.frame: np.ndarray | None = None
        self.lock = threading.Lock()
        self.jpeg = b""
        self.state: dict = {"status": "starting"}
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while self.running:
            ok, frame = self.capture.read()
            if not ok:
                time.sleep(0.03)
                continue
            height, width = frame.shape[:2]
            success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if success:
                with self.lock:
                    self.frame = frame.copy()
                    pose = (self.tracker.process(frame).to_dict(width, height)
                            if self.mode == "aruco" else self.markerless.process(frame).to_dict())
                    self.jpeg = encoded.tobytes()
                    self.state = {
                        "mode": self.mode,
                        "frame_width": width,
                        "frame_height": height,
                        "dictionary": "DICT_4X4_50",
                        "required_ids": [0, 1, 2, 3],
                        **pose,
                    }

    def set_mode(self, mode: str) -> None:
        if mode not in ("aruco", "markerless"):
            raise ValueError("无效模式")
        with self.lock:
            if self.mode != mode:
                self.mode = mode
                self.frozen_frame = None
                self.frozen_at = None
                if mode == "markerless":
                    self.markerless.reset()

    def freeze(self) -> bytes:
        with self.lock:
            if self.frame is None or not self.jpeg:
                raise ValueError("摄像头画面尚未就绪")
            self.frozen_frame = self.frame.copy()
            self.frozen_at = time.monotonic()
            return self.jpeg

    def calibrate(self, corners: list[dict[str, float]]) -> None:
        with self.lock:
            if self.mode != "markerless" or self.frozen_frame is None:
                raise ValueError("请先切换无标记模式并冻结画面")
            self.markerless.calibrate(self.frozen_frame, corners, self.frozen_at)
            self.frozen_frame = None
            self.frozen_at = None

    def snapshot(self) -> tuple[bytes, dict]:
        with self.lock:
            return self.jpeg, self.state.copy()

    def close(self) -> None:
        self.running = False
        self.thread.join(timeout=1)
        self.capture.release()


class MotionMuseHandler(server.BaseHTTPRequestHandler):
    worker: CameraWorker

    def log_message(self, *_: object) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/state":
            _, state = self.worker.snapshot()
            payload = json.dumps(state, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if path == "/stream.mjpg":
            self._stream_mjpeg()
            return
        self._serve_static("app.html" if path in ("/", "/app.html") else path.lstrip("/"))

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/markerless/freeze":
            try:
                data = self.worker.freeze()
            except ValueError as exc:
                self.send_error(409, str(exc))
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if path not in ("/api/mode", "/api/markerless/calibrate"):
            self.send_error(404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size < 1 or size > 8192:
                raise ValueError("请求大小无效")
            payload = json.loads(self.rfile.read(size))
            if path == "/api/mode":
                self.worker.set_mode(payload["mode"])
            else:
                self.worker.calibrate(payload["corners"])
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            data = json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8")
            self.send_response(400)
        else:
            data = b'{"ok":true}'
            self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _stream_mjpeg(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            while True:
                jpeg, _ = self.worker.snapshot()
                if jpeg:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg + b"\r\n")
                time.sleep(1 / 30)
        # 浏览器刷新、关闭页面或重连 MJPEG 时会主动中止连接；这不是服务错误。
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return

    def _serve_static(self, relative: str) -> None:
        target = (WEB_ROOT / relative).resolve()
        if WEB_ROOT not in target.parents or not target.is_file():
            self.send_error(404)
            return
        content_type = "text/html; charset=utf-8" if target.suffix == ".html" else "application/javascript"
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    worker = CameraWorker(args.camera)
    MotionMuseHandler.worker = worker
    httpd = server.ThreadingHTTPServer(("0.0.0.0", args.port), MotionMuseHandler)
    print(f"MotionMuse server: http://localhost:{args.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        worker.close()


if __name__ == "__main__":
    main()
