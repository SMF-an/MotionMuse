"""Preview an Ultralytics YOLO pose model on a camera or video."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_source(value: str) -> int | str:
    """Treat a numeric source as a camera index; otherwise use it as a path/URL."""
    return int(value) if value.isdecimal() else value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="摄像头编号或视频文件路径，默认 0")
    parser.add_argument(
        "--weights",
        default="yolo11n-pose.pt",
        help="Ultralytics Pose 权重；默认权重首次运行时会自动下载",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="推理输入尺寸，默认 640")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值，默认 0.25")
    parser.add_argument("--device", help="推理设备，例如 cpu、0；默认由 Ultralytics 自动选择")
    args = parser.parse_args()

    weights = args.weights
    if (weights.endswith((".pt", ".onnx", ".engine"))
            and not Path(weights).exists()
            and Path(weights).name != "yolo11n-pose.pt"):
        parser.error(f"模型文件不存在：{weights}")

    model = YOLO(weights)
    capture = cv2.VideoCapture(parse_source(args.source))
    if not capture.isOpened():
        parser.error(f"无法打开摄像头或视频：{args.source}")

    is_stock_pose = Path(weights).name.lower() == "yolo11n-pose.pt"
    previous_time = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            prediction_args = {
                "source": frame,
                "imgsz": args.imgsz,
                "conf": args.conf,
                "verbose": False,
            }
            if args.device:
                prediction_args["device"] = args.device
            result = model.predict(**prediction_args)[0]
            preview = result.plot()

            now = time.perf_counter()
            fps = 1.0 / max(now - previous_time, 1e-6)
            previous_time = now
            cv2.putText(preview, f"FPS: {fps:.1f}", (16, 32), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (40, 240, 120), 2, cv2.LINE_AA)
            if is_stock_pose:
                cv2.putText(preview, "COCO pretrained: human pose only; board corners are not trained",
                            (16, preview.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 220, 255), 2, cv2.LINE_AA)

            cv2.imshow("MotionMuse - YOLO Pose Preview (Q to quit)", preview)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
