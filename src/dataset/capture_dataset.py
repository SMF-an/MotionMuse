"""Record labeled board-tracking clips with a JSON sidecar for each session."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-id", required=True, help="板子编号，如 B01；无板负样本用 NONE")
    parser.add_argument("--condition", required=True, help="采集类别，如 slow / performance / fast / occlusion / negative")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV 摄像头编号，默认 0")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = args.output / args.board_id / session_id
    video_path = session_dir / "recording.mp4"

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit(f"无法打开摄像头 {args.camera}；可用 --camera 指定设备编号")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    camera.set(cv2.CAP_PROP_FPS, args.fps)

    writer = None
    frames = 0
    started = None
    actual_width = actual_height = 0
    actual_fps = 0.0
    recording = False
    print("摄像头预览已打开：按空格开始录制，按 Q 退出。")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("摄像头读取失败，结束当前录制。")
                break
            if recording and writer is None:
                actual_height, actual_width = frame.shape[:2]
                actual_fps = camera.get(cv2.CAP_PROP_FPS) or args.fps
                session_dir.mkdir(parents=True, exist_ok=False)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(video_path), fourcc, actual_fps, (actual_width, actual_height))
                if not writer.isOpened():
                    raise RuntimeError(f"无法创建视频文件：{video_path}")
                started = time.monotonic()
            if recording:
                writer.write(frame)
                frames += 1
                hint = f"{args.board_id} | {args.condition} | Q: stop"
                color = (40, 220, 170)
            else:
                hint = f"{args.board_id} | {args.condition} | SPACE: start | Q: cancel"
                color = (0, 190, 255)
            cv2.putText(frame, hint, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            cv2.imshow("MotionMuse dataset capture", frame)
            key = cv2.waitKey(1) & 0xFF
            if not recording and key == ord(" "):
                recording = True
                print("开始录制：按 Q 结束并保存。")
            elif key in (ord("q"), ord("Q")):
                break
    finally:
        camera.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    if started is None:
        print("已退出，未开始录制，没有保存视频。")
        return 0
    duration = time.monotonic() - started
    if frames == 0:
        video_path.unlink(missing_ok=True)
        session_dir.rmdir()
        raise SystemExit("没有录到有效视频帧。")
    metadata = {
        "board_id": args.board_id, "session_id": session_id,
        "condition": args.condition, "camera_index": args.camera,
        "width": actual_width, "height": actual_height, "fps_requested": args.fps,
        "fps_reported": actual_fps, "frame_count": frames,
        "duration_seconds": round(duration, 3),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "video": video_path.name,
    }
    (session_dir / "session.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已保存 {frames} 帧，约 {duration:.1f} 秒：{video_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
