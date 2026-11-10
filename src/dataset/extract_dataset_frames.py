"""Sample frames from capture_dataset.py sessions and create a CVAT manifest."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/frames"))
    parser.add_argument("--interval", type=float, default=1.0, help="抽帧间隔秒数，默认 1 秒")
    parser.add_argument("--max-frames-per-clip", type=int, default=150)
    args = parser.parse_args()
    if args.interval <= 0 or args.max_frames_per_clip <= 0:
        parser.error("interval 和 max-frames-per-clip 必须大于 0")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.csv"
    count = 0

    with manifest_path.open("w", newline="", encoding="utf-8-sig") as manifest_file:
        fields = ["image", "board_id", "session_id", "condition", "timestamp_seconds", "frame_index"]
        writer = csv.DictWriter(manifest_file, fieldnames=fields)
        writer.writeheader()
        for sidecar in sorted(args.raw.rglob("session.json")):
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            video = sidecar.parent / meta["video"]
            cap = cv2.VideoCapture(str(video))
            if not cap.isOpened():
                raise SystemExit(f"无法读取视频：{video}")
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = float(meta.get("fps_reported") or meta.get("fps_requested") or 30)
            frame_index = 0
            next_sample = 0.0
            clip_count = 0
            while clip_count < args.max_frames_per_clip:
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp = frame_index / fps
                if timestamp + 1e-6 >= next_sample:
                    name = f"{meta['board_id']}_{meta['session_id']}_{frame_index:07d}.jpg"
                    image_path = args.output / name
                    if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                        raise RuntimeError(f"无法写入图像：{image_path}")
                    writer.writerow({
                        "image": name, "board_id": meta["board_id"],
                        "session_id": meta["session_id"], "condition": meta["condition"],
                        "timestamp_seconds": f"{timestamp:.3f}", "frame_index": frame_index,
                    })
                    count += 1
                    clip_count += 1
                    next_sample += args.interval
                frame_index += 1
            cap.release()
            print(f"{meta['session_id']}: 抽取 {clip_count} 帧")
    print(f"共抽取 {count} 帧；清单：{manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
