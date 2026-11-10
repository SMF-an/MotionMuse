"""Generate printable MotionMuse ArUco markers (IDs 0--3)."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "assets" / "aruco_markers")
    parser.add_argument("--pixels", type=int, default=600, help="black-square side, excluding the white border")
    args = parser.parse_args()
    if args.pixels < 120:
        parser.error("--pixels must be at least 120")
    args.output.mkdir(parents=True, exist_ok=True)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    border = args.pixels // 6
    for marker_id in range(4):
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, args.pixels)
        printable = cv2.copyMakeBorder(marker, border, border, border, border, cv2.BORDER_CONSTANT, value=255)
        output = args.output / f"DICT_4X4_50_ID_{marker_id}.png"
        if not cv2.imwrite(str(output), printable):
            raise RuntimeError(f"无法写入 {output}")
        print(output)
    print("将四个标记贴在板角：上排 0、1；下排 2、3。黑色方块建议边长约 50 mm，并保留白边。")


if __name__ == "__main__":
    main()
