"""普通矩形板的坐标变换与虚拟吉他区域定义。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

CHORD_NAMES = ("C", "G", "Am", "F")


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class BoardCalibration:
    """四角顺序固定为左上、右上、右下、左下，坐标为画面归一化值。"""
    corners: tuple[Point, Point, Point, Point]

    @classmethod
    def load(cls, path: str | Path) -> "BoardCalibration":
        with Path(path).open(encoding="utf-8") as file:
            data = json.load(file)
        if data.get("coordinate_system") != "board_corners_tl_tr_br_bl":
            raise ValueError("不是普通板四角标定文件")
        raw = data.get("corners", [])
        if len(raw) != 4:
            raise ValueError("标定文件必须包含 4 个板角")
        corners = tuple(Point(float(p["x"]), float(p["y"])) for p in raw)
        if not all(0 <= p.x <= 1 and 0 <= p.y <= 1 for p in corners):
            raise ValueError("板角坐标必须在 0 到 1 之间")
        return cls(corners)  # type: ignore[arg-type]


class BoardGeometry:
    """通过单应性变换在图像坐标与板子局部坐标之间转换。"""
    def __init__(self, corners: Iterable[Point], frame_size: tuple[int, int]):
        corners = tuple(corners)
        if len(corners) != 4:
            raise ValueError("需要 4 个板角")
        width, height = frame_size
        source = np.float32([[p.x * width, p.y * height] for p in corners])
        board = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
        self.corners = corners
        self.frame_size = frame_size
        self.to_board = cv2.getPerspectiveTransform(source, board)
        self.to_image = cv2.getPerspectiveTransform(board, source)

    def image_to_board(self, point: Point) -> Point:
        width, height = self.frame_size
        value = cv2.perspectiveTransform(np.float32([[[point.x * width, point.y * height]]]), self.to_board)[0, 0]
        return Point(float(value[0]), float(value[1]))

    def board_to_image(self, point: Point) -> Point:
        width, height = self.frame_size
        value = cv2.perspectiveTransform(np.float32([[[point.x, point.y]]]), self.to_image)[0, 0]
        return Point(float(value[0] / width), float(value[1] / height))

    def chord_at(self, point: Point) -> Optional[str]:
        """板内左侧琴颈区：x∈[.04,.60]、y∈[.20,.80]。"""
        p = self.image_to_board(point)
        if not (0.04 <= p.x <= 0.60 and 0.20 <= p.y <= 0.80):
            return None
        index = min(int((p.x - 0.04) / 0.14), len(CHORD_NAMES) - 1)
        return CHORD_NAMES[index]

    def in_strum_band(self, point: Point) -> bool:
        p = self.image_to_board(point)
        return 0.68 <= p.x <= 0.90 and 0.15 <= p.y <= 0.85

    def strum_axis(self, point: Point) -> float:
        """扫弦轴是板内 y；跨过 .5 即为一次扫弦。"""
        return self.image_to_board(point).y

