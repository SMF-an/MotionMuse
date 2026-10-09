"""ArUco-driven board localisation for MotionMuse.

The playing plane is defined by four DICT_4X4_50 markers.  Their *centres*
must be installed as: ID 0 / 1 on the top edge and ID 2 / 3 on the bottom
edge.  Every valid frame computes a fresh homography, so it does not inherit
optical-flow drift from previous frames.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import time
from typing import Any

import cv2
import numpy as np


REQUIRED_IDS = (0, 1, 2, 3)
BOARD_UV = np.float32(((0, 0), (1, 0), (0, 1), (1, 1)))
PERIMETER = (0, 1, 3, 2)


@dataclass(frozen=True)
class BoardPose:
    status: str
    reason: str | None
    corners_px: list[list[float]] | None
    age_s: float | None
    visible_ids: list[int]
    aruco_ms: float

    def to_dict(self, width: int, height: int) -> dict[str, Any]:
        corners = None
        if self.corners_px is not None:
            corners = [
                {"x": x / width, "y": y / height}
                for x, y in self.corners_px
            ]
        return {
            "status": self.status,
            "reason": self.reason,
            "corners": corners,
            "age_s": self.age_s,
            "visible_ids": self.visible_ids,
            "aruco_ms": self.aruco_ms,
        }


def _valid_quad(points: np.ndarray, min_area_px: float) -> bool:
    polygon = points[list(PERIMETER)].astype(np.float32)
    if not cv2.isContourConvex(polygon):
        return False
    if abs(cv2.contourArea(polygon)) < min_area_px:
        return False
    edges = np.roll(polygon, -1, axis=0) - polygon
    following = np.roll(edges, -1, axis=0)
    cross = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    lengths = np.linalg.norm(edges, axis=1) * np.linalg.norm(following, axis=1)
    return bool(np.all(lengths > 0) and np.min(np.abs(cross) / lengths) >= 0.01)


class ArucoBoardTracker:
    """Stateless-per-frame detection with a short, explicit occlusion hold."""

    def __init__(self, hold_s: float = 0.25, min_area_px: float = 400.0) -> None:
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("当前 OpenCV 没有 aruco 模块；请安装 opencv-python >= 4.7")
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        self.hold_s = hold_s
        self.min_area_px = min_area_px
        self._last_corners: np.ndarray | None = None
        self._last_valid_at: float | None = None

    def reset(self) -> None:
        self._last_corners = None
        self._last_valid_at = None

    def process(self, frame_bgr: np.ndarray) -> BoardPose:
        started = time.perf_counter()
        now = time.monotonic()
        detected, ids, _ = self.detector.detectMarkers(frame_bgr)
        centres: dict[int, np.ndarray] = {}
        counts: Counter[int] = Counter()
        for marker_id, marker_corners in zip(() if ids is None else ids.flatten(), detected):
            marker_id = int(marker_id)
            counts[marker_id] += 1
            if marker_id in REQUIRED_IDS:
                centres[marker_id] = np.asarray(marker_corners, dtype=np.float32).reshape(4, 2).mean(axis=0)
        visible = sorted(counts)
        elapsed = lambda: round((time.perf_counter() - started) * 1000, 3)

        if any(counts[marker_id] > 1 for marker_id in REQUIRED_IDS):
            self.reset()
            return BoardPose("lost", "duplicate_required_ids", None, None, visible, elapsed())
        if all(counts[marker_id] == 1 for marker_id in REQUIRED_IDS):
            ordered = np.float32([centres[marker_id] for marker_id in REQUIRED_IDS])
            if not _valid_quad(ordered, self.min_area_px):
                self.reset()
                return BoardPose("lost", "invalid_marker_layout", None, None, visible, elapsed())
            matrix = cv2.getPerspectiveTransform(ordered, BOARD_UV)
            if not np.isfinite(matrix).all():
                self.reset()
                return BoardPose("lost", "invalid_homography", None, None, visible, elapsed())
            # UI polygon order is TL, TR, BR, BL; marker-ID order is 0, 1, 2, 3.
            self._last_corners = ordered[list(PERIMETER)]
            self._last_valid_at = now
            return BoardPose("valid", None, self._last_corners.tolist(), 0.0, visible, elapsed())
        if self._last_corners is not None and self._last_valid_at is not None:
            age = now - self._last_valid_at
            if age <= self.hold_s:
                return BoardPose("held", "temporary_marker_occlusion", self._last_corners.tolist(), age, visible, elapsed())
        self.reset()
        return BoardPose("lost", "missing_markers", None, None, visible, elapsed())
