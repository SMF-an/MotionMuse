"""基于 KLT 光流和 RANSAC 单应性的普通板持续追踪。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import cv2
import numpy as np


@dataclass
class TrackingResult:
    corners_px: np.ndarray
    tracked_points: int


class BoardTracker:
    """初始四角由用户标定；之后在板面纹理与边缘上追踪特征点。"""
    MIN_POINTS = 12

    def __init__(self, frame: np.ndarray, corners_px: Iterable[Iterable[float]]):
        self.corners_px = np.asarray(tuple(corners_px), dtype=np.float32).reshape(4, 1, 2)
        self.previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.points = self._detect_points(self.previous_gray)
        if len(self.points) < self.MIN_POINTS:
            raise ValueError("板面可追踪特征过少；请确保四角清晰且板子与背景有对比")

    def _mask(self, shape: tuple[int, int]) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, self.corners_px.reshape(-1, 2).astype(np.int32), 255)
        return mask

    def _detect_points(self, gray: np.ndarray) -> np.ndarray:
        points = cv2.goodFeaturesToTrack(
            gray, maxCorners=150, qualityLevel=0.008, minDistance=7,
            mask=self._mask(gray.shape), blockSize=7,
        )
        return points if points is not None else np.empty((0, 1, 2), dtype=np.float32)

    def update(self, frame: np.ndarray) -> Optional[TrackingResult]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if len(self.points) < self.MIN_POINTS:
            self.points = self._detect_points(self.previous_gray)
        if len(self.points) < self.MIN_POINTS:
            return None

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.previous_gray, gray, self.points, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if next_points is None or status is None:
            return None
        keep = status.reshape(-1).astype(bool)
        old, new = self.points[keep], next_points[keep]
        if len(old) < self.MIN_POINTS:
            return None

        homography, inliers = cv2.findHomography(old, new, cv2.RANSAC, 3.0)
        if homography is None or inliers is None or int(inliers.sum()) < self.MIN_POINTS:
            return None
        self.corners_px = cv2.perspectiveTransform(self.corners_px, homography)
        self.previous_gray = gray
        self.points = new[inliers.reshape(-1).astype(bool)].reshape(-1, 1, 2)
        if len(self.points) < 45:
            fresh = self._detect_points(gray)
            if len(fresh):
                self.points = np.concatenate((self.points, fresh), axis=0)[:150]
        return TrackingResult(self.corners_px.copy(), int(inliers.sum()))
