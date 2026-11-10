"""Markerless board localisation with per-frame contour re-detection.

Optical flow is only a short-lived bridge from the last *detected* board pose;
it is never chained from one estimated pose to the next.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class MarkerlessPose:
    status: str
    reason: str
    corners: list[dict[str, float]] | None
    confidence: float
    age_s: float | None

    def to_dict(self) -> dict:
        return {"status": self.status, "reason": self.reason,
                "corners": self.corners, "confidence": round(self.confidence, 3),
                "age_s": None if self.age_s is None else round(self.age_s, 3)}


class MarkerlessBoardTracker:
    """Detect a calibrated quadrilateral anew on every frame.

    Works best when the board has a visible rectangular outline and an
    appearance distinct from the surrounding scene. No generic single-camera
    method can guarantee reacquisition of an arbitrary occluded board.
    """

    def __init__(self, bridge_s: float = 0.28) -> None:
        self.bridge_s = bridge_s
        self.reference_hist: np.ndarray | None = None
        self.reference_aspect = 1.0
        self.last_detected: np.ndarray | None = None
        self.anchor_gray: np.ndarray | None = None
        self.anchor_features: np.ndarray | None = None
        self.last_detection_at: float | None = None
        self.frame_shape: tuple[int, int] | None = None

    def reset(self) -> None:
        self.__init__(self.bridge_s)

    @staticmethod
    def _valid(quad: np.ndarray, shape: tuple[int, int]) -> bool:
        h, w = shape
        p = quad.astype(np.float32)
        if not np.isfinite(p).all() or not cv2.isContourConvex(p):
            return False
        area = abs(cv2.contourArea(p))
        if area < h * w * .012 or area > h * w * .85:
            return False
        lengths = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
        if min(lengths) < min(h, w) * .045:
            return False
        edges = np.roll(p, -1, axis=0) - p
        following = np.roll(edges, -1, axis=0)
        cross = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
        return bool(np.all(np.abs(cross) > .07 * lengths * np.roll(lengths, -1)))

    @staticmethod
    def _aspect(p: np.ndarray) -> float:
        e = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
        return float((e[0] + e[2]) / max(e[1] + e[3], 1))

    @staticmethod
    def _appearance(frame: np.ndarray, quad: np.ndarray) -> np.ndarray:
        target = np.float32([[0, 0], [127, 0], [127, 63], [0, 63]])
        patch = cv2.warpPerspective(frame, cv2.getPerspectiveTransform(quad, target), (128, 64))
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        # Ignore the outer edge, where the background and fingers can leak in.
        roi = hsv[7:-7, 12:-12]
        hist = cv2.calcHist([roi], [0, 1], None, [24, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=1, norm_type=cv2.NORM_L1)
        return hist

    @staticmethod
    def _orient(quad: np.ndarray, reference: np.ndarray) -> np.ndarray:
        # Corner indices stay attached to the board, not to image left/right.
        options = [quad[list(p)] for p in permutations(range(4))
                   if cv2.isContourConvex(quad[list(p)].astype(np.float32))]
        return min(options, key=lambda p: np.mean(np.linalg.norm(p - reference, axis=1)))

    def calibrate(self, frame: np.ndarray, normalized: list[dict[str, float]],
                  captured_at: float | None = None) -> None:
        if len(normalized) != 4:
            raise ValueError("请依次标出板子的四个角")
        h, w = frame.shape[:2]
        quad = np.float32([[p["x"] * w, p["y"] * h] for p in normalized])
        if not np.all((quad[:, 0] >= 0) & (quad[:, 0] <= w) &
                      (quad[:, 1] >= 0) & (quad[:, 1] <= h)) or not self._valid(quad, (h, w)):
            raise ValueError("四角无效：请按板面顺时针依次点击，且不要交叉")
        self.reference_hist = self._appearance(frame, quad)
        self.reference_aspect = self._aspect(quad)
        self.frame_shape = (h, w)
        self._accept(frame, quad, time.monotonic() if captured_at is None else captured_at)

    def _accept(self, frame: np.ndarray, quad: np.ndarray, now: float) -> None:
        self.last_detected = quad.copy()
        self.last_detection_at = now
        self.anchor_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros(self.anchor_gray.shape, np.uint8)
        cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
        mask = cv2.erode(mask, np.ones((9, 9), np.uint8))
        self.anchor_features = cv2.goodFeaturesToTrack(
            self.anchor_gray, maxCorners=120, qualityLevel=.01,
            minDistance=7, mask=mask)

    def _candidates(self, frame: np.ndarray) -> list[np.ndarray]:
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (max(1, w // 2), max(1, h // 2)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 45, 135)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        found: list[np.ndarray] = []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:60]:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, .025 * perimeter, True)
            if len(approx) != 4:
                continue
            quad = approx.reshape(4, 2).astype(np.float32) * np.float32([w / small.shape[1], h / small.shape[0]])
            if self._valid(quad, (h, w)):
                found.append(quad)
        return found

    def _flow_bridge(self, frame: np.ndarray) -> np.ndarray | None:
        if self.anchor_features is None or len(self.anchor_features) < 12 or self.anchor_gray is None:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        moved, forward, _ = cv2.calcOpticalFlowPyrLK(
            self.anchor_gray, gray, self.anchor_features, None,
            winSize=(25, 25), maxLevel=4)
        if moved is None or forward is None:
            return None
        back, backward, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.anchor_gray, moved, None, winSize=(25, 25), maxLevel=4)
        if back is None or backward is None:
            return None
        valid = (forward.ravel() == 1) & (backward.ravel() == 1)
        valid &= np.linalg.norm(back.reshape(-1, 2) - self.anchor_features.reshape(-1, 2), axis=1) < 2.5
        if valid.sum() < 12:
            return None
        matrix, inliers = cv2.findHomography(self.anchor_features[valid], moved[valid], cv2.RANSAC, 3.0)
        if matrix is None or inliers is None or inliers.sum() < 10:
            return None
        quad = cv2.perspectiveTransform(self.last_detected.reshape(1, 4, 2), matrix)[0]
        return quad if self._valid(quad, frame.shape[:2]) else None

    def process(self, frame: np.ndarray) -> MarkerlessPose:
        if self.reference_hist is None or self.last_detected is None:
            return MarkerlessPose("uncalibrated", "请先在静止画面上标定四角", None, 0, None)
        now = time.monotonic()
        h, w = frame.shape[:2]
        if self.frame_shape != (h, w):
            self.reset()
            return MarkerlessPose("uncalibrated", "摄像头分辨率变化，请重新标定", None, 0, None)
        age = now - self.last_detection_at
        predicted = self._flow_bridge(frame) if age < self.bridge_s else None
        reference = predicted if predicted is not None else self.last_detected
        best: tuple[float, np.ndarray] | None = None
        for candidate in self._candidates(frame):
            quad = self._orient(candidate, reference)
            aspect = self._aspect(quad)
            ratio = min(aspect / self.reference_aspect, self.reference_aspect / max(aspect, .001))
            if ratio < .45:
                continue
            similarity = 1 - cv2.compareHist(self.reference_hist, self._appearance(frame, quad), cv2.HISTCMP_BHATTACHARYYA)
            travel = np.mean(np.linalg.norm(quad - reference, axis=1)) / max(h, w)
            score = .7 * similarity + .3 * ratio - min(travel, 1) * (.25 if age < self.bridge_s else .08)
            if best is None or score > best[0]:
                best = (score, quad)
        if best is not None and best[0] >= .67:
            self._accept(frame, best[1], now)
            return self._pose("valid", "逐帧轮廓定位", best[1], best[0], 0, w, h)
        if predicted is not None:
            return self._pose("bridged", "短暂遮挡：基于最近检测帧预测", predicted, .45, age, w, h)
        return MarkerlessPose("lost", "轮廓或外观不匹配；请让板子完整入镜，必要时重新标定", None, 0, age)

    @staticmethod
    def _pose(status: str, reason: str, quad: np.ndarray, confidence: float,
              age: float, w: int, h: int) -> MarkerlessPose:
        return MarkerlessPose(status, reason,
                              [{"x": float(x / w), "y": float(y / h)} for x, y in quad],
                              confidence, age)
