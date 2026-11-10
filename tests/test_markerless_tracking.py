"""Synthetic checks for the no-marker board locator."""
import unittest

import cv2
import numpy as np

from serving.markerless_tracking import MarkerlessBoardTracker


def scene(dx: int = 0) -> np.ndarray:
    frame = np.full((480, 640, 3), (30, 30, 30), dtype=np.uint8)
    cv2.rectangle(frame, (170 + dx, 150), (430 + dx, 310), (30, 60, 190), -1)
    cv2.rectangle(frame, (170 + dx, 150), (430 + dx, 310), (240, 240, 240), 3)
    cv2.putText(frame, "GUITAR", (210 + dx, 240), cv2.FONT_HERSHEY_SIMPLEX,
                .8, (250, 250, 250), 2)
    return frame


class MarkerlessTrackingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = MarkerlessBoardTracker()
        self.tracker.calibrate(scene(), [
            {"x": 170 / 640, "y": 150 / 480},
            {"x": 430 / 640, "y": 150 / 480},
            {"x": 430 / 640, "y": 310 / 480},
            {"x": 170 / 640, "y": 310 / 480},
        ])

    def test_four_corner_calibration_is_valid(self) -> None:
        self.assertIsNotNone(self.tracker.reference_hist)
        self.assertEqual(self.tracker.last_detected.shape, (4, 2))

    def test_redetects_moved_board_without_chaining_flow(self) -> None:
        pose = self.tracker.process(scene(35))
        self.assertEqual(pose.status, "valid")
        self.assertAlmostEqual(pose.corners[0]["x"], 205 / 640, delta=.04)

    def test_stops_reporting_pose_after_bridge_window(self) -> None:
        self.tracker.last_detection_at -= 1
        pose = self.tracker.process(np.full((480, 640, 3), 30, np.uint8))
        self.assertEqual(pose.status, "lost")
        self.assertIsNone(pose.corners)


if __name__ == "__main__":
    unittest.main()
