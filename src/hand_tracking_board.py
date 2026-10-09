"""普通板吉他演示：四角标定 + 板子追踪 + 双手交互。

先在 web/app.html 导出 board_calibration.json，放到 config/ 目录，
再运行本文件。该脚本不镜像画面，确保与网页标定坐标一致。
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from board_geometry import BoardCalibration, BoardGeometry, Point
from board_tracker import BoardTracker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
CALIBRATION_PATH = PROJECT_ROOT / "config" / "board_calibration.json"
VIDEO_PATH = PROJECT_ROOT / "assets" / "test_video_2.mp4"

CHORD_HAND = "Left"
STRUM_HAND = "Right"
CHORD_STABLE_FRAMES = 4
STRUM_COOLDOWN_MS = 180
MIN_STRUM_DELTA = 0.018  # 板内归一化 y 位移


def normalized_corners(corners_px: np.ndarray, width: int, height: int) -> list[Point]:
    return [Point(float(x / width), float(y / height)) for x, y in corners_px.reshape(4, 2)]


def draw_board(frame: np.ndarray, geometry: BoardGeometry, active_chord: str) -> None:
    height, width = frame.shape[:2]
    corners = np.array([[p.x * width, p.y * height] for p in geometry.corners], np.int32)
    cv2.polylines(frame, [corners], True, (80, 214, 176), 2)

    # 四段和弦区：板内左侧 x=.04~.60、y=.20~.80。
    for index, name in enumerate(("C", "G", "Am", "F")):
        x0, x1 = 0.04 + index * 0.14, 0.18 + index * 0.14
        local = [Point(x0, .20), Point(x1, .20), Point(x1, .80), Point(x0, .80)]
        image = [geometry.board_to_image(p) for p in local]
        polygon = np.array([[p.x * width, p.y * height] for p in image], np.int32)
        color = (0, 230, 255) if name == active_chord else (80, 214, 176)
        cv2.polylines(frame, [polygon], True, color, 2)
        center = geometry.board_to_image(Point((x0 + x1) / 2, .50))
        cv2.putText(frame, name, (int(center.x * width - 12), int(center.y * height + 7)), cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2)

    # 扫弦带中心线，板内 x=.78、y=.15~.85。
    start, end = geometry.board_to_image(Point(.78, .15)), geometry.board_to_image(Point(.78, .85))
    cv2.line(frame, (int(start.x * width), int(start.y * height)), (int(end.x * width), int(end.y * height)), (78, 174, 243), 4)


def main() -> None:
    if not MODEL_PATH.exists() or not CALIBRATION_PATH.exists() or not VIDEO_PATH.exists():
        raise FileNotFoundError("请检查模型、config/board_calibration.json 和测试视频是否存在")
    calibration = BoardCalibration.load(CALIBRATION_PATH)
    video = cv2.VideoCapture(str(VIDEO_PATH))
    ok, first_frame = video.read()
    if not ok:
        raise RuntimeError("无法读取测试视频")
    height, width = first_frame.shape[:2]
    initial_corners = np.array([[p.x * width, p.y * height] for p in calibration.corners], np.float32)
    tracker = BoardTracker(first_frame, initial_corners)
    fps = video.get(cv2.CAP_PROP_FPS) or 30.0

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=.6,
        min_hand_presence_confidence=.6,
        min_tracking_confidence=.6,
    )
    current_chord, candidate, candidate_count = "None", None, 0
    previous_strum, last_strum_at = None, -STRUM_COOLDOWN_MS
    frame_index, fps_start, fps_count, display_fps = 0, time.perf_counter(), 0, 0.0

    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        frame = first_frame
        while True:
            if frame_index > 0:
                ok, frame = video.read()
                if not ok:
                    break
            tracking = tracker.update(frame) if frame_index else None
            corners = normalized_corners(tracking.corners_px if tracking else initial_corners, width, height)
            geometry = BoardGeometry(corners, (width, height))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(frame_index * 1000 / fps))
            hands = {handedness[0].category_name: landmarks for landmarks, handedness in zip(result.hand_landmarks, result.handedness)}

            if CHORD_HAND in hands:
                wrist = hands[CHORD_HAND][0]
                detected = geometry.chord_at(Point(wrist.x, wrist.y))
                candidate_count = candidate_count + 1 if detected == candidate else 1
                candidate = detected
                if detected and candidate_count >= CHORD_STABLE_FRAMES:
                    current_chord = detected

            now_ms = int(frame_index * 1000 / fps)
            if STRUM_HAND in hands:
                tip = hands[STRUM_HAND][8]
                point = Point(tip.x, tip.y)
                if previous_strum and geometry.in_strum_band(point) and now_ms - last_strum_at >= STRUM_COOLDOWN_MS:
                    previous_y, current_y = geometry.strum_axis(previous_strum), geometry.strum_axis(point)
                    if previous_y < .5 <= current_y and current_y - previous_y >= MIN_STRUM_DELTA:
                        print(f"[Strum] DOWN | chord={current_chord}")
                        last_strum_at = now_ms
                    elif current_y <= .5 < previous_y and previous_y - current_y >= MIN_STRUM_DELTA:
                        print(f"[Strum] UP   | chord={current_chord}")
                        last_strum_at = now_ms
                previous_strum = point
            else:
                previous_strum = None

            draw_board(frame, geometry, current_chord)
            cv2.putText(frame, f"Chord: {current_chord}", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, .9, (0, 230, 255), 2)
            status = f"Board tracking: {tracking.tracked_points if tracking else 'initial'} points"
            cv2.putText(frame, status, (20, 68), cv2.FONT_HERSHEY_SIMPLEX, .55, (80, 214, 176), 2)
            fps_count += 1
            elapsed = time.perf_counter() - fps_start
            if elapsed >= 1:
                display_fps, fps_count, fps_start = fps_count / elapsed, 0, time.perf_counter()
            cv2.putText(frame, f"FPS: {display_fps:.1f}", (width - 130, 38), cv2.FONT_HERSHEY_SIMPLEX, .65, (80, 214, 176), 2)
            cv2.imshow("MotionMuse - Board Guitar", frame)
            frame_index += 1
            if cv2.waitKey(max(1, int(1000 / fps))) & 0xFF == ord("q"):
                break
    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
