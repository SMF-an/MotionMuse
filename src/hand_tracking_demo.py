from pathlib import Path
import time

import cv2
import mediapipe as mp


# ===== 路径配置 =====
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
VIDEO_PATH = PROJECT_ROOT / "assets" / "test_video_2.mp4"

# ===== 交互配置 =====
# 如果视频经过镜像翻转后，MediaPipe 标签与实际左右手不一致，可改为 "Right"
CHORD_HAND_LABEL = "Left"
STRUM_HAND_LABEL = "Right"

# 左手连续多少帧位于同一个和弦区，才确认切换和弦
CHORD_STABLE_FRAMES = 5

# 两次扫弦的最短间隔，避免一次动作触发多次
STRUM_COOLDOWN_MS = 180

# 单帧最小纵向位移；数值使用归一化坐标
MIN_STRUM_DELTA_Y = 0.012

# 虚拟琴颈：4 个和弦选择区域，格式为 (x1, y1, x2, y2)
# 坐标比例相对于整个画面，范围为 0.0~1.0
CHORD_ZONES = {
    "C":  (0.05, 0.12, 0.25, 0.32),
    "G":  (0.28, 0.12, 0.48, 0.32),
    "Am": (0.05, 0.36, 0.25, 0.56),
    "F":  (0.28, 0.36, 0.48, 0.56),
}

# 虚拟琴弦区域。扫弦时，右手食指应穿过其水平中线。
STRUM_ZONE = (0.48, 0.22, 0.90, 0.82)


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def point_in_rect(x, y, rect):
    """判断归一化坐标点是否位于矩形区域内。"""
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def draw_normalized_rect(frame, rect, color, text=None):
    """在 OpenCV 图像上绘制归一化坐标矩形。"""
    height, width, _ = frame.shape
    x1, y1, x2, y2 = rect

    start = (int(x1 * width), int(y1 * height))
    end = (int(x2 * width), int(y2 * height))

    cv2.rectangle(frame, start, end, color, 2)

    if text:
        cv2.putText(
            frame,
            text,
            (start[0] + 8, start[1] + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )


def draw_hand(frame, landmarks, label):
    """绘制手部 21 个关键点、骨架和左右手标签。"""
    height, width, _ = frame.shape

    points = []
    for landmark in landmarks:
        points.append((
            int(landmark.x * width),
            int(landmark.y * height),
        ))

    for start_index, end_index in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start_index],
            points[end_index],
            (0, 255, 0),
            2,
        )

    for index, point in enumerate(points):
        # 手腕和各手指指尖用红色突出显示
        color = (0, 0, 255) if index in (0, 4, 8, 12, 16, 20) else (255, 0, 0)
        cv2.circle(frame, point, 4, color, -1)

    wrist_x, wrist_y = points[0]
    cv2.putText(
        frame,
        label,
        (wrist_x, wrist_y - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )


class ChordSelector:
    """根据左手位置稳定地选择和弦，避免关键点抖动导致频繁切换。"""

    def __init__(self):
        self.current_chord = "None"
        self.candidate_chord = None
        self.candidate_count = 0

    def detect_zone_chord(self, x, y):
        for chord, rect in CHORD_ZONES.items():
            if point_in_rect(x, y, rect):
                return chord
        return None

    def update(self, x, y):
        detected_chord = self.detect_zone_chord(x, y)

        if detected_chord == self.candidate_chord:
            self.candidate_count += 1
        else:
            self.candidate_chord = detected_chord
            self.candidate_count = 1

        changed = False

        if (
            detected_chord is not None
            and self.candidate_count >= CHORD_STABLE_FRAMES
            and detected_chord != self.current_chord
        ):
            self.current_chord = detected_chord
            changed = True

        return self.current_chord, changed


class StrumDetector:
    """根据右手食指越过琴弦区域中线的动作识别上下扫弦。"""

    def __init__(self):
        self.previous_point = None
        self.previous_timestamp_ms = None
        self.last_strum_timestamp_ms = -STRUM_COOLDOWN_MS

    def update(self, x, y, timestamp_ms):
        event = None
        speed = 0.0

        if self.previous_point is not None and self.previous_timestamp_ms is not None:
            previous_x, previous_y = self.previous_point
            delta_y = y - previous_y

            delta_time_s = max(
                (timestamp_ms - self.previous_timestamp_ms) / 1000.0,
                0.001,
            )
            speed = abs(delta_y) / delta_time_s

            x1, y1, x2, y2 = STRUM_ZONE
            center_y = (y1 + y2) / 2

            within_strum_x = x1 <= x <= x2
            cooldown_finished = (
                timestamp_ms - self.last_strum_timestamp_ms >= STRUM_COOLDOWN_MS
            )

            # 从中线上方穿到下方：下扫
            is_down_strum = (
                previous_y < center_y <= y
                and delta_y > MIN_STRUM_DELTA_Y
            )

            # 从中线下方穿到上方：上扫
            is_up_strum = (
                y <= center_y < previous_y
                and delta_y < -MIN_STRUM_DELTA_Y
            )

            if within_strum_x and cooldown_finished:
                if is_down_strum:
                    event = "DOWN"
                elif is_up_strum:
                    event = "UP"

                if event is not None:
                    self.last_strum_timestamp_ms = timestamp_ms

        self.previous_point = (x, y)
        self.previous_timestamp_ms = timestamp_ms

        return event, speed

    def reset(self):
        self.previous_point = None
        self.previous_timestamp_ms = None


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到模型文件：{MODEL_PATH}")

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"找不到测试视频：{VIDEO_PATH}")

    video = cv2.VideoCapture(str(VIDEO_PATH))
    if not video.isOpened():
        raise RuntimeError(f"无法打开视频：{VIDEO_PATH}")

    video_fps = video.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0

    frame_delay_ms = max(1, int(1000 / video_fps))
    frame_index = 0

    chord_selector = ChordSelector()
    strum_detector = StrumDetector()

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    fps_start_time = time.perf_counter()
    fps_frame_count = 0
    display_fps = 0.0

    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = video.read()
            if not ok:
                print("视频播放结束。")
                break

            # 若测试视频已是镜像画面，可删除这一行
            frame = cv2.flip(frame, 1)

            height, width, _ = frame.shape

            # 绘制和弦区
            for chord, rect in CHORD_ZONES.items():
                color = (0, 255, 255) if chord == chord_selector.current_chord else (120, 120, 120)
                draw_normalized_rect(frame, rect, color, chord)

            # 绘制琴弦区和琴弦中线
            draw_normalized_rect(frame, STRUM_ZONE, (255, 255, 0), "STRUM AREA")

            x1, y1, x2, y2 = STRUM_ZONE
            center_y = int(((y1 + y2) / 2) * height)
            cv2.line(
                frame,
                (int(x1 * width), center_y),
                (int(x2 * width), center_y),
                (255, 255, 0),
                2,
            )

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            timestamp_ms = int(frame_index * 1000 / video_fps)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            hands = {}

            for landmarks, handedness in zip(
                result.hand_landmarks,
                result.handedness,
            ):
                label = handedness[0].category_name
                hands[label] = landmarks
                draw_hand(frame, landmarks, label)

            # 左手：通过手腕坐标选择和弦
            if CHORD_HAND_LABEL in hands:
                chord_hand = hands[CHORD_HAND_LABEL]
                wrist = chord_hand[0]

                current_chord, chord_changed = chord_selector.update(
                    wrist.x,
                    wrist.y,
                )

                if chord_changed:
                    print(f"[Chord] {current_chord}")

            # 右手：使用食指指尖坐标识别扫弦
            if STRUM_HAND_LABEL in hands:
                strum_hand = hands[STRUM_HAND_LABEL]
                index_tip = strum_hand[8]

                strum_event, speed = strum_detector.update(
                    index_tip.x,
                    index_tip.y,
                    timestamp_ms,
                )

                if strum_event is not None:
                    print(
                        f"[Strum] {strum_event:<4} | "
                        f"chord={chord_selector.current_chord:<4} | "
                        f"speed={speed:.2f}"
                    )

                    # 下一步接音频时，在这里调用播放函数：
                    # play_chord(chord_selector.current_chord, strum_event, speed)
            else:
                strum_detector.reset()

            # 显示状态文字
            cv2.putText(
                frame,
                f"Chord: {chord_selector.current_chord}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                "Press Q to quit",
                (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

            # 计算实际处理帧率
            fps_frame_count += 1
            elapsed = time.perf_counter() - fps_start_time
            if elapsed >= 1.0:
                display_fps = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_start_time = time.perf_counter()

            cv2.putText(
                frame,
                f"FPS: {display_fps:.1f}",
                (width - 150, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.imshow("MotionMuse - Gesture Test", frame)

            frame_index += 1

            if cv2.waitKey(frame_delay_ms) & 0xFF == ord("q"):
                break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()