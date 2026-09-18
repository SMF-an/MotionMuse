"""吉他标定几何模块。

读取网页标定页导出的 ``guitar_calibration.json``，把「琴头」与「琴颈-琴身连接点」
两个归一化坐标转换为琴颈方向、垂直方向、四个和弦段与扫弦带的几何，供演奏程序把
左右手关键点投影到和弦与扫弦事件上。

本模块只依赖 Python 标准库，独立于手部追踪代码；网页端 ``web/calibration.html``
中的同名常量需与这里保持一致。

坐标系约定：x、y 均为相对图像宽高的归一化值，范围 ``[0, 1]``，
x 向右、y 向下（与 OpenCV 图像坐标一致）。

.. note:: 镜像一致性
    手部追踪脚本（``hand_tracking_demo.py`` / ``hand_tracking_camera.py``）在推理前会执行
    ``cv2.flip(frame, 1)``，即在镜像后的画面上做检测。标定页默认按原始画面显示并保存坐标，
    因此接入时二者需对齐：要么在标定页预览同样镜像，要么在读取标定结果后对 x 取
    ``1 - x``，要么取消手部追踪脚本中的翻转。选择哪种方式应由集成阶段统一决定。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# ===== 几何常量（与 web/calibration.html 保持一致）=====
CHORD_NAMES: Tuple[str, ...] = ("C", "G", "Am", "F")  # 从琴头到琴身依次排列
NUM_CHORDS: int = len(CHORD_NAMES)
NECK_HALF_WIDTH: float = 0.04   # 和弦框沿 v 方向的半宽（归一化）
STRUM_OFFSET: float = 0.10      # 扫弦带中心相对琴身连接点沿 u 的偏移（归一化）
STRUM_HALF_LENGTH: float = 0.09  # 扫弦带半长轴，沿 v（归一化）
STRUM_HALF_WIDTH: float = 0.05   # 扫弦带半短轴，沿 u（归一化）

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "guitar_calibration.json"


@dataclass(frozen=True)
class Point:
    """归一化二维坐标点。"""
    x: float
    y: float


@dataclass(frozen=True)
class Calibration:
    """一次标定结果：琴头与琴颈-琴身连接点。"""
    neck_head: Point
    neck_body: Point

    @classmethod
    def from_dict(cls, data: dict) -> "Calibration":
        head = data["neck_head"]
        body = data["neck_body"]
        return cls(
            Point(float(head["x"]), float(head["y"])),
            Point(float(body["x"]), float(body["y"])),
        )

    @classmethod
    def load(cls, path) -> "Calibration":
        """从 JSON 文件读取标定结果。"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


class NeckGeometry:
    """由标定结果推导出的琴颈/扫弦带几何，并提供投影方法。"""

    def __init__(self, calibration: Calibration):
        head = calibration.neck_head
        body = calibration.neck_body

        dx = body.x - head.x
        dy = body.y - head.y
        length = math.hypot(dx, dy)
        if length <= 0.0:
            raise ValueError("琴头与琴身连接点重合，无法确定琴颈方向")

        self.head = head
        self.body = body
        self.length = length

        # 琴颈方向单位向量 u：从琴头指向琴身连接点
        self.u = Point(dx / length, dy / length)
        # 垂直方向 v = (-u_y, u_x)，用于定义扫弦带长轴
        self.v = Point(-self.u.y, self.u.x)

        # 四个和弦段沿 u 均分琴颈有效长度
        self.segment_length = length / NUM_CHORDS

        # 扫弦带中心：位于琴身连接点沿 u 方向偏移 STRUM_OFFSET 处
        self.strum_center = Point(
            body.x + self.u.x * STRUM_OFFSET,
            body.y + self.u.y * STRUM_OFFSET,
        )

    def project(self, point: Point) -> float:
        """返回点相对琴头沿 u 的标量投影，范围约 0（琴头）到 length（琴身）。"""
        return (
            (point.x - self.head.x) * self.u.x
            + (point.y - self.head.y) * self.u.y
        )

    def chord_at(self, point: Point) -> Optional[str]:
        """把点投影到琴颈，返回对应和弦名；不在琴颈上则返回 None。"""
        t = self.project(point)
        if t < 0.0 or t > self.length:
            return None
        index = min(int(t // self.segment_length), NUM_CHORDS - 1)
        return CHORD_NAMES[index]

    def strum_coord(self, point: Point) -> float:
        """返回点相对扫弦带中心沿 v 的偏移，用于判断上扫/下扫。"""
        return (
            (point.x - self.strum_center.x) * self.v.x
            + (point.y - self.strum_center.y) * self.v.y
        )

    def strum_u_coord(self, point: Point) -> float:
        """返回点相对扫弦带中心沿 u 的偏移，用于判断是否位于扫弦带内。"""
        return (
            (point.x - self.strum_center.x) * self.u.x
            + (point.y - self.strum_center.y) * self.u.y
        )

    def in_strum_band(self, point: Point) -> bool:
        """判断点是否落在扫弦带椭圆内。"""
        along_u = self.strum_u_coord(point)
        along_v = self.strum_coord(point)
        return (
            (along_u / STRUM_HALF_WIDTH) ** 2
            + (along_v / STRUM_HALF_LENGTH) ** 2
        ) <= 1.0

    def chord_centers(self) -> List[Tuple[str, Point]]:
        """返回各和弦段中心的归一化坐标，可用于界面叠加绘制。"""
        centers = []
        for i in range(NUM_CHORDS):
            t = (i + 0.5) * self.segment_length
            centers.append((
                CHORD_NAMES[i],
                Point(
                    self.head.x + self.u.x * t,
                    self.head.y + self.u.y * t,
                ),
            ))
        return centers


def strum_direction(previous: Point, current: Point, geometry: NeckGeometry) -> Optional[str]:
    """判断右手食指是否沿 v 穿过扫弦带中线（v=0），返回 "DOWN"/"UP"/None。

    与手部追踪脚本中的 ``StrumDetector`` 语义一致：v 增大视为下扫（DOWN），
    v 减小视为上扫（UP）。调用方通常还需配合 ``in_strum_band`` 过滤误触发。
    """
    prev_v = geometry.strum_coord(previous)
    curr_v = geometry.strum_coord(current)
    delta_v = curr_v - prev_v

    if prev_v < 0.0 <= curr_v and delta_v > 0.0:
        return "DOWN"
    if curr_v <= 0.0 < prev_v and delta_v < 0.0:
        return "UP"
    return None


def _validate_normalized(calibration: Calibration) -> None:
    """校验所有坐标都在 [0, 1] 区间内，否则抛出异常。"""
    for name, point in (("neck_head", calibration.neck_head), ("neck_body", calibration.neck_body)):
        if not (0.0 <= point.x <= 1.0 and 0.0 <= point.y <= 1.0):
            raise ValueError(f"{name} 坐标超出 [0, 1] 范围：({point.x}, {point.y})")


def main() -> None:
    """演示：读取标定 JSON、打印坐标并校验、输出推导几何。"""
    if not DEFAULT_CONFIG.exists():
        raise FileNotFoundError(f"找不到标定配置文件：{DEFAULT_CONFIG}")

    calibration = Calibration.load(DEFAULT_CONFIG)
    _validate_normalized(calibration)

    print("已读取标定配置：", DEFAULT_CONFIG)
    print(f"  neck_head = ({calibration.neck_head.x:.4f}, {calibration.neck_head.y:.4f})")
    print(f"  neck_body = ({calibration.neck_body.x:.4f}, {calibration.neck_body.y:.4f})")
    print("坐标校验：所有数值均在 [0, 1] 范围内 [OK]")

    geometry = NeckGeometry(calibration)
    print(f"  琴颈方向 u = ({geometry.u.x:.4f}, {geometry.u.y:.4f})")
    print(f"  垂直方向 v = ({geometry.v.x:.4f}, {geometry.v.y:.4f})")
    print(f"  琴颈长度 = {geometry.length:.4f}")
    print(f"  扫弦带中心 = ({geometry.strum_center.x:.4f}, {geometry.strum_center.y:.4f})")

    print("  和弦段：")
    for name, center in geometry.chord_centers():
        print(f"    {name:<3} 中心 = ({center.x:.4f}, {center.y:.4f})")


if __name__ == "__main__":
    main()
