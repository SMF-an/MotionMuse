# MotionMuse 幻奏

基于板子定位、MediaPipe Hand 和 OpenCV 的虚拟吉他原型。网页演奏台可切换 ArUco 与无标记板子两种模式。

## 环境配置

在项目根目录执行以下命令。推荐使用 Python 3.13 和 `uv` 管理环境；Windows 可先通过 `winget install Astral.uv` 安装 `uv`。

```powershell
uv python install 3.13
uv sync
```

`uv sync` 会依据 `pyproject.toml` 与 `uv.lock` 创建虚拟环境并安装 OpenCV、MediaPipe 等依赖。确认 ArUco 模块可用：

```powershell
uv run python -c "import cv2; print(cv2.__version__); print(hasattr(cv2, 'aruco'))"
```

最后一项应输出 `True`。若摄像头不是默认设备，后续启动服务时将 `--camera 0` 改为实际编号。

## 手部模型

网页端会首次打开时自动从 MediaPipe 官方模型地址下载 Hand Landmarker，因此需要能访问网络。Python 离线原型使用本地模型 `models/hand_landmarker.task`；仓库已包含该文件。

如果模型缺失，可在项目根目录重新下载：

```powershell
New-Item -ItemType Directory -Force models
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" `
  -OutFile "models/hand_landmarker.task"
```

下载完成后确认文件存在：

```powershell
Get-Item models/hand_landmarker.task
```

## 当前入口

- `web/app.html`：统一演奏台，可在 ArUco 和无标记板子之间切换；由 Python 提供摄像头流和实时板子坐标。
- `src/serving/hand_tracking_board.py`：保留的 Python/OpenCV 桌面原型，用于离线视频调试。

## 运行网页演奏台

1. 如需使用 ArUco 模式，生成并打印四张标记：

   ```powershell
   uv run python src/serving/generate_aruco_markers.py
   ```

2. ArUco 模式下，将标记贴在普通板子的四角：上排 ID `0 1`，下排 ID `2 3`；保留白边。无标记模式可跳过这一步。

3. 启动本地服务：

   ```powershell
   uv run python src/serving/motionmuse_server.py --camera 0 --port 8000
   ```

4. 在浏览器打开 `http://localhost:8000`，在右侧切换定位模式。

服务每帧用四个 Marker 中心重新计算板子坐标，不会积累光流漂移。短暂遮挡 Marker 时最多保持上一次坐标 0.25 秒。

无标记模式下，点击“冻结画面，标定板子四角”，然后在冻结画面上依次点击左上、右上、右下、左下四个角。服务会逐帧寻找与标定板子外观匹配的矩形轮廓；短暂遮挡时仅从最近一次轮廓检测帧做光流预测，超过短时窗口仍未重检到轮廓就暂停演奏并提示重新标定，不会无限累积漂移。请使用边缘清晰、与背景有对比的板子，尽量避免手遮住整个边框。普通板子在强反光、相似背景或大幅遮挡下仍可能跟丢；该模式不能保证所有场景都像 ArUco 一样稳定。

按 `Ctrl+C` 可停止本地服务。若网页显示 `Marker 0 / 4`，说明摄像头、服务和网页已连通，但四张 Marker 尚未全部被识别。

## 无标记板子数据采集

录制和抽取训练数据、使用 CVAT 标注及转换为 YOLO 数据集的步骤见 [`docs/data_collection.md`](docs/data_collection.md)。

可运行无标记定位的合成画面测试：

```powershell
uv run python -m unittest discover -s tests
```

## 目录

- `web/`：统一网页演奏台和前端依赖。
- `src/serving/aruco_tracking.py`：四 Marker 检测、几何校验与短暂遮挡保持。
- `src/serving/markerless_tracking.py`：无标记板子的轮廓重检测、外观匹配与短时光流补偿。
- `src/serving/motionmuse_server.py`：摄像头 MJPEG 流、模式切换、冻结标定和板子状态接口。
- `src/serving/generate_aruco_markers.py`：生成可打印的 ID 0--3 标记。
- `models/`：MediaPipe 手部模型。
- `assets/`：测试视频。

## YOLO Pose Preview

Preview the pretrained human pose model with your camera:

```powershell
uv run python src/serving/preview_pose_model.py --source 0
```

The default weights detect human pose, not board corners. See [the preview guide](docs/pretrained_pose_preview.md) for video input and fine-tuned model usage.
