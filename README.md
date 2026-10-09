# MotionMuse 幻奏

基于 ArUco 板子定位、MediaPipe Hand 和 OpenCV 的虚拟吉他原型。

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

- `web/app.html`：默认 ArUco 演奏台；由 Jetson/Python 提供摄像头流和实时板子坐标。
- `web/app_klt_legacy.html`：保留的浏览器 KLT 光流备用方案。
- `src/hand_tracking_board.py`：保留的 Python/OpenCV 桌面原型，用于离线视频调试。

## 运行 ArUco 网页

1. 生成并打印四张标记：

   ```powershell
   uv run python src/generate_aruco_markers.py
   ```

2. 将标记贴在普通板子的四角：上排 ID `0 1`，下排 ID `2 3`；保留白边。

3. 启动本地服务：

   ```powershell
   uv run python src/aruco_server.py --camera 0 --port 8000
   ```

4. 在浏览器打开 `http://localhost:8000`。

服务每帧用四个 Marker 中心重新计算板子坐标，不会积累光流漂移。短暂遮挡 Marker 时最多保持上一次坐标 0.25 秒。

若要回退到旧的无标记方案，可打开 `http://localhost:8000/app_klt_legacy.html`。

按 `Ctrl+C` 可停止本地服务。若网页显示 `Marker 0 / 4`，说明摄像头、服务和网页已连通，但四张 Marker 尚未全部被识别。

## 目录

- `web/`：ArUco 网页演示、KLT 备用页和前端依赖。
- `src/aruco_tracking.py`：四 Marker 检测、几何校验与短暂遮挡保持。
- `src/aruco_server.py`：摄像头 MJPEG 流和板子状态接口。
- `src/generate_aruco_markers.py`：生成可打印的 ID 0--3 标记。
- `models/`：MediaPipe 手部模型。
- `assets/`：测试视频。
