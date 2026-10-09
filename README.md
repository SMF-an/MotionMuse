# MotionMuse 幻奏

基于 ArUco 板子定位、MediaPipe Hand 和 OpenCV 的虚拟吉他原型。

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

## 目录

- `web/`：ArUco 网页演示、KLT 备用页和前端依赖。
- `src/aruco_tracking.py`：四 Marker 检测、几何校验与短暂遮挡保持。
- `src/aruco_server.py`：摄像头 MJPEG 流和板子状态接口。
- `src/generate_aruco_markers.py`：生成可打印的 ID 0--3 标记。
- `models/`：MediaPipe 手部模型。
- `assets/`：测试视频。
