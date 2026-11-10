# 预训练 YOLO Pose 效果预览

安装依赖后，可以先用 Ultralytics 的 `yolo11n-pose.pt` 预训练权重查看人体姿态推理效果。第一次运行时，Ultralytics 会自动下载权重文件。

## 摄像头预览

在项目根目录运行：

```powershell
uv run python src/serving/preview_pose_model.py --source 0
```

如果要使用其他摄像头，把 `0` 换成设备编号。按 `Q` 或 `Esc` 退出。

## 视频预览

```powershell
uv run python src/serving/preview_pose_model.py --source "data/frames/example.mp4"
```

还可以指定模型文件、置信度、推理尺寸或设备：

```powershell
uv run python src/serving/preview_pose_model.py `
  --source "data/frames/example.mp4" `
  --weights "runs/pose/train/weights/best.pt" `
  --imgsz 640 `
  --conf 0.25 `
  --device 0
```

自定义权重路径应替换成实际文件位置。`--device 0` 使用编号为 0 的 CUDA GPU；省略该参数时由 Ultralytics 自动选择设备。

## 预训练权重的能力范围

默认 `yolo11n-pose.pt` 是通用人体姿态模型，输出人体框和 COCO 人体关键点。它没有学习“板子”类别，也没有学习板子的四个角，因此此预览只能用于确认模型下载、推理和画面显示流程，不能用来判断板子追踪效果。预览画面会显示这条提示。

要检测板子四角，需要在已标注的板子 Pose 数据集上微调该预训练权重，再把训练得到的 `best.pt` 传给 `--weights`。例如：

```powershell
uv run yolo pose train `
  model=yolo11n-pose.pt `
  data=data/yolo_sessions/data.yaml `
  imgsz=640 `
  epochs=100
```

完成训练后，将输出的 `best.pt` 交给同一个预览脚本。此时脚本会显示模型预测的板框和关键点；板角准确度取决于标注质量、数据覆盖和训练结果。
