# 无标记板子数据采集与标注

## 推荐标注软件：CVAT

建议三人小组用 CVAT 管理图片标注。首版只训练四角关键点模型，因此只建立一个 Pose 标注任务，并导出 Ultralytics YOLO Pose 格式。官方格式文档：[YOLO 导入导出](https://docs.cvat.ai/docs/dataset_management/formats/format-yolo-ultralytics/)，[骨架标注](https://docs.cvat.ai/docs/manual/advanced/skeletons/)。

所有图片名以 `B01_`、`B02_` 等开头；无板负样本以 `NONE_` 开头。琴颈端方向不属于逐帧标签：应用首次标定时由演奏者选择哪条短边作为琴颈端，程序再依据板坐标计算和弦区与扫弦区。

## 1. 录制视频

从项目根目录运行后会先显示摄像头预览。确认板子和取景准备好后，按**空格键**开始录制；录制中按 `Q` 结束并保存。尚未开始时按 `Q` 会退出，不保存视频：

```powershell
uv run python src/dataset/capture_dataset.py --board-id B01 --condition slow --camera 0
```

每块板至少录下表四段，每段约 30 秒。标签描述的是**这段视频主要测试的动作条件**；四段尽量使用相同摄像头和分辨率，板子始终是要追踪的目标。

| `--condition` | 怎么拍 | 主要检查什么 |
| --- | --- | --- |
| `slow` | 板子正面朝向摄像头并完整入镜。先静止几秒，再缓慢向左右、上下平移，轻轻旋转并前后倾斜；尽量让手不要挡住角点。 | 基础检测、角点顺序，以及缓慢移动和透视变化时的定位稳定性。 |
| `performance` | 按平常演奏姿势拿板子，连续做和弦手型与扫弦动作。板子可自然小幅移动，手按正常方式遮挡板面，但不要刻意快速甩动或长时间盖住整块板。 | 最接近日常使用的综合情况，包括双手在板前活动和部分角点短时被挡。 |
| `fast` | 板子保持在摄像头视野内，做较快的左右/上下移动、旋转、倾斜；每次快速动作后短暂停稳，再换方向。可让边缘短暂靠近画面边缘，但不要整段都严重模糊或完全出画。 | 快速运动、运动模糊、突然转向和重新定位能力。 |
| `occlusion` | 板子尽量保持位置不动，分别用手或前臂短暂挡住一个角、两个相邻角和一段边缘；每次遮挡后露出板子几秒，再换位置遮挡。不要一直把板子全部盖住。 | 部分遮挡下角点预测，以及遮挡结束后的恢复；遮挡位置和时长应有变化。 |

如果板子被完全遮住或移出画面，也可以在对应视频末尾短暂加入再露出的过程，并在 `manifest.csv` 中按 `condition` 回看；这种片段是测试“丢失后重捕”，不能作为模型应在完全看不见板子时仍输出准确位置的依据。另录一些 `negative` 场景（画面里没有板子，或有相似长方形干扰物）：

```powershell
uv run python src/dataset/capture_dataset.py --board-id NONE --condition negative --camera 0
```

默认视频写到 `data/raw/<board_id>/<session_id>/recording.mp4`，并在旁边生成 `session.json`。每次录制会自动用时间戳生成 session ID。

采集者应主动轮换拍摄位置与背景，并尽量覆盖明亮、偏暗、顺光和背光等条件；同一块板的四类动作不要全在同一个场景、同一种光线下录完。若某段有特别的环境情况，可记在小组采集日志中。

## 2. 抽取待标注帧

默认每秒抽一帧，每段最多 150 帧：

```powershell
uv run python src/dataset/extract_dataset_frames.py --raw data/raw --output data/frames --interval 0.4
```

生成 `data/frames/*.jpg` 和 `manifest.csv`，清单保留板子编号、session、动作类别和帧时间，不包含场景和光照字段。先检查图片是否清晰、是否抽到动作变化和遮挡过程；对快速动作、遮挡开始/结束、重新出现等关键瞬间，可在 CVAT 中另外补入从原视频提取的帧。不要只靠连续相邻帧凑足数量。

## 3. 在 CVAT 标注并导出

以下步骤以 CVAT 的图像任务界面为准；版本更新后按钮位置或翻译可能略有差异。可以使用 CVAT Online 或自建 CVAT。

### 创建任务并定义四点骨架

1. 打开 CVAT，在 **Tasks** 页面点右上角 `+`，选择 `Create new task`。任务名填 `board-pose-pilot`（正式采集可按批次命名）。
2. 在 `Labels` / `Constructor` 中添加一个标签，名称填 `board`，形状设为 `Skeleton`，然后选择 `Setup skeleton`。
3. 在 Skeleton Configurator 里按顺时针顺序添加 4 个点，并分别命名 `1`、`2`、`3`、`4`。编号 `1` 是该板固定的起始角，`2`–`4` 沿板周界顺时针排列；同一块板在所有图片中保持相同的起始角和顺序，不能随画面旋转而交换。可依次连接 `1-2-3-4-1` 辅助检查四边形；连线只用于显示，不是额外标注。保存骨架配置。
4. 选择 `Select files`，上传 `data/frames` 中的图片。然后点 `Submit & Open` 创建任务。

骨架配置在任务创建后不能修改，所以正式标注前先用少量图片建 pilot task，导出并核对点的顺序；有问题就删掉试验任务、修正配置后重建。

### 逐图标四个角

1. 打开任务的 Job。在标注画布左侧选择 `Draw new skeleton`，选择 `Shape` 模式（这里是抽取后的独立图片，不使用视频 `Track` 模式），选择 `board` 标签。
2. 在每张**有板子**的图片上，先在板子外侧对角位置点击两次，画出包住板子的骨架外接框。**这两次点击是在画框，不是依次标四个关键点。** CVAT 创建骨架后会显示配置好的四个点。
3. 把四个点分别拖到真实角点，按 `1 → 2 → 3 → 4` 的固定顺序核对。点要放在板子四边的交点，不要点到手指边缘、阴影或贴纸角上。放大画面检查并微调位置。
4. 角点可见时保留普通可见状态；只有角点本身被手等物体挡住时才标为遮挡。选中被挡角后，在右侧 `Objects → PARTS` 中找到对应点（`1 [POINTS]` 等），点击该点行的**人形图标**切换 `Occluded`；也可以把鼠标悬停在画布上的那个关键点并按 `Q`。不要点击 `board` 父级行的人形图标，那会把整个骨架标为遮挡。若角点被挡但位置仍能从附近清晰帧和矩形几何可靠推断，先把点拖到推断位置，再设置 `Occluded`。CVAT 的 `Occluded` 会保留点坐标；YOLO Pose 导出会用 visibility 值区分可见点和遮挡点。
5. 如果角点位置无法可靠推断，不要猜一个坐标。首轮训练集应跳过这类图片，或把它们单独留作困难测试样本；只有四角坐标都可信的图片才作为带板 Pose 训练样本。完全没有板子的 `NONE_` 图片不画骨架，作为负样本保留。
6. 每完成一批按 `Ctrl+S` 保存。用帧导航抽查不同板子、快速移动和遮挡图片，重点检查同一板子的四点顺序没有翻转/交换。

CVAT 的官方说明：[创建任务](https://docs.cvat.ai/docs/workspace/tasks-page/)、[设置和标注 Skeleton](https://docs.cvat.ai/docs/manual/advanced/skeletons/)、[YOLO 格式与 Pose 导出](https://docs.cvat.ai/docs/manual/advanced/formats/format-yolo-ultralytics/)。

### 导出标注

1. 回到 Tasks 列表，找到任务，打开 `Actions` 菜单，选择 `Export task dataset`。
2. 格式选择 `Ultralytics YOLO Pose 1.0`，导出 ZIP，保存为 `data/exports/board-pose.zip`。不要修改导出图片文件名；转换脚本会用文件名与抽帧清单匹配每张图所属的录制片段。
3. 如果导出对话框提供 `Save images`，可以打开它，让 ZIP 同时带上原图和标签。如果 CVAT 不允许导出图片，直接导出标注 ZIP，并保留原始 `data/frames`；下面的转换脚本可以用 `--images-dir` 从本地帧目录配回原图。

CVAT 官方导出步骤见[导出数据集](https://docs.cvat.ai/docs/dataset_management/export-datasets/)。

## 4. 按录制片段划分训练集并转换

不需要手动维护板子划分文件。抽帧步骤生成的 `data/frames/manifest.csv` 记录每张图对应的 `board_id` 和 `session_id`。转换脚本会把同一段录制视频抽出的所有帧作为一个组，按默认 80/10/10 比例随机分配到训练、验证、测试集；使用固定随机种子保证重复运行时划分一致。不同录制片段即使拍摄同一块板，也可以进入不同集合，以评估模型对新录制视频的泛化能力。

运行转换：

```powershell
uv run python src/dataset/prepare_yolo_dataset.py `
  --pose-zip data/exports/board-pose.zip `
  --images-dir data/frames `
  --manifest data/frames/manifest.csv `
  --ratios 0.8,0.1,0.1 `
  --seed 42 `
  --output data/yolo_sessions
```

`--ratios` 和 `--seed` 可以省略，默认值分别为 `0.8,0.1,0.1` 和 `42`。相同抽帧清单、标注和 seed 会得到相同分组；需要重新随机分配时改 seed。即使 ZIP 已含有图片，也可以保留 `--images-dir data/frames`；脚本会按导出 ZIP 中的图片名配对本地原图。若 ZIP 仅含标注，脚本会使用 `data/frames` 目录中的图片，因此这个目录应当只放入本次 CVAT 任务对应的帧，不要混入其他批次图片。

一个集合至少需要一个完整录制片段。当前标注数据如果只来自一两段视频，脚本会尽量按比例分配，但会有一个或多个空集合，并在生成结果中提示；这时先把数据集作为流程试跑，不要据此判断模型泛化效果。后续采集多段独立视频后重新转换。负样本视频也按录制片段分组，避免同一视频的帧跨集合。

输出：

```text
data/yolo_sessions/
├── data.yaml
├── split.json
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

脚本会检查图片、标签格式和类别 ID、关键点数量/可见性范围，并将本次分组结果与 seed 保存到输出目录的 `split.json`。没有有效 Pose 标签的 `Bxx_` 图片会被跳过，只把已标注样本写入数据集；`NONE_` 负样本会保留。注意：CVAT 的 YOLO 导出不保留自定义属性，因此图片文件名必须保留，供脚本与 `manifest.csv` 对应。

## 5. 检查并训练

先打开每个集合里的随机图片，确认对应 `.txt` 标签存在且匹配；负样本可没有目标标签。人工抽查同一块板的四角编号是否始终沿板周界按顺时针排列，并检查 `split.json` 确认同一录制片段没有跨集合。验证集和测试集应包含训练过程中未出现过的录制片段。若还要单独测量模型对全新板子的泛化能力，应额外按板子留出测试数据，作为另一项评估。

训练时使用 `data/yolo_sessions/data.yaml`。每次转换请使用新的空输出目录，避免旧划分留下的图片混入新数据集。如果四角模型在未见录制片段或遮挡测试中达不到目标，再根据失败案例决定是否补充分割模型。原始视频、帧图、导出 ZIP、转换后的数据集体积可能很大，默认不会纳入版本控制；请至少备份原视频与抽帧 `manifest.csv`、转换输出中的 `split.json`、标注导出文件。
