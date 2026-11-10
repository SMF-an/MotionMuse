"""Prepare a CVAT Ultralytics YOLO Pose export, splitting by recording session."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path, PurePosixPath
import random
import zipfile


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def archive_index(archive: zipfile.ZipFile) -> tuple[dict[str, str], dict[str, str]]:
    images: dict[str, str] = {}
    labels: dict[str, str] = {}
    for info in archive.infolist():
        if info.is_dir():
            continue
        path = PurePosixPath(info.filename)
        if path.name.startswith("."):
            continue
        if path.suffix.lower() in IMAGE_SUFFIXES and "images" in path.parts:
            key = path.stem
            if key in images:
                raise ValueError(f"压缩包内图片重名：{key}")
            images[key] = info.filename
        elif path.suffix.lower() == ".txt" and "labels" in path.parts:
            key = path.stem
            if key in labels:
                raise ValueError(f"压缩包内标签重名：{key}")
            labels[key] = info.filename
    return images, labels


def directory_image_index(directory: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.stem in images:
            raise ValueError(f"图片目录中存在重名文件：{path.stem}")
        images[path.stem] = path
    return images


def check_pose_label(text: str, filename: str) -> None:
    for line_no, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        parts = raw.split()
        try:
            values = [float(value) for value in parts]
        except ValueError as exc:
            raise ValueError(f"{filename}:{line_no} 存在非数字标签") from exc
        expected = 17  # class + bbox(4) + 4 keypoints(3 each)
        if len(values) != expected:
            raise ValueError(f"{filename}:{line_no} 应有 17 个数值（1 类别、4 框值、4×3 关键点），实际 {len(values)}")
        if values[0] != 0:
            raise ValueError(f"{filename}:{line_no} 类别 ID 应为 0，实际为 {values[0]}")
        bbox_and_points = values[1:]
        visibility = bbox_and_points[6::3]
        coordinates = [value for index, value in enumerate(bbox_and_points) if index not in {6, 9, 12, 15}]
        if any(value < -1e-4 or value > 1.0001 for value in coordinates):
            raise ValueError(f"{filename}:{line_no} 关键点/框坐标超出 [0,1]")
        if any(value not in (0, 1, 2) for value in visibility):
            raise ValueError(f"{filename}:{line_no} 关键点可见性必须为 0、1 或 2")


def write_yaml(path: Path) -> None:
    content = [
        f"path: {path.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
        "  0: board",
    ]
    content += ["kpt_shape: [4, 3]"]
    path.joinpath("data.yaml").write_text("\n".join(content) + "\n", encoding="utf-8")


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    """Index extract_dataset_frames.py rows by image stem."""
    result: dict[str, dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as manifest_file:
        reader = csv.DictReader(manifest_file)
        required = {"image", "board_id", "session_id"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"manifest 缺少必需列：{', '.join(sorted(required))}")
        for row in reader:
            stem = Path(row["image"]).stem
            if stem in result:
                raise ValueError(f"manifest 中图片名重复：{stem}")
            row["board_id"] = row["board_id"].strip().upper()
            row["session_id"] = row["session_id"].strip()
            if not row["board_id"] or not row["session_id"]:
                raise ValueError(f"manifest 中 {stem} 的 board_id 或 session_id 为空")
            result[stem] = row
    return result


def parse_ratios(value: str) -> dict[str, float]:
    try:
        numbers = [float(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("比例格式应为 train,val,test，例如 0.8,0.1,0.1") from exc
    if len(numbers) != 3 or any(not math.isfinite(number) or number < 0 for number in numbers):
        raise argparse.ArgumentTypeError("需要三个非负有限比例，例如 0.8,0.1,0.1")
    if sum(numbers) <= 0:
        raise argparse.ArgumentTypeError("三个比例不能全为 0")
    total = sum(numbers)
    return dict(zip(("train", "val", "test"), (number / total for number in numbers)))


def split_sessions(session_keys: set[str], ratios: dict[str, float], seed: int) -> dict[str, list[str]]:
    """Randomly assign whole recording sessions with largest-remainder counts."""
    active = [name for name in ("train", "val", "test") if ratios[name] > 0]
    if not session_keys:
        raise ValueError("没有可用于划分的已标注录制片段")
    shuffled = sorted(session_keys)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) < len(active):
        selected = sorted(active, key=lambda name: ratios[name], reverse=True)[:len(shuffled)]
        result = {name: [] for name in ("train", "val", "test")}
        for index, name in enumerate(selected):
            result[name] = [shuffled[index]]
        return result
    counts = {name: math.floor(len(shuffled) * ratios[name]) for name in active}
    remaining = len(shuffled) - sum(counts.values())
    order = sorted(active, key=lambda name: (len(shuffled) * ratios[name] - counts[name], ratios[name]), reverse=True)
    for name in order[:remaining]:
        counts[name] += 1
    if len(shuffled) >= len(active):
        for empty in [name for name in active if counts[name] == 0]:
            donors = [name for name in active if counts[name] > 1]
            donor = max(donors, key=lambda name: counts[name] - len(shuffled) * ratios[name])
            counts[donor] -= 1
            counts[empty] += 1
    result = {name: [] for name in ("train", "val", "test")}
    offset = 0
    for name in active:
        result[name] = shuffled[offset:offset + counts[name]]
        offset += counts[name]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-zip", type=Path, required=True, help="CVAT Ultralytics YOLO Pose 导出 ZIP")
    parser.add_argument("--images-dir", type=Path, help="原始抽帧图片目录；ZIP 未包含 images/ 时使用，例如 data/frames")
    parser.add_argument("--manifest", type=Path, required=True, help="extract_dataset_frames.py 生成的 manifest.csv")
    parser.add_argument("--ratios", type=parse_ratios, default=parse_ratios("0.8,0.1,0.1"), help="train,val,test 比例，默认 0.8,0.1,0.1")
    parser.add_argument("--seed", type=int, default=42, help="随机划分种子；相同数据和种子会得到相同划分")
    parser.add_argument("--output", type=Path, default=Path("data/yolo_sessions"))
    args = parser.parse_args()
    manifest = read_manifest(args.manifest)
    if any((args.output / kind).exists() and any((args.output / kind).rglob("*")) for kind in ("images", "labels")):
        raise SystemExit(f"输出目录已有图片或标签：{args.output}。请指定一个新的空目录，避免新旧划分结果混在一起。")

    with zipfile.ZipFile(args.pose_zip) as pose_zip:
        pose_images, pose_labels = archive_index(pose_zip)
        if args.images_dir:
            local_images = directory_image_index(args.images_dir)
            if not local_images:
                parser.error(f"图片目录中没有可用图片：{args.images_dir}")
            if pose_images:
                missing = set(pose_images) - set(local_images)
                if missing:
                    raise SystemExit(f"图片目录缺少 CVAT 导出中的 {len(missing)} 张图片，例如：{next(iter(missing))}")
                image_sources = {stem: local_images[stem] for stem in pose_images}
            else:
                image_sources = local_images
        else:
            if not pose_images:
                parser.error("导出 ZIP 不含原图；请用 --images-dir 指向 data/frames")
            image_sources = pose_images

        missing_label_images = set(pose_labels) - set(image_sources)
        if missing_label_images:
            raise SystemExit(f"有 {len(missing_label_images)} 个标签在图片集中找不到对应图片，例如：{next(iter(missing_label_images))}")
        session_by_stem: dict[str, str] = {}
        for stem in image_sources:
            if stem not in manifest:
                raise SystemExit(f"manifest.csv 中找不到图片 {stem}；请确认 CVAT 导出对应 data/frames")
            row = manifest[stem]
            session_by_stem[stem] = f"{row['board_id']}/{row['session_id']}"

        dataset_dir = args.output
        included: dict[str, str] = {}
        skipped_unlabeled = 0
        eligible: list[tuple[str, Path | str, str]] = []
        for stem, image_source in image_sources.items():
            label_member = pose_labels.get(stem)
            label_text = pose_zip.read(label_member).decode("utf-8") if label_member else ""
            is_negative = stem.upper().startswith("NONE_")
            if not is_negative and not label_text.strip():
                skipped_unlabeled += 1
                continue
            check_pose_label(label_text, f"pose/{stem}.txt")

            eligible.append((stem, image_source, label_text))

        session_splits = split_sessions({session_by_stem[stem] for stem, _, _ in eligible}, args.ratios, args.seed)
        split_by_session = {session: split for split, sessions in session_splits.items() for session in sessions}
        for stem, image_source, label_text in eligible:
            split = split_by_session[session_by_stem[stem]]
            image_ext = image_source.suffix.lower() if isinstance(image_source, Path) else PurePosixPath(image_source).suffix.lower()
            target_image = dataset_dir / "images" / split / f"{stem}{image_ext}"
            target_label = dataset_dir / "labels" / split / f"{stem}.txt"
            target_image.parent.mkdir(parents=True, exist_ok=True)
            target_label.parent.mkdir(parents=True, exist_ok=True)
            image_bytes = image_source.read_bytes() if isinstance(image_source, Path) else pose_zip.read(image_source)
            target_image.write_bytes(image_bytes)
            target_label.write_text(label_text, encoding="utf-8")
            included[stem] = split
        write_yaml(dataset_dir)
        split_record = {
            "strategy": "recording_session",
            "seed": args.seed,
            "ratios": args.ratios,
            "group_key": "board_id/session_id",
            "splits": session_splits,
        }
        (dataset_dir / "split.json").write_text(json.dumps(split_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已生成四角关键点数据集：{args.output.resolve()}")
    for split in ("train", "val", "test"):
        print(f"  {split}: {sum(value == split for value in included.values())} 张")
    print(f"跳过未标注图片：{skipped_unlabeled} 张；NONE_ 负样本仍会保留。")
    print(f"按录制片段划分（seed={args.seed}）：")
    for split in ("train", "val", "test"):
        print(f"  {split}: {len(session_splits[split])} 个片段")
    requested_splits = sum(ratio > 0 for ratio in args.ratios.values())
    if len({session for sessions in session_splits.values() for session in sessions}) < requested_splits:
        print("警告：独立录制片段数量不足，部分集合为空；增加独立录制后重新生成数据集。")
    print("划分记录已保存到 split.json；同一录制片段中的帧不会跨集合。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
