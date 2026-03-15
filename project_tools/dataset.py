from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import yaml

MINECRAFT_CLASSES = (
    "bee",
    "chicken",
    "cow",
    "creeper",
    "enderman",
    "fox",
    "frog",
    "ghast",
    "goat",
    "llama",
    "pig",
    "sheep",
    "skeleton",
    "spider",
    "turtle",
    "wolf",
    "zombie",
)

CLASS_NAME_TO_INDEX = {name: idx for idx, name in enumerate(MINECRAFT_CLASSES)}


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fp:
        return json.load(fp)


def coco_annotation_summary(annotation_path: str | Path) -> dict[str, Any]:
    data = load_json(annotation_path)
    categories = {item["id"]: item["name"] for item in data["categories"]}
    image_ids = {item["id"] for item in data["images"]}
    class_counts = Counter(categories[item["category_id"]] for item in data["annotations"])
    missing_images = sorted(
        item["image_id"] for item in data["annotations"] if item["image_id"] not in image_ids
    )
    return {
        "images": len(data["images"]),
        "annotations": len(data["annotations"]),
        "categories": categories,
        "class_counts": dict(class_counts),
        "missing_image_ids": missing_images,
    }


def build_yolo_data_yaml(dataset_root: str | Path, output_path: str | Path) -> Path:
    dataset_root = Path(dataset_root).resolve()
    output_path = Path(output_path)
    payload = {
        "path": str(dataset_root),
        "train": "train",
        "val": "val",
        "test": "test",
        "names": {idx: name for idx, name in enumerate(MINECRAFT_CLASSES)},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(payload, fp, sort_keys=False, allow_unicode=False)
    return output_path


def convert_coco_split_to_yolo(
    annotation_path: str | Path,
    images_dir: str | Path,
    labels_dir: str | Path,
) -> Path:
    data = load_json(annotation_path)
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    categories = {item["id"]: item["name"] for item in data["categories"]}
    image_info_by_id = {item["id"]: item for item in data["images"]}
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in data["annotations"]:
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    for image_id, image_info in image_info_by_id.items():
        width = image_info["width"]
        height = image_info["height"]
        label_lines: list[str] = []

        for ann in annotations_by_image.get(image_id, []):
            category_name = categories[ann["category_id"]]
            if category_name not in CLASS_NAME_TO_INDEX:
                continue

            x_min, y_min, box_w, box_h = ann["bbox"]
            x_center = (x_min + box_w / 2) / width
            y_center = (y_min + box_h / 2) / height
            norm_w = box_w / width
            norm_h = box_h / height
            class_id = CLASS_NAME_TO_INDEX[category_name]
            label_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
            )

        label_path = labels_dir / f"{Path(image_info['file_name']).stem}.txt"
        label_path.write_text("\n".join(label_lines), encoding="utf-8")

        image_path = images_dir / image_info["file_name"]
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image referenced in COCO json: {image_path}")

    return labels_dir


def prepare_yolo_labels(dataset_root: str | Path) -> dict[str, Path]:
    dataset_root = Path(dataset_root)
    annotations_root = dataset_root / "annotations"
    outputs = {}
    split_mapping = {
        "train": "train_annotations.json",
        "val": "val_annotations.json",
        "test": "test_annotations.json",
    }
    for split, ann_name in split_mapping.items():
        legacy_labels_dir = dataset_root / split / "labels"
        if legacy_labels_dir.exists():
            for old_label in legacy_labels_dir.glob("*.txt"):
                old_label.unlink()
            legacy_labels_dir.rmdir()

        outputs[split] = convert_coco_split_to_yolo(
            annotation_path=annotations_root / ann_name,
            images_dir=dataset_root / split,
            labels_dir=dataset_root / split,
        )
    return outputs
