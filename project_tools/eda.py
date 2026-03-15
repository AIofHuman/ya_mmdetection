from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def load_coco_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fp:
        return json.load(fp)


def validate_coco_annotations(annotation_path: str | Path, images_dir: str | Path) -> dict[str, Any]:
    data = load_coco_json(annotation_path)
    images_dir = Path(images_dir)
    image_ids = {item["id"] for item in data["images"]}
    image_names = {item["file_name"] for item in data["images"]}
    disk_names = {path.name for path in images_dir.iterdir() if path.is_file()}

    annotations_without_images = [
        ann["id"] for ann in data["annotations"] if ann["image_id"] not in image_ids
    ]
    missing_on_disk = sorted(image_names - disk_names)
    extra_on_disk = sorted(disk_names - image_names)

    return {
        "images_in_json": len(data["images"]),
        "annotations_in_json": len(data["annotations"]),
        "categories_in_json": len(data["categories"]),
        "annotations_without_images": annotations_without_images,
        "missing_on_disk": missing_on_disk,
        "extra_on_disk": extra_on_disk,
        "is_valid": not annotations_without_images and not missing_on_disk,
    }


def plot_class_distribution(annotation_path: str | Path, figsize: tuple[int, int] = (12, 5)):
    data = load_coco_json(annotation_path)
    categories = {item["id"]: item["name"] for item in data["categories"]}
    counts = Counter(categories[ann["category_id"]] for ann in data["annotations"])
    labels = list(counts.keys())
    values = [counts[label] for label in labels]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(labels, values)
    ax.set_title("Minecraft Class Distribution")
    ax.set_ylabel("Annotations")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig, ax, counts


def draw_coco_sample(
    annotation_path: str | Path,
    images_dir: str | Path,
    image_index: int = 0,
    figsize: tuple[int, int] = (10, 10),
):
    data = load_coco_json(annotation_path)
    images = data["images"]
    image_info = images[image_index]
    categories = {item["id"]: item["name"] for item in data["categories"]}
    annotations = [ann for ann in data["annotations"] if ann["image_id"] == image_info["id"]]

    image_path = Path(images_dir) / image_info["file_name"]
    image = Image.open(image_path).convert("RGB")

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)
        ax.text(x, y, categories[ann["category_id"]], color="white", backgroundcolor="green")

    ax.set_title(image_info["file_name"])
    ax.axis("off")
    fig.tight_layout()
    return fig, ax, image_info, annotations
