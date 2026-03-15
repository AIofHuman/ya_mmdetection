from __future__ import annotations

from pathlib import Path


def _resolve_yolo_device(device: str | int) -> str | int:
    import torch

    if isinstance(device, int) and not torch.cuda.is_available():
        return "cpu"
    return device


def load_yolo_model(weights: str = "yolov8s.pt"):
    from ultralytics import YOLO

    return YOLO(weights)


def run_yolo_training(
    data_yaml: str | Path,
    project_dir: str | Path,
    model_weights: str = "yolov8s.pt",
    epochs: int = 30,
    imgsz: int = 640,
    batch: int = 16,
    device: str | int = 0,
):
    model = load_yolo_model(model_weights)
    return model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=_resolve_yolo_device(device),
        project=str(project_dir),
        name="train",
        exist_ok=True,
    )


def run_yolo_validation(model_path: str | Path, data_yaml: str | Path, split: str = "test"):
    model = load_yolo_model(str(model_path))
    return model.val(data=str(data_yaml), split=split)


def run_yolo_image_inference(
    model_path: str | Path,
    image_path: str | Path,
    project_dir: str | Path,
    name: str = "predict",
    conf: float = 0.25,
):
    model = load_yolo_model(str(model_path))
    return model.predict(
        source=str(image_path),
        conf=conf,
        project=str(project_dir),
        name=name,
        exist_ok=True,
        save=True,
    )


def export_yolo_predictions(
    model_path: str | Path,
    source_dir: str | Path,
    project_dir: str | Path,
    name: str,
    conf: float = 0.25,
):
    model = load_yolo_model(str(model_path))
    return model.predict(
        source=str(source_dir),
        conf=conf,
        project=str(project_dir),
        name=name,
        exist_ok=True,
        save=True,
    )
