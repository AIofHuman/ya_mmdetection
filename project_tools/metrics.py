from __future__ import annotations

from pathlib import Path

import pandas as pd


def compare_model_metrics(
    fcos_map: float,
    fcos_map50: float,
    fcos_fps: float,
    yolo_map: float,
    yolo_map50: float,
    yolo_fps: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"model": "FCOS", "mAP": fcos_map, "mAP_50": fcos_map50, "FPS": fcos_fps},
            {"model": "YOLOv8s", "mAP": yolo_map, "mAP_50": yolo_map50, "FPS": yolo_fps},
        ]
    )


def save_metrics_comparison(metrics_df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_path, index=False)
    return output_path


def load_fcos_log_metrics(log_json_path: str | Path) -> pd.DataFrame:
    log_json_path = Path(log_json_path)
    rows = []
    with log_json_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rows.append(pd.read_json(line, typ="series").to_dict())
    return pd.DataFrame(rows)


def load_yolo_results(results_csv_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(results_csv_path)
