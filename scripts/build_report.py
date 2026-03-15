from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
REPORT_PATH = ARTIFACTS / "report.pdf"
NOTEBOOK_PATH = ROOT / "notebook.ipynb"
YOLO_RESULTS_PATH = ARTIFACTS / "yolo" / "train" / "results.csv"
REPORT_ASSETS = ARTIFACTS / "report_assets"
FCOS_IMAGE = REPORT_ASSETS / "fcos_detection.jpg"
YOLO_IMAGE = REPORT_ASSETS / "yolo_detection.jpg"


def extract_notebook_text() -> tuple[str, str]:
    data = json.loads(NOTEBOOK_PATH.read_text())
    eval_text = ""
    video_text = ""
    for cell in data["cells"]:
        outputs = cell.get("outputs", [])
        joined = []
        for out in outputs:
            if "text" in out:
                joined.append("".join(out["text"]))
            elif "data" in out and "text/plain" in out["data"]:
                joined.append("".join(out["data"]["text/plain"]))
        text = "\n".join(joined)
        if "Speed: 1.3ms preprocess, 9.6ms inference" in text:
            eval_text = text
        if "Processed 756 frames in 241.10s" in text:
            video_text = text
    return eval_text, video_text


def parse_fcos_quality(eval_text: str) -> dict[str, float]:
    match = re.search(
        r"coco/bbox_mAP:\s+([0-9.]+)\s+coco/bbox_mAP_50:\s+([0-9.]+)\s+coco/bbox_mAP_75:\s+([0-9.]+)",
        eval_text,
    )
    m_ap, m_ap50, m_ap75 = map(float, match.groups())
    return {"mAP50_95": m_ap, "mAP50": m_ap50, "mAP75": m_ap75}


def parse_yolo_quality(eval_text: str) -> dict[str, float]:
    match = re.search(
        r"\ball\s+155\s+351\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)",
        eval_text,
    )
    precision, recall, m_ap50, m_ap50_95 = map(float, match.groups())
    return {
        "precision": precision,
        "recall": recall,
        "mAP50": m_ap50,
        "mAP50_95": m_ap50_95,
    }


def parse_speed(eval_text: str, video_text: str) -> dict[str, float]:
    yolo_match = re.search(
        r"Speed:\s+([0-9.]+)ms preprocess,\s+([0-9.]+)ms inference,\s+([0-9.]+)ms loss,\s+([0-9.]+)ms postprocess per image",
        eval_text,
    )
    yolo_pre, yolo_inf, _, yolo_post = map(float, yolo_match.groups())
    yolo_total_ms = yolo_pre + yolo_inf + yolo_post
    yolo_fps = 1000.0 / yolo_total_ms

    fcos_match = re.search(r"Processed\s+756\s+frames\s+in\s+([0-9.]+)s\s+\(([0-9.]+)\s+FPS\)", video_text)
    fcos_total_sec, fcos_fps = map(float, fcos_match.groups())
    return {
        "fcos_fps_video": fcos_fps,
        "fcos_total_video_sec": fcos_total_sec,
        "yolo_ms_per_image": yolo_total_ms,
        "yolo_fps_equivalent": yolo_fps,
    }


def parse_yolo_train_best() -> dict[str, float]:
    with YOLO_RESULTS_PATH.open(newline="") as fp:
        rows = list(csv.DictReader(fp))
    best = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    return {
        "epoch": int(best["epoch"]),
        "mAP50": float(best["metrics/mAP50(B)"]),
        "mAP50_95": float(best["metrics/mAP50-95(B)"]),
    }


def add_text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    plt.axis("off")
    fig.text(0.08, 0.95, title, fontsize=20, fontweight="bold", va="top")

    y = 0.9
    for line in lines:
        if line == "":
            y -= 0.03
            continue
        fig.text(0.08, y, line, fontsize=11, va="top")
        y -= 0.035
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_table_page(pdf: PdfPages, title: str, headers: list[str], rows: list[list[str]], notes: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    ax.text(0.0, 1.02, title, fontsize=20, fontweight="bold", transform=ax.transAxes, va="bottom")

    table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="upper left", bbox=[0, 0.55, 1, 0.28])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.4)

    y = 0.48
    for note in notes:
        ax.text(0.0, y, note, fontsize=11, transform=ax.transAxes, va="top")
        y -= 0.05

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_examples_page(pdf: PdfPages, title: str, left_title: str, right_title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=20, fontweight="bold", y=0.97)

    for ax, img_path, subtitle in [
        (axes[0], FCOS_IMAGE, left_title),
        (axes[1], YOLO_IMAGE, right_title),
    ]:
        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_title(subtitle, fontsize=12)
        ax.axis("off")

    fig.text(
        0.05,
        0.05,
        "Дополнительные ссылки пользователя на примеры инференса:\n"
        "FCOS: https://disk.yandex.ru/i/xxbgumQahrU7iw\n"
        "YOLO: https://disk.yandex.ru/i/rVEtIt70ZjqdfA",
        fontsize=10,
        va="bottom",
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    REPORT_ASSETS.mkdir(parents=True, exist_ok=True)
    eval_text, video_text = extract_notebook_text()
    fcos_quality = parse_fcos_quality(eval_text)
    yolo_quality = parse_yolo_quality(eval_text)
    speed = parse_speed(eval_text, video_text)
    yolo_train_best = parse_yolo_train_best()

    with PdfPages(REPORT_PATH) as pdf:
        add_text_page(
            pdf,
            "Minecraft Mob Detection Report",
            [
                "Проект сравнивает две модели детекции объектов на датасете мобов Minecraft:",
                "FCOS (MMDetection) и YOLOv8s (Ultralytics).",
                "",
                "В отчёте использованы артефакты полного GPU-прогона ноутбука:",
                "- test-метрики обеих моделей;",
                "- итоговые веса и логи обучения;",
                "- видео с детекциями и извлечённые из них кадры.",
            ],
        )

        add_table_page(
            pdf,
            "Качество Детекции",
            ["Модель", "mAP50-95", "mAP50", "Доп. метрика"],
            [
                ["FCOS", f"{fcos_quality['mAP50_95']:.3f}", f"{fcos_quality['mAP50']:.3f}", f"mAP75={fcos_quality['mAP75']:.3f}"],
                [
                    "YOLOv8s",
                    f"{yolo_quality['mAP50_95']:.3f}",
                    f"{yolo_quality['mAP50']:.3f}",
                    f"P={yolo_quality['precision']:.3f}, R={yolo_quality['recall']:.3f}",
                ],
            ],
            [
                "Вывод по качеству: YOLOv8s победила.",
                "На тестовом наборе YOLO показывает заметно более высокий mAP50-95 и mAP50.",
                "Даже лучший тренировочный результат YOLO (mAP50-95 "
                f"{yolo_train_best['mAP50_95']:.3f}) остаётся существенно выше, чем у FCOS.",
            ],
        )

        add_table_page(
            pdf,
            "Скорость Инференса",
            ["Модель", "Источник", "Скорость"],
            [
                ["FCOS", "Видео-инференс из ноутбука", f"{speed['fcos_fps_video']:.2f} FPS"],
                [
                    "YOLOv8s",
                    "test inference log",
                    f"{speed['yolo_ms_per_image']:.1f} ms/изобр. (~{speed['yolo_fps_equivalent']:.1f} FPS)",
                ],
            ],
            [
                "Вывод по скорости: YOLOv8s заметно быстрее FCOS.",
                "Для FCOS в ноутбуке зафиксирован end-to-end видео-инференс 3.14 FPS.",
                "Для YOLO по логам тестового инференса суммарная задержка около "
                f"{speed['yolo_ms_per_image']:.1f} ms на изображение, что соответствует примерно "
                f"{speed['yolo_fps_equivalent']:.1f} FPS.",
                "Источники скорости различаются по сценарию, но оба указывают на уверенное преимущество YOLO.",
            ],
        )

        add_examples_page(
            pdf,
            "Примеры Детекций",
            "FCOS: кадр из artifacts/videos/fcos_inference.mp4",
            "YOLOv8s: кадр из artifacts/videos/yolo_inference.mp4",
        )


if __name__ == "__main__":
    main()
