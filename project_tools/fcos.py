from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def _resolve_torch_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def build_fcos_config_path(project_root: str | Path) -> Path:
    return Path(project_root) / "configs" / "fcos" / "fcos_minecraft.py"


def run_fcos_training(
    project_root: str | Path,
    config_path: str | Path,
    work_dir: str | Path,
) -> subprocess.CompletedProcess:
    project_root = Path(project_root)
    command = [
        sys.executable,
        "tools/train.py",
        str(config_path),
        "--work-dir",
        str(work_dir),
    ]
    return subprocess.run(command, cwd=project_root, check=True)


def run_fcos_test_evaluation(
    project_root: str | Path,
    config_path: str | Path,
    checkpoint_path: str | Path,
    work_dir: str | Path | None = None,
) -> subprocess.CompletedProcess:
    project_root = Path(project_root)
    command = [
        sys.executable,
        "tools/test.py",
        str(config_path),
        str(checkpoint_path),
        "--work-dir",
        str(work_dir or project_root / "artifacts" / "fcos"),
    ]
    return subprocess.run(command, cwd=project_root, check=True)


def run_fcos_image_inference(
    config_path: str | Path,
    checkpoint_path: str | Path,
    image_path: str | Path,
    output_path: str | Path,
    score_thr: float = 0.3,
):
    from mmdet.apis import DetInferencer

    inferencer = DetInferencer(
        model=str(config_path),
        weights=str(checkpoint_path),
        device=_resolve_torch_device(),
    )
    return inferencer(
        str(image_path),
        out_dir=str(Path(output_path).parent),
        no_save_pred=True,
        pred_score_thr=score_thr,
    )
