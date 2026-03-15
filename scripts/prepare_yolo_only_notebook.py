from __future__ import annotations

import json
import shutil
from pathlib import Path

from project_tools.dataset import prepare_yolo_labels


def main() -> None:
    root = Path("datasets/minecraft")
    prepare_yolo_labels(root)

    for cache in root.glob("*.cache"):
        cache.unlink()

    shutil.rmtree("artifacts/yolo", ignore_errors=True)
    Path("artifacts/yolo").mkdir(parents=True, exist_ok=True)

    src = Path("notebook.ipynb")
    dst = Path("notebook_yolo_only.ipynb")
    nb = json.loads(src.read_text())

    setup_cell = nb["cells"][2]
    source = "".join(setup_cell["source"])
    source = source.replace("RUN_FCOS_TRAINING = True", "RUN_FCOS_TRAINING = False")
    setup_cell["source"] = [line + "\n" for line in source.split("\n")]

    trained_cell = nb["cells"][20]
    trained_source = """trained_fcos_checkpoint = FCOS_ARTIFACTS / 'missing_fcos_checkpoint.pth'
trained_yolo_checkpoint = YOLO_ARTIFACTS / 'train' / 'weights' / 'best.pt'

pd.Series({
    'trained_fcos_checkpoint_exists': trained_fcos_checkpoint.exists(),
    'trained_yolo_checkpoint_exists': trained_yolo_checkpoint.exists(),
})
"""
    trained_cell["source"] = [line + "\n" for line in trained_source.split("\n")]

    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    dst.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    print(dst)


if __name__ == "__main__":
    main()
