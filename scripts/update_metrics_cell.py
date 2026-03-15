from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell


def main() -> None:
    nb_path = Path("notebook.ipynb").resolve()
    nb = nbformat.read(nb_path, as_version=4)

    setup_idx = 2
    metrics_idx = 23

    runtime_nb = nbformat.v4.new_notebook()
    runtime_nb.cells = [
        nb.cells[setup_idx],
        new_code_cell(
            "from pathlib import Path\n"
            "PROJECT_ROOT = Path.cwd()\n"
            "DATASET_ROOT = PROJECT_ROOT / 'datasets' / 'minecraft'\n"
            "ANNOTATIONS_ROOT = DATASET_ROOT / 'annotations'\n"
            "ARTIFACTS_ROOT = PROJECT_ROOT / 'artifacts'\n"
            "FCOS_ARTIFACTS = ARTIFACTS_ROOT / 'fcos'\n"
            "YOLO_ARTIFACTS = ARTIFACTS_ROOT / 'yolo'\n"
            "INFERENCE_ROOT = ARTIFACTS_ROOT / 'inference'\n"
            "VIDEOS_ROOT = ARTIFACTS_ROOT / 'videos'\n"
            "METRICS_ROOT = ARTIFACTS_ROOT / 'metrics'\n"
        ),
        nb.cells[metrics_idx],
    ]

    client = NotebookClient(
        runtime_nb,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(nb_path.parent)}},
    )
    client.execute()

    nb.cells[metrics_idx]["outputs"] = runtime_nb.cells[-1]["outputs"]
    nb.cells[metrics_idx]["execution_count"] = runtime_nb.cells[-1]["execution_count"]
    nbformat.write(nb, nb_path)


if __name__ == "__main__":
    main()
