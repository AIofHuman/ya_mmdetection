# Environment Setup

Локально для проекта лучше использовать `Python 3.10` или `Python 3.11`.
`Python 3.13` для MMDetection и связанных пакетов брать не стоит.

## Рекомендуемая схема

1. Создать venv на `Python 3.10`.
2. Установить PyTorch под вашу версию CUDA.
3. Установить зависимости проекта.
4. Открыть Jupyter из каталога `mmdetection`.

## Быстрый старт

Из каталога [`mmdetection`](/Users/borisfox/Documents/ML/nn/sprint%205/project/mmdetection):

```bash
bash setup_env.sh
```

Если нужно явно указать бинарник Python:

```bash
PYTHON_BIN=python3.10 bash setup_env.sh
```

Если `python3.10` не найден, можно использовать `python3.11`:

```bash
PYTHON_BIN=python3.11 bash setup_env.sh
```

## Ручная установка

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-gpu.txt
```

Если у вас не CUDA 11.8, замените индекс PyTorch на соответствующий вашей ВМ.

Примечание: для этого проекта используется рабочая связка из `requirements.txt`:

- `openmim>=0.3.9,<0.4`
- `mmengine==0.10.7`
- `mmdet==3.3.0`
- `mmcv==2.1.0` через `python -m mim install "mmcv>=2.0.0,<2.2.0"`

На GPU-ВМ нужен именно полный `mmcv`, а не `mmcv-lite`, иначе FCOS не сможет штатно обучаться и выполнять инференс.

Датасеты, веса и `artifacts/` в Git не хранятся. Их лучше копировать на ВМ напрямую.

## После установки

Запуск ноутбука:

```bash
source .venv/bin/activate
jupyter notebook
```

Рабочий ноутбук проекта:

- [`notebook.ipynb`](/Users/borisfox/Documents/ML/nn/sprint%205/project/mmdetection/notebook.ipynb)

Ключевые файлы:

- [`requirements.txt`](/Users/borisfox/Documents/ML/nn/sprint%205/project/mmdetection/requirements.txt)
- [`requirements-gpu.txt`](/Users/borisfox/Documents/ML/nn/sprint%205/project/mmdetection/requirements-gpu.txt)
- [`setup_env.sh`](/Users/borisfox/Documents/ML/nn/sprint%205/project/mmdetection/setup_env.sh)
