# Minecraft Mob Detection

Проект по детекции мобов Minecraft двумя моделями: `FCOS` из MMDetection и `YOLOv8s` из Ultralytics. Основной сценарий работы собран в [notebook.ipynb](/Users/borisfox/Documents/ML/nn/sprint%205/project/mmdetection/notebook.ipynb): EDA, подготовка данных, обучение, инференс и сравнение результатов.

## Что внутри

- `datasets/minecraft/` — датасет и аннотации.
- `configs/fcos/fcos_minecraft.py` — конфиг FCOS под датасет проекта.
- `project_tools/` — вспомогательные модули для ноутбука.
- `artifacts/` — веса моделей, метрики, примеры инференса, видео и итоговый отчёт.
- `setup_env.sh` — подготовка окружения для локального запуска или GPU-ВМ.

## Результат

По итоговым тестовым метрикам в проекте лучше показала себя `YOLOv8s`: она заметно опережает FCOS как по качеству детекции, так и по скорости инференса. Сводка и примеры результатов собраны в [artifacts/report.pdf](/Users/borisfox/Documents/ML/nn/sprint%205/project/mmdetection/artifacts/report.pdf).

## Как запустить

1. Подготовить окружение через `setup_env.sh`.
2. Положить датасет в `datasets/minecraft/`.
3. Открыть и выполнить [notebook.ipynb](/Users/borisfox/Documents/ML/nn/sprint%205/project/mmdetection/notebook.ipynb).

Для запуска на GPU-ВМ удобнее копировать проект и датасет напрямую, а не хранить большие артефакты в Git.
