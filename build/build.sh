#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
CONVERTER_DIR="$PROJECT_ROOT/converter"

MAX_SIZE="${1:-1280}"
TARGET="${2:-rk3588}"

if [ ! -d "$VENV_DIR" ]; then
    echo "Создание виртуального окружения..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "Обновление базовых пакетов..."
pip install --upgrade pip setuptools wheel
pip install -r "$SCRIPT_DIR/requirements.txt"
pip install rknn-toolkit2 --no-deps

cd "$CONVERTER_DIR"

echo "Запуск шага 1: Экспорт в ONNX..."
python build_onnx_model.py

echo "Запуск шага 2: Сборка в RKNN..."
python build_rknn_model.py --target "$TARGET" --max-size "$MAX_SIZE"

ls -lh "$PROJECT_ROOT/models/onnx/best.onnx"
ls -lh "$PROJECT_ROOT/models/rknn/best.rknn"