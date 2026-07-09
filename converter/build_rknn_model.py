import argparse
from pathlib import Path


def main():
    BASE_DIR = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Build RKNN FP16 model with 3 profiles")
    parser.add_argument("--onnx", default=str(BASE_DIR / "models" / "onnx" / "best.onnx"),help="Путь к исходному ONNX")
    parser.add_argument("--output", default=str(BASE_DIR / "models" / "rknn" / "best.rknn"),help="Путь для сохранения готового RKNN")
    parser.add_argument("--target", default="rk3588",help="Целевая платформа NPU")
    parser.add_argument("--max-size", type=int, default=1280, help="Максимальный размер (принимается из build.sh для совместимости)")
    
    args = parser.parse_args()

    if not Path(args.onnx).exists():
        raise FileNotFoundError(f"ONNX-модель не найдена: {args.onnx}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    from rknn.api import RKNN
    rknn = RKNN(verbose=False)

    rknn.config(
        target_platform=args.target,
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        dynamic_input=[
            [[1, 3, 320, 320]],    # Профиль #0 (Минимум)
            [[1, 3, 640, 640]],    # Профиль #1 (Стандарт)
            [[1, 3, 1280, 1280]],  # Профиль #2 (Максимум для FP16)
        ],
    )

    print(f"Загрузка ONNX: {args.onnx}")
    ret = rknn.load_onnx(model=args.onnx)
    if ret != 0:
        print("Ошибка загрузки ONNX модели!")
        return

    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print("Ошибка сборки модели!")
        return

    rknn.export_rknn(args.output)
    rknn.release()

    size = Path(args.output).stat().st_size / (1024 * 1024)
    print(f"Успешно сохранено: {args.output} ({size:.1f} MB)\n")

if __name__ == "__main__":
    main()
