import argparse
import shutil
from pathlib import Path
import torch                   
from ultralytics import YOLO

def main():
    BASE_DIR = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Export YOLO .pt to ONNX with explicit dynamic input")
    parser.add_argument("--weights", type=Path, default=BASE_DIR / "checkpoints" / "best.pt")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "models" / "onnx" / "best.onnx")
    args = parser.parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("=== [1/2] YOLO to ONNX Exporter (Explicit Dynamic Input) ===")
    
    yolo_model = YOLO(str(args.weights))
    torch_model = yolo_model.model.eval() 
    fake_input = torch.randn(1, 3, 1280, 1280)

    torch.onnx.export(
        torch_model,
        fake_input,
        str(args.output),
        opset_version=12,
        do_constant_folding=True,
        input_names=['images'],
        output_names=['output0'],
        dynamic_axes={
            'images': {2: 'height', 3: 'width'} 
        }
    )

    print(f"ONNX успешно экспортирован с чистыми осями: {args.output.resolve()}\n")

if __name__ == "__main__":
    main()
