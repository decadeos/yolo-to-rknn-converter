import argparse
from pathlib import Path
from ultralytics import YOLO
import shutil 

def main():
    parser = argparse.ArgumentParser(description="Export YOLO .pt to ONNX with multiscale support")
    parser.add_argument("--weights", type=Path, default="../checkpoints/best.pt", help="Path to .pt weights file")
    parser.add_argument("--output", type=Path, default="../models/onnx/best.onnx", help="Output ONNX file path")
    parser.add_argument("--no-dynamic",action="store_true",help="Disable dynamic axes (fixed size only)")
    args = parser.parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"❌ Weights not found: {args.weights}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("YOLO to ONNX Exporter")
    print(f"Weights:  {args.weights}")
    print(f"Output:   {args.output}")
    print(f"Dynamic:  {'OFF' if args.no_dynamic else 'ON'}")

    model = YOLO(str(args.weights))

    export_path = model.export(
        format="onnx",
        imgsz=640,                      
        opset=12,
        simplify=True,
        dynamic=not args.no_dynamic,    # Мультимасштаб
        nms=False,
        device="cpu",
        verbose=True,
    )

    src = Path(export_path)
    if src != args.output:
        shutil.move(str(src), str(args.output))

    print(f"ONNX exported")
    print(f"File: {args.output.resolve()}")
    print(f"Size: {args.output.stat().st_size / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    main()