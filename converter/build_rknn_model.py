#!/usr/bin/env python3
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="../models/onnx/best.onnx")
    parser.add_argument("--output", default="../models/rknn/best.rknn")
    parser.add_argument("--target", default="rk3588")
    parser.add_argument("--max-size", type=int, default=1280)
    args = parser.parse_args()

    if not Path(args.onnx).exists():
        raise FileNotFoundError(f"ONNX not found: {args.onnx}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    from rknn.api import RKNN
    rknn = RKNN(verbose=False)
    rknn.config(
        target_platform=args.target,
        dynamic_input=[[[1, 3, args.max_size, args.max_size]]]
    )

    rknn.load_onnx(model=args.onnx)
    rknn.build(do_quantization=False)
    rknn.export_rknn(args.output)
    rknn.release()

    size = Path(args.output).stat().st_size / (1024 * 1024)
    print(f"RKNN saved: {args.output} ({size:.1f} MB)")

if __name__ == "__main__":
    main()