#!/usr/bin/env python
import os
import argparse
import numpy as np

# 1. TVM Imports - Keep at the top
import tvm
from tvm import relax
from tvm import runtime as rt
import tvm.dlight as dl
from tvm.relax.frontend.onnx import from_onnx

# 2. Supporting imports
import onnx
from ultralytics import YOLO


# Fix for numpy versions that deprecated np.math
if not hasattr(np, "math"):
    import math

    np.math = math


def export_yolo_to_onnx(model_path):
    """Exports YOLOv8 model to ONNX format."""
    base_name = os.path.splitext(os.path.basename(model_path))[0]
    onnx_path = os.path.join(os.path.dirname(model_path), f"{base_name}.onnx")
    print(f"[INFO] Exporting {model_path} to {onnx_path}...")
    model = YOLO(model_path)
    # Simplify and static shapes are highly recommended for TVM
    model.export(format="onnx", simplify=True, dynamic=False)
    return onnx_path


def run_yolo_relax(model_path, device_type="cuda", quantize=False):
    input_shape = (1, 3, 640, 640)
    input_name = "images"

    # 1. Load ONNX model
    onnx_model = onnx.load(model_path)
    shape_dict = {input_name: input_shape}

    print(f"[INFO] Importing ONNX into TVM Relax for {device_type}...")
    mod = from_onnx(onnx_model, shape_dict)

    if device_type == "cuda":
        target = tvm.target.Target("cuda", host="llvm")
        dev = tvm.cuda(0)
    elif device_type == "opencl":
        target = tvm.target.Target("opencl", host="llvm")
        dev = tvm.opencl(0)
    elif device_type == "cpu":
        target = tvm.target.Target("llvm -mcpu=znver3")
        dev = tvm.cpu(0)
    else:
        print("No such device")

    print(f"[INFO] Applying Relax transformation passes...")

    # DLight handles the scheduling (mapping loops to GPU threads)
    # This prevents the 'Memory verification failed' error
    seq = tvm.transform.Sequential(
        [
            relax.transform.DecomposeOpsForInference(),
            relax.transform.LegalizeOps(),
            dl.ApplyDefaultSchedule(dl.gpu.Fallback()),
            relax.transform.FoldConstant(),
        ]
    )

    with target:
        mod = seq(mod)

    if quantize:
        print(
            "[WARNING] Quantization is not yet implemented for Relax. Continuing without quantization."
        )

    # 4. Build the Relax model
    print(f"[INFO] Compiling for Target: {target}")
    with tvm.transform.PassContext(opt_level=3):
        ex = relax.build(mod, target=target)

    # 5. Initialize Virtual Machine
    print("[INFO] Initializing Relax Virtual Machine...")
    vm = relax.VirtualMachine(ex, dev)

    # 6. Prepare Input
    data = np.random.uniform(0, 1, input_shape).astype("float32")
    tvm_data = rt.tensor(data, dev)

    print("[INFO] Warming up...")
    for _ in range(3):
        _ = vm["main"](tvm_data)

    print("[INFO] Running benchmark...")
    timer = vm.module.time_evaluator("main", dev, number=1, repeat=10)
    prof_res = timer(tvm_data)

    raw_output = vm["main"](tvm_data)

    if isinstance(raw_output, (list, tuple)):
        output_np = raw_output[0].numpy()
    else:
        output_np = raw_output.numpy()

    print(f"\n[SUCCESS] Inference finished!")
    print(f"Device: {device_type}")
    print(f"Mean Inference Time: {prof_res.mean * 1000:.4f} ms")
    print(f"FPS: {1.0 / prof_res.mean:.2f}")
    print(f"Output Shape: {output_np.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv8 on GPU using TVM Relax")
    parser.add_argument(
        "--model", type=str, default="yolov8n.pt", help="Path to .pt or .onnx"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="cuda",
        choices=["cuda", "opencl", "cpu"],
        help="Target device",
    )
    parser.add_argument(
        "--quantize", action="store_true", help="Enable Mixed Precision/Quantization."
    )

    args = parser.parse_args()

    if args.model.endswith(".pt"):
        onnx_file = export_yolo_to_onnx(args.model)
    else:
        onnx_file = args.model

    # 2. Run Inference
    try:
        run_yolo_relax(onnx_file, device_type=args.target, quantize=args.quantize)
    except Exception as e:
        print(f"\n[ERROR] Execution failed: {e}")
        print("\nTip: If you see 'AttributeError: module tvm has no attribute nd',")
        print("ensure your PYTHONPATH points to your 'tvm-upstream/python' folder")
        print("and that you have uninstalled any pip versions of tvm.")
