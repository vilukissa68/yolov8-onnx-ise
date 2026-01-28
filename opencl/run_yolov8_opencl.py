#!/usr/bin/env python
import os
import argparse
import numpy as np
import onnx
import time
import sys

# 1. TVM Imports (RELAY ONLY)
import tvm
from tvm import relay
from tvm.contrib import graph_executor

# 2. Quantization Imports
try:
    from onnxruntime.quantization import (
        quantize_static,
        CalibrationDataReader,
        QuantType,
        QuantFormat,
    )

    HAS_ORT = True
except ImportError:
    HAS_ORT = False
    print("[WARN] onnxruntime-quantization not found. Int8 calibration will fail.")

from ultralytics import YOLO

# Fix for numpy deprecation
if not hasattr(np, "math"):
    import math

    np.math = math


# ==============================================================================
# 1. Calibration Data Reader
# ==============================================================================
class YoloCalibrationDataReader(CalibrationDataReader):
    def __init__(self, model_path, n_samples=20):
        self.model_path = model_path
        self.enum_data = None
        self.n_samples = n_samples
        self.input_name = "images"
        self.input_shape = (1, 3, 640, 640)
        self.cnt = 0

    def get_next(self):
        if self.cnt >= self.n_samples:
            return None
        self.cnt += 1
        # Random data for calibration
        data = np.random.uniform(0.0, 1.0, size=self.input_shape).astype("float32")
        return {self.input_name: data}


# ==============================================================================
# 2. Quantization Logic (ONNX Runtime -> Q-ONNX)
# ==============================================================================
def perform_int8_quantization(onnx_path):
    if not HAS_ORT:
        raise ImportError(
            "Install 'onnxruntime' and 'onnxruntime-quantization' for Int8."
        )

    q_model_path = onnx_path.replace(".onnx", "_int8.onnx")

    if os.path.exists(q_model_path):
        print(f"[INFO] Found existing Int8 model: {q_model_path}")
        return q_model_path

    print(f"[INFO] Calibrating and Quantizing {onnx_path}...")
    dr = YoloCalibrationDataReader(onnx_path)

    # QDQ Format is critical for TVM Relay
    quantize_static(
        model_input=onnx_path,
        model_output=q_model_path,
        calibration_data_reader=dr,
        quant_format=QuantFormat.QDQ,
        per_channel=False,
        weight_type=QuantType.QUInt8,
        activation_type=QuantType.QUInt8,
    )
    print(f"[INFO] Int8 Model saved to: {q_model_path}")
    return q_model_path


# ==============================================================================
# 3. TVM RELAY Compiler (Stable)
# ==============================================================================
def compile_model_relay(model_path, target_str, device, input_shape):
    print(f"[INFO] Loading ONNX model: {model_path}")
    onnx_model = onnx.load(model_path)

    shape_dict = {"images": input_shape}

    # Import into Relay
    # Relay's frontend is mature and handles numpy.int64 types correctly
    print(f"[INFO] Importing into TVM Relay...")
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    # Define Target
    if target_str == "opencl":
        # -device=mali is a good default hint for mobile GPUs, typically harmless on others
        target = tvm.target.Target("opencl", host="llvm")
    elif target_str == "cuda":
        target = tvm.target.Target("cuda", host="llvm")
    elif target_str == "cpu":
        # Optimized CPU instructions
        target = tvm.target.Target("llvm")
    else:
        raise ValueError(f"Unsupported target: {target_str}")

    print(f"[INFO] Compiling for target: {target}")

    # Build with optimization level 3
    # This handles both FP32 and Int8 fusion automatically
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

    # Create the runtime module
    m = graph_executor.GraphModule(lib["default"](device))
    return m


# ==============================================================================
# 4. Main Runner
# ==============================================================================
def run_benchmark(args):
    input_shape = (1, 3, 640, 640)

    # 1. Device Setup
    if args.target == "cuda":
        dev = tvm.cuda(0)
    elif args.target == "opencl":
        dev = tvm.opencl(0)
    elif args.target == "cpu":
        dev = tvm.cpu(0)
    else:
        print(f"[ERROR] Unsupported target: {args.target}")
        sys.exit(1)

    # 2. Pipeline Selection
    final_model_path = args.model

    if args.quantize:
        print("[INFO] Mode: INT8 Quantization")
        # Generate or Load Q-ONNX
        final_model_path = perform_int8_quantization(args.model)
    else:
        print("[INFO] Mode: FP32 (Standard)")

    # 3. Compile (Using Relay for EVERYTHING)
    executor = compile_model_relay(final_model_path, args.target, dev, input_shape)

    # 4. Data Prep
    data_np = np.random.uniform(0, 1, input_shape).astype("float32")
    tvm_data = tvm.nd.array(data_np, dev)

    # 5. Benchmark Loop
    print(f"[INFO] Warming up...")
    executor.set_input("images", tvm_data)
    for _ in range(3):
        executor.run()

    print(f"[INFO] Benchmarking...")
    # Sync before start if GPU
    if args.target in ["cuda", "opencl"]:
        dev.sync()

    start = time.time()
    for _ in range(50):
        executor.run()
        # Sync required for accurate GPU timing
        if args.target in ["cuda", "opencl"]:
            dev.sync()
    end = time.time()

    avg = (end - start) / 50.0

    print(f"\n[SUCCESS] Finished!")
    print(f"Model:   {final_model_path}")
    print(f"Target:  {args.target}")
    print(f"Avg:     {avg * 1000:.4f} ms")
    print(f"FPS:     {1.0 / avg:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument(
        "--target", type=str, default="cuda", choices=["cuda", "opencl", "cpu"]
    )
    parser.add_argument("--quantize", action="store_true", help="Use Int8 Quantization")
    args = parser.parse_args()

    # Auto-export .pt to .onnx
    if args.model.endswith(".pt"):
        print(f"[INFO] Exporting {args.model} to ONNX...")
        model = YOLO(args.model)
        # opset=12 is historically more stable for Relay, but 13 works for most recent versions
        model.export(format="onnx", simplify=True, dynamic=False, opset=12)
        args.model = args.model.replace(".pt", ".onnx")

    run_benchmark(args)
