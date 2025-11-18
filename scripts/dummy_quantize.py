#!/usr/bin/env python3
"""
Static (post-training) INT8 quantization example (eager mode) following
the PyTorch 2.8 quantization guide:
https://docs.pytorch.org/docs/2.8/quantization.html

Works for simple nn.Modules (DummyModel) and can be applied to yolo.model
(not the ultralytics.YOLO wrapper).

Usage:
    python static_int8_quantize.py
"""

import io
import os
import torch
import torch.nn as nn
import torch.quantization as tq
from typing import Optional
import copy


# ---------------------------
# Utility: choose quant engine
# ---------------------------
def choose_quantized_engine():
    print("PyTorch version:", torch.__version__)
    print("supported engines:", torch.backends.quantized.supported_engines)
    print("current engine:", torch.backends.quantized.engine)
    if "fbgemm" in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = "fbgemm"
        print("Selected quantized engine: fbgemm")
    elif "qnnpack" in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = "qnnpack"
        print("Selected quantized engine: qnnpack")
    else:
        raise RuntimeError(
            "No supported quantized backend (fbgemm/qnnpack) found in this PyTorch build."
        )


# ---------------------------
# Example model (you can substitute your model)
# ---------------------------
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # simple MLP - for conv models you'd use same flow
        self.fc1 = nn.Linear(16, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


# ---------------------------
# Static quantization flow
# ---------------------------
def static_int8_quantize(
    model: torch.nn.Module,
    calibration_loader: Optional[torch.utils.data.DataLoader] = None,
    num_calibration_batches: int = 10,
    inplace: bool = False,
):
    """
    Performs eager-mode static post-training quantization on `model`.

    Args:
        model: a torch.nn.Module in eval() mode.
               For Ultralytics YOLO, pass yolo.model (not the YOLO wrapper).
        calibration_loader: optional dataloader yielding input tensors for calibration.
        num_calibration_batches: if no loader provided, use this many random batches.
        inplace: whether to quantize in-place. Returns quantized model either way.

    Returns:
        quantized_model: the INT8 converted model (torch.nn.Module)
    """
    model = model if inplace else copy.deepcopy(model)  # deep copy if not inplace
    model.eval()

    # 1) Optionally fuse modules (Conv+BN+ReLU etc.). Only if your model defines fuse utilities.
    # If your model has a fuse_model method, run it. Many torchvision models provide it.
    if hasattr(model, "fuse_model"):
        try:
            model.fuse_model()
            print("Ran model.fuse_model()")
        except Exception as e:
            print("model.fuse_model() exists but failed or is not applicable:", e)

    # 2) Set qconfig for static quant (use fbgemm for x86)
    # Use a per-channel weight qconfig if available (better accuracy for convs)
    qconfig = tq.get_default_qconfig(torch.backends.quantized.engine)
    print("Using qconfig:", qconfig)
    model.qconfig = qconfig

    # 3) Prepare the model (insert observers)
    # eager-mode prepare:
    tq.prepare(model, inplace=True)
    print("Prepared model (observers inserted).")

    # 4) Calibration: run representative data through the prepared model
    if calibration_loader is not None:
        print("Running calibration using provided calibration_loader ...")
        with torch.no_grad():
            for i, batch in enumerate(calibration_loader):
                # Accept either (inputs) or (inputs, labels)
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                else:
                    inputs = batch
                # send inputs to same device as model parameters
                if next(model.parameters(), None) is not None:
                    dev = next(model.parameters()).device
                    inputs = inputs.to(dev)
                model(inputs)
    else:
        # Use synthetic random data for calibration (not ideal for accuracy)
        print(
            f"No calibration_loader provided. Running {num_calibration_batches} batches of random data for calibration (synthetic)."
        )
        in_shape = None
        # try to infer input shape from forward signature or first layer if available
        # fallback to [1, 16] for DummyModel
        for _ in range(num_calibration_batches):
            # For generality, assume the model accepts a single tensor input
            if in_shape is None:
                # Default for DummyModel
                in_shape = (1, 16)
            x = torch.randn(*in_shape).to("mps")
            model(x)

    print("Calibration finished.")

    # 5) Convert to quantized model (swap modules with quantized counterparts)
    tq.convert(model, inplace=True)
    print("Converted model to quantized INT8.")

    return model


# ---------------------------
# Small helper: compare outputs and size
# ---------------------------
def compare_model_outputs_and_size(model_fp32, model_int8, input_example=None):
    model_fp32.eval()
    model_int8.eval()
    if input_example is None:
        input_example = torch.randn(1, 16)

    with torch.no_grad():
        out_fp32 = model_fp32(input_example)
        out_int8 = model_int8(input_example)

    print("FP32 output:", out_fp32)
    print("INT8 output:", out_int8)
    # simple L2 difference
    diff = (out_fp32 - out_int8).abs().mean().item()
    print(f"Mean absolute difference FP32 vs INT8: {diff:.6f}")

    # size
    b_fp32 = io.BytesIO()
    b_int8 = io.BytesIO()
    torch.save(model_fp32.state_dict(), b_fp32)
    torch.save(model_int8.state_dict(), b_int8)
    size_fp32 = b_fp32.getbuffer().nbytes
    size_int8 = b_int8.getbuffer().nbytes
    print(
        f"State dict sizes: FP32={size_fp32 / 1024:.1f} KB, INT8={size_int8 / 1024:.1f} KB"
    )


# ---------------------------
# Demo main
# ---------------------------
def main():
    choose_quantized_engine()

    # Create float model (replace with your model or yolo.model)
    model_fp32 = DummyModel().eval()
    model_fp32.to("mps")
    print("=== Original FP32 model ===")
    print(model_fp32)
    print("Example param dtype:", next(model_fp32.parameters()).dtype)

    # Run static quantization (calibration with synthetic data)
    model_int8 = static_int8_quantize(
        model_fp32, calibration_loader=None, num_calibration_batches=20, inplace=False
    )

    print("=== Quantized model structure ===")
    print(model_int8)

    # Compare outputs/sizes
    compare_model_outputs_and_size(model_fp32, model_int8)

    # If you want to apply to a Ultralytics YOLO model:
    # from ultralytics import YOLO
    # y = YOLO("yolov8n.pt")
    # core = y.model  # core is torch.nn.Module
    # core.eval()
    # core_q = static_int8_quantize(core, calibration_loader=my_cal_loader)
    # torch.save(core_q.state_dict(), "yolo_static_int8_state_dict.pth")
    # note: saving/loading whole quantized model sometimes requires extra care


if __name__ == "__main__":
    main()
