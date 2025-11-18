#!/usr/bin/env python3

import argparse
import torch
from ultralytics import YOLO
from torchao.quantization import Int4WeightOnlyConfig, quantize_


def quantize_weight_only_int4(yolo_model):
    core_model = yolo_model.model  # access underlying nn.Module
    quantize_(
        core_model,
        Int4WeightOnlyConfig(
            group_size=32,
            version=1,
            int4_packing_format="tile_packed_to_4d",
            int4_choose_qparams_algorithm="hqq",
        ),
    )
    print("Weight-only INT4 quantization completed.")
    for name, param in core_model.named_parameters():
        print(f"Parameter: {name}, Values: {param.flatten()[:10]}")
    return core_model


def quantize_yolo_model(model_path, method):
    yolo_model = YOLO(model_path)

    if method == "weight_only_int4":
        core_model_q = quantize_weight_only_int4(yolo_model.model)
    else:
        raise ValueError(f"Unsupported quantization method: {method}")

    torch.save(core_model_q.state_dict(), f"quantized_{method}_yolo.pth")
    print(f"Saved quantized model to quantized_{method}_yolo.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize a YOLO model")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--quantization_method", type=str, default="weight_only_int4")

    args = parser.parse_args()
    quantize_yolo_model(args.model, args.quantization_method)
