#!/usr/bin/env python3

import torch
import argparse
from ultralytics import YOLO
from ptflops import get_model_complexity_info
from torch.profiler import profile, record_function, ProfilerActivity
from thop import profile as thop_profile
from thop import clever_format


def get_flops(model, input_size=(1, 3, 640, 640)):
    macs, params = get_model_complexity_info(
        model,
        input_size[1:],
        as_strings=True,
        print_per_layer_stat=False,
        verbose=False,
    )
    print(f"FLOPs: {macs}, Params: {params}")
    return macs, params


def get_macs(model, input_size=(1, 3, 640, 640)):
    dummy_input = torch.randn(input_size)
    macs, params = thop_profile(model, inputs=(dummy_input,), verbose=True)
    macs, params = clever_format([macs, params], "%.3f")
    print(f"MACs: {macs}, Params: {params}")
    return macs, params


def profile_model(model, input_size=(1, 3, 640, 640)):
    dummy_input = torch.randn(input_size)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        with record_function("model_inference"):
            model(dummy_input)

    print(prof.key_averages().table(sort_by="flops", row_limit=10))


def find_largest_layer(model, input_size=(1, 3, 640, 640)):
    dummy_input = torch.randn(input_size)
    layer_flops = {}

    def hook_fn_linear(module, input, output):
        flops = 0
        if hasattr(module, "weight"):
            flops += module.weight.numel() * output.numel() // module.out_features
        layer_flops[module] = flops

    def hook_fn_conv(module, input, output):
        flops = 0
        if hasattr(module, "weight"):
            out_channels, in_channels, kH, kW = module.weight.shape
            output_dims = output.shape[2:]
            flops += (
                out_channels * in_channels * kH * kW * output_dims[0] * output_dims[1]
            )
        layer_flops[module] = flops

    hooks = []
    for module in model.modules():
        if not isinstance(module, torch.nn.Sequential) and not isinstance(
            module, torch.nn.ModuleList
        ):
            if isinstance(module, torch.nn.Linear):
                hooks.append(module.register_forward_hook(hook_fn_linear))
            elif isinstance(module, torch.nn.Conv2d):
                hooks.append(module.register_forward_hook(hook_fn_conv))

    model(dummy_input)

    for hook in hooks:
        hook.remove()

    largest_layer = max(layer_flops, key=layer_flops.get)
    print(
        f"Largest layer: {largest_layer.__class__.__name__} with FLOPs: {layer_flops[largest_layer]}"
    )
    return largest_layer, layer_flops[largest_layer]


def profile_model_onnx(model, input_shape=(1, 3, 640, 640), onnx_file="model.onnx"):
    import onnx_tool

    dummy_input = torch.randn(input_shape)
    tmp_file = onnx_file
    with torch.no_grad():
        exported_model = torch.onnx.export(
            model,
            dummy_input,
            tmp_file,
            opset_version=12,
            do_constant_folding=True,
        )
        onnx_tool.model_profile(tmp_file, save_profile="onnx_profile.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get FLOPs and Params of a YOLO model")
    parser.add_argument("--model", type=str, default="yolov8n.pt")

    args = parser.parse_args()

    yolo_model = YOLO(args.model)
    flops, params = get_flops(yolo_model.model)
    profile_model(yolo_model.model)
    macs, params_thop = get_macs(yolo_model.model)
    largest_layer, largest_flops = find_largest_layer(yolo_model.model)
    profile_model_onnx(model=yolo_model.model, onnx_file="yolo_model.onnx")
