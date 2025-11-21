#!/usr/bin/env python3

import argparse
import torch
from ultralytics import YOLO
import copy
import os

from torch.utils.data import DataLoader
from run_coco import COCODataset, collate_fn, COCO_DATASET_PATH

from torchao.quantization.pt2e.quantize_pt2e import (
    prepare_pt2e,
    convert_pt2e,
)

from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)

dataset_val = COCODataset(
    root_dir=os.path.join(COCO_DATASET_PATH, "val2017"),
    annotation_file=os.path.join(
        COCO_DATASET_PATH, "annotations", "instances_val2017.json"
    ),
)
dataloader_val = DataLoader(
    dataset_val,
    batch_size=16,
    shuffle=False,
    num_workers=1,
    collate_fn=collate_fn,
)


def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for image, target in data_loader:
            model(image)


def quantize_yolo_model(model_path, method):
    model = YOLO(model_path)
    example_inputs = (torch.randn(1, 3, 640, 640),)
    model = torch.export.export(model.model, example_inputs).module()

    quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
    model = prepare_pt2e(model, quantizer)

    # calibration omitted

    q_model = convert_pt2e(model)
    print(q_model)

    # Save quantized model
    quantized_model_path = f"quantized_{method}_{os.path.basename(model_path)}"
    # torch.save(q_model.state_dict(), quantized_model_path)
    torch.save(q_model.state_dict(), quantized_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize a YOLO model")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--quantization_method", type=str, default="weight_only_int4")

    args = parser.parse_args()
    quantize_yolo_model(args.model, args.quantization_method)
