#!/usr/bin/env python3

import argparse
import os
from ultralytics import YOLO
from pathlib import Path

FILE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_ROOT.parent
CURRENT_DIR = Path.cwd()


def export_onnx(model, output_dir, image_size=(480, 640)):
    model.export(format="onnx", imgsz=image_size, opset=13, dynamic=True, simplify=True)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Move all .onnx files to models
    for file in os.listdir(CURRENT_DIR):
        if file.endswith(".onnx"):
            os.replace(CURRENT_DIR / file, PROJECT_ROOT / output_dir / file)

    print(f"Model exported to {PROJECT_ROOT / output_dir}")


def get_yolo_pt(model_name="yolov8n.pt", image_size=(480, 640)):
    model = None

    # Check if model in output_dir
    os.makedirs(PROJECT_ROOT / "models", exist_ok=True)
    if os.path.exists(PROJECT_ROOT / "models" / model_name):
        model = YOLO(PROJECT_ROOT / "models" / model_name)
    else:
        model = YOLO(
            model_name,
        )  # Download model
        # Move model to models
        os.replace(
            CURRENT_DIR / model_name,
            PROJECT_ROOT / "models" / model_name,
        )
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Get YOLOv8 ONNX model")
    parser.add_argument(
        "--model", type=str, default="yolov8n.pt", help="YOLOv8 model to download"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[480, 640],
        help="Image size for the model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/",
        help="Directory to save the model. Relative to project root.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = get_yolo_pt(model_name=args.model, image_size=tuple(args.image_size))
    export_onnx(model, output_dir=args.output_dir, image_size=tuple(args.image_size))
