#!/usr/bin/env python3

import argparse
import os
from ultralytics import YOLO
from pathlib import Path

FILE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_ROOT.parent
CURRENT_DIR = Path.cwd()


def export_onnx(model, output_dir, image_size=(640, 640)):
    print(f"Exporting model to ONNX with image size {image_size}...")

    # The ultralytics export function saves the model in the current directory
    # It returns the path to the exported model, but we'll build it ourselves
    # to be certain.
    model.export(
        format="onnx",
        imgsz=image_size,
        opset=17,
        dynamic=False,  # Export with static input dimensions
        simplify=True,  # Runs onnx-simplifier, which includes constant folding
    )

    # --- Move the generated file to the desired output directory ---
    # Construct the expected source path (in the current directory)
    model_name_pt = Path(model.ckpt_path).name
    onnx_file_name = model_name_pt.replace(".pt", ".onnx")
    src_path = CURRENT_DIR / onnx_file_name

    # Construct the destination path
    dest_dir = PROJECT_ROOT / output_dir
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = dest_dir / onnx_file_name

    # Move the file, overwriting if it exists
    if src_path.exists():
        os.replace(src_path, dest_path)
        print(f"Model successfully exported and moved to {dest_path}")
    else:
        print(f"ERROR: Exported ONNX file not found at expected path: {src_path}")


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
        default=[640, 640],
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
