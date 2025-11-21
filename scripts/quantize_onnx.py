# Load onnx model
# Load calibration data from COCO
# Benchmark inference for model
# Quantize model using static quantization
# Benchmark inference for quantized model
# Save quantized model

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from torch.utils.data import DataLoader

# Add scripts directory to path to import from run_coco
FILE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_ROOT.parent
sys.path.append(str(FILE_ROOT))

# Now we can import from run_coco
from run_coco import COCODataset, collate_fn, get_coco_data

# --- Calibration Data Reader ---


class COCOCalibrationDataReader(CalibrationDataReader):
    def __init__(self, dataloader, model_path):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

        # Get the input name of the model
        model = onnx.load(model_path)
        self.input_name = model.graph.input[0].name
        print(f"Model input name: {self.input_name}")

    def get_next(self):
        try:
            # The dataloader yields (images_tensor, image_ids, original_shapes)
            images_tensor, _ = next(self.iterator)

            # Convert to numpy and create the input dictionary
            images_np = images_tensor.cpu().numpy()
            return {self.input_name: images_np}
        except StopIteration:
            return None


# --- Benchmarking ---


def benchmark(model_path, dataloader, num_images=100, device="cpu"):
    available_providers = ort.get_available_providers()
    providers = []
    if device == "cuda" and "CUDAExecutionProvider" in available_providers:
        print("Using CUDAExecutionProvider for ONNX Runtime.")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        if device == "cuda":
            print("CUDAExecutionProvider not available. Falling back to CPU.")
        print("Using CPUExecutionProvider for ONNX Runtime.")
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name

    total_time = 0
    num_processed = 0

    print(f"Benchmarking {model_path}...")

    for i, (images_tensor, _) in enumerate(dataloader):
        if num_processed >= num_images:
            break

        images_np = images_tensor.cpu().numpy()

        start_time = time.perf_counter()
        outputs = session.run(["output0"], {input_name: images_np})

        end_time = time.perf_counter()
        # Print output
        print(outputs)

        total_time += end_time - start_time
        num_processed += images_np.shape[0]

    avg_latency_ms = (total_time / num_processed) * 1000
    print(f"Average inference time: {avg_latency_ms:.4f} ms per image.")
    return avg_latency_ms


# --- Main ---


def main(args):
    # 1. Load calibration data from COCO
    print("Loading COCO validation dataset for calibration...")
    # Ensure dataset exists
    if not os.path.exists(args.coco_path):
        print("COCO dataset not found. Downloading...")
        get_coco_data(train=False, val=True, test=False, unlabeled=False)
        print("Download complete.")

    dataset_val = COCODataset(
        root_dir=os.path.join(args.coco_path, "val2017"),
        annotation_file=os.path.join(
            args.coco_path, "annotations", "instances_val2017.json"
        ),
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=1,  # Static quantization works with batch_size=1
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn,
    )

    # 2. Benchmark original model
    print("\n--- Benchmarking Original FP32 Model ---")
    benchmark(
        args.model_path, dataloader_val, device=args.device, num_images=args.num_images
    )

    # 3. Quantize model using static quantization
    print("\n--- Quantizing Model ---")
    calibration_data_reader = COCOCalibrationDataReader(dataloader_val, args.model_path)

    quantize_static(
        model_input=args.model_path,
        model_output=args.output_path,
        calibration_data_reader=calibration_data_reader,
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=False,
        reduce_range=True,
        nodes_to_exclude=[
            "/model.22/Concat_3",
            "/model.22/Split",
            "/model.22/dfl/Reshape",
            "/model.22/dfl/Transpose",
            "/model.22/dfl/Softmax",
            "/model.22/dfl/conv/Conv",
            "/model.22/dfl/Reshape_1",
            "/model.22/Slice",
            "/model.22/Slice_1",
            "/model.22/Sub",
            "/model.22/Add_1",
            "/model.22/Add_2",
            "/model.22/Sub_1",
            "/model.22/Div_1",
            "/model.22/Concat_4",
            "/model.22/Mul_2",
            "/model.22/Sigmoid",
            "/model.22/Concat_5",
        ],
    )
    print(f"Static quantized model saved to {args.output_path}")

    # 4. Benchmark quantized model
    print("\n--- Benchmarking Quantized INT8 Model ---")
    benchmark(
        args.output_path, dataloader_val, device=args.device, num_images=args.num_images
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ONNX YOLOv8 Static Quantization Script"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the FP32 ONNX model.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the INT8 quantized ONNX model.",
    )
    parser.add_argument(
        "--coco_path",
        type=str,
        default="datasets/coco",
        help="Path to the COCO dataset directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for benchmarking ('cpu' or 'cuda').",
    )

    parser.add_argument("--num_images", type=int, default=100)

    args = parser.parse_args()
    main(args)
