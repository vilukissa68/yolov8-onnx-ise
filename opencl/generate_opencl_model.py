import os
import glob
import onnx
import numpy as np
import tvm
from tvm import relay
from ultralytics import YOLO
import argparse
from tvm.contrib import cc


def export_yolo_to_onnx(model_path):
    """
    Exports YOLOv8 model to ONNX format.
    Automatically determines output name based on input (e.g., yolov8s.pt -> yolov8s.onnx).
    """
    # Derive ONNX filename from .pt filename
    base_name = os.path.splitext(os.path.basename(model_path))[0]
    onnx_path = os.path.join(os.path.dirname(model_path), f"{base_name}.onnx")

    print(f"[INFO] Exporting {model_path} to {onnx_path}...")

    # Initialize YOLO model
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(
            f"[ERROR] Could not load model {model_path}. Make sure ultralytics is installed and the file exists."
        )
        raise e

    # Export
    # dynamic=False is crucial for standard TVM AOT compilation
    model.export(format="onnx", simplify=True, dynamic=False)

    # Verify file exists (Ultralytics usually saves it next to the .pt file)
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX export failed. Expected file at {onnx_path}")

    return onnx_path


def get_calibration_dataset(shape, n_samples=10):
    """
    Generates dummy calibration data for quantization.
    In production, load real images here using cv2 or PIL.
    """
    print("[INFO] Generating calibration data...")
    dataset = []
    for _ in range(n_samples):
        # YOLOv8 expects normalized data (0-1) or 0-255 depending on preprocessing.
        # Usually ONNX export includes the normalization. We assume 0-255 inputs here.
        data = np.random.uniform(0, 255, size=shape).astype("float32")
        dataset.append({"images": data})
    return dataset


def quantize_model(mod, params, shape):
    """
    Quantizes the model to Int8 using TVM Relay.
    """
    print("[INFO] Quantizing model to Int8...")

    # Prepare calibration dataset
    calibration_samples = get_calibration_dataset(shape)

    # Calibration data generator
    def calibrate_dataset_gen():
        for sample in calibration_samples:
            yield sample

    # 'global_scale' is a simple strategy. 'kl_divergence' is better for accuracy but slower.
    with relay.quantize.qconfig(calibrate_mode="global_scale", global_scale=8.0):
        mod = relay.quantize.quantize(mod, params, dataset=calibrate_dataset_gen())

    return mod, params


def clean_code(src_dir):
    bad_strings = ['section(".rodata.tvm"),', 'section(".bss.noinit.tvm"),']

    # Find all C files in the directory
    c_files = glob.glob(os.path.join(src_dir, "*.c"))

    print(f"Cleaning {len(c_files)} files in {src_dir}...")

    for c_file in c_files:
        with open(c_file, "r") as f:
            content = f.read()

        original_len = len(content)

        # Remove the offending section attributes
        for bad in bad_strings:
            content = content.replace(bad, "")

        # Optional: Clean up empty attributes __attribute__(( )) if they are left behind
        # (GCC usually ignores them, but let's be tidy)
        content = content.replace("__attribute__(( ))", "")

        if len(content) != original_len:
            print(f"Fixed MacOS section errors in: {c_file}")
            with open(c_file, "w") as f:
                f.write(content)
        else:
            print(f"No changes needed for: {c_file}")

    print("Done.")


def generate_opencl_library(
    model_path,
    output_dir="tvm_opencl_out",
    quantize=False,
):
    input_shape = (1, 3, 640, 640)
    input_name = "images"

    # Load onnx
    onnx_model = onnx.load(model_path)
    shape_dict = {input_name: input_shape}

    print("[INFO] Importing ONNX into TVM Relay...")
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)

    # Quantization
    if quantize:
        mod, params = quantize_model(mod, params, input_shape)

    # Setup OpenCL target
    target = tvm.target.Target("opencl")
    dev = tvm.opencl(0)

    print("[INFO] Compiling for OpenCL GPU...")
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(
            mod,
            target=target,
            params=params,
        )

    # Export shared library with the model
    os.makedirs(output_dir, exist_ok=True)
    so_path = os.path.join(output_dir, "yolov8_opencl.so")
    lib.export_library(so_path, cc.create_shared)

    print("\n[SUCCESS] OpenCL compilation complete!")
    print(f"• Shared library: {so_path}")

    # Try to export raw OpenCL kernels as .cl file
    try:
        cl_src = lib.get_lib().get_source("opencl")
        cl_path = os.path.join(output_dir, "kernels.cl")
        with open(cl_path, "w") as f:
            f.write(cl_src)
        print(f"• OpenCL kernels: {cl_path}")
    except Exception:
        print("[INFO] Kernels embedded (no raw .cl emitted)")

    return so_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile YOLOv8 to OpenCL using TVM")

    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Path to the YOLOv8 input model (.pt). Default: yolov8n.pt",
    )

    # Argument: Output Directory
    parser.add_argument(
        "--output",
        type=str,
        default="tvm_c_out",
        help="Directory to save the generated C code. Default: tvm_c_out",
    )

    # Argument: Quantization Flag
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Enable Int8 Quantization. (Warning: Reduces accuracy significantly without real calibration data)",
    )

    args = parser.parse_args()

    onnx_file = export_yolo_to_onnx(args.model)

    generate_opencl_library(onnx_file, output_dir=args.output, quantize=args.quantize)
