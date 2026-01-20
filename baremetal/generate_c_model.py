import os
import glob
import onnx
import numpy as np
import tvm
from tvm import relay
from tvm.micro import export_model_library_format
from ultralytics import YOLO
import argparse


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

    # 1. Create calibration data
    calibration_samples = get_calibration_dataset(shape)

    # 2. Define calibration logic
    def calibrate_dataset_gen():
        for sample in calibration_samples:
            yield sample

    # 3. Configure Quantization
    # 'global_scale' is a simple strategy. 'kl_divergence' is better for accuracy but slower.
    with relay.quantize.qconfig(calibrate_mode="global_scale", global_scale=8.0):
        mod = relay.quantize.quantize(mod, params, dataset=calibrate_dataset_gen())

    return mod, params


def clean_code(src_dir):
    # The specific section tags TVM generates that macOS hates
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


def generate_c_source(model_path, output_dir="tvm_c_out", quantize=False):
    """
    Main function to compile YOLOv8 to C source.
    """
    input_shape = (1, 3, 640, 640)  # Standard YOLOv8n resolution
    input_name = "images"  # Default YOLOv8 ONNX input name

    # 1. Load ONNX Model
    onnx_model = onnx.load(model_path)
    shape_dict = {input_name: input_shape}

    print("[INFO] Importing ONNX into TVM Relay...")
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    # 2. (Optional) Quantization
    if quantize:
        mod, params = quantize_model(mod, params, input_shape)

    # 3. Define Compilation Targets
    # Target "c" generates C source code for operators
    # We use the AOT (Ahead of Time) executor for bare-metal C generation
    target = tvm.target.Target("c")

    # Runtime Configuration: CRT (C Runtime) for embedded/standalone
    runtime = tvm.relay.backend.Runtime("crt", {"system-lib": False})

    # Executor Configuration: AOT
    # unpacked-api=1: generates cleaner C function signatures
    # interface-api="c": generates a C-compatible header entry point
    executor = tvm.relay.backend.Executor(
        "aot",
        {
            "interface-api": "c",
            "unpacked-api": True,
            "link-params": True,  # Embed weights directly into the C code (Warning: Large file size)
        },
    )

    print("[INFO] Compiling (this may take a moment)...")
    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        lib = tvm.relay.build(
            mod, target, executor=executor, runtime=runtime, params=params
        )

    # 4. Save Output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the main C source (Operator implementations)
    # The 'lib' object contains a C-Module because we targeted "c"
    c_source = lib.get_lib().get_source()

    c_file_path = os.path.join(output_dir, "yolov8_impl.c")
    with open(c_file_path, "w") as f:
        f.write(c_source)

    # Save the Library (includes headers and metadata)
    # Model Library Format (MLF) is best for embedded integration,
    # but here we export a generic tar for inspection
    model_library_name = "yolov8_lib.tar"
    lib_file_path = os.path.join(output_dir, model_library_name)
    # lib.export_library(lib_file_path)
    export_model_library_format(lib, lib_file_path)

    print(f"\n[SUCCESS] Conversion complete!")
    print(f"1. C Implementation: {c_file_path}")
    print(f"2. Full Library Archive: {lib_file_path}")

    # Untar the library to get headers and params as hex dumps
    print("[INFO] Extracting library contents...")
    owd = os.getcwd()
    os.chdir(output_dir)
    os.system(f"tar -xvf {model_library_name}")
    os.chdir(owd)

    print("[INFO] Extracted library contents.")
    clean_code(os.path.join(output_dir, "codegen", "host", "src"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile YOLOv8 to C-Source using TVM")

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

    # 1. Export YOLOv8 to ONNX
    # We pass the user provided model path
    onnx_file = export_yolo_to_onnx(args.model)

    # 2. Convert to C-Source
    generate_c_source(onnx_file, output_dir=args.output, quantize=args.quantize)
