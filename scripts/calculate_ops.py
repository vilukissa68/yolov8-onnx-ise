#!/usr/bin/env python3

import argparse
import onnx_tool


def profile_onnx_model(onnx_model_path):
    """
    Profiles an ONNX model to calculate FLOPs, parameters, and other stats.
    """
    print(f"Profiling ONNX model: {onnx_model_path}")
    # The onnx_tool.model_profile function prints a summary table to the console.
    onnx_tool.model_profile(onnx_model_path, save_profile="onnx_profile.csv")
    print("\nDetailed profile saved to onnx_profile.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate FLOPs and Params of an ONNX model"
    )
    parser.add_argument(
        "--onnx_model",
        type=str,
        required=True,
        help="Path to the ONNX model file.",
    )

    args = parser.parse_args()
    profile_onnx_model(args.onnx_model)
