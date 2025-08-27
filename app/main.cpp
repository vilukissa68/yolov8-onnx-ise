#include <iostream>
#include <onnxruntime_cxx_api.h>

auto main(int argc, char *argv[]) -> int {
    // Load the model and create InferenceSession
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntimeDemo");

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    // Optional: enable optimization
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // Path to your model
    const char *model_path = "models/yolon.onnx";

    Ort::Session session(env, model_path, session_options);

    std::cout << "Model loaded successfully!" << std::endl;
    // Load model to onnx
    // Load image from disk
    // Run inference
    // Get output
    // Compute bounding boxes
    // Write image with bounding boxes

    return 0;
}
