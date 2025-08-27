#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

auto main(int argc, char *argv[]) -> int {
    // Load the model and create InferenceSession
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntimeDemo");

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    // Optional: enable optimization
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // Path to your model
    const char *model_path = "../models/yolov8n.onnx";

    // Ort::Session session(env, model_path, session_options);
    Ort::Session session = Ort::Session(env, model_path, session_options);

    std::cout << "Model loaded successfully!" << std::endl;
    // Load model to onnx
    // Load image from disk
    // Run inference
    // Get output
    // Compute bounding boxes
    // Write image with bounding boxes

    // 1. Load image with OpenCV
    cv::Mat img = cv::imread("../images/test.jpg");
    if (img.empty()) {
        std::cerr << "Failed to load image\n";
        return -1;
    }

    // 2. Resize to model’s expected size (e.g. 640x640 for YOLOv8)
    int target_w = 640;
    int target_h = 640;
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(target_w, target_h));

    // 3. Convert to float32 and normalize to [0,1]
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    // 4. Convert HWC (OpenCV) → CHW (NCHW expected by ONNX)
    std::vector<float> input_tensor_values(target_w * target_h * 3);
    size_t index = 0;
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < resized.rows; ++y) {
            for (int x = 0; x < resized.cols; ++x) {
                input_tensor_values[index++] = resized.at<cv::Vec3f>(y, x)[c];
            }
        }
    }

    // 5. Create input tensor
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> input_shape = {1, 3, target_h, target_w};
    size_t input_tensor_size = input_tensor_values.size();

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_size,
        input_shape.data(), input_shape.size());

    // 6. Run inference
    const char *input_names[] = {"images"};   // check your model input name
    const char *output_names[] = {"output0"}; // check your model output name

    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names,
                                      &input_tensor, 1, output_names, 1);

    std::cout << "Inference done, got " << output_tensors.size()
              << " output tensors\n";

    // Output tensor values
    float *output_data = output_tensors[0].GetTensorMutableData<float>();
    size_t output_size =
        output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::cout << "Output tensor size: " << output_size << std::endl;
    for (size_t i = 0; i < std::min(output_size, size_t(10)); ++i) {
        std::cout << output_data[i] << " ";
    }
    return 0;
}
