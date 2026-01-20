#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace tvm::runtime;

const int INPUT_W = 640;
const int INPUT_H = 640;
const int NUM_CLASSES = 80;              // COCO classes
const int NUM_CANDIDATES = 8400;         // 640x640 grid
const int OUTPUT_ATTR = 4 + NUM_CLASSES; // 84
const float CONF_THRESHOLD = 0.25f;
const float IOU_THRESHOLD = 0.45f;

// COCO 2017 Class Names
const std::vector<std::string> COCO_CLASSES = {
    "person",        "bicycle",      "car",
    "motorcycle",    "airplane",     "bus",
    "train",         "truck",        "boat",
    "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench",        "bird",
    "cat",           "dog",          "horse",
    "sheep",         "cow",          "elephant",
    "bear",          "zebra",        "giraffe",
    "backpack",      "umbrella",     "handbag",
    "tie",           "suitcase",     "frisbee",
    "skis",          "snowboard",    "sports ball",
    "kite",          "baseball bat", "baseball glove",
    "skateboard",    "surfboard",    "tennis racket",
    "bottle",        "wine glass",   "cup",
    "fork",          "knife",        "spoon",
    "bowl",          "banana",       "apple",
    "sandwich",      "orange",       "broccoli",
    "carrot",        "hot dog",      "pizza",
    "donut",         "cake",         "chair",
    "couch",         "potted plant", "bed",
    "dining table",  "toilet",       "tv",
    "laptop",        "mouse",        "remote",
    "keyboard",      "cell phone",   "microwave",
    "oven",          "toaster",      "sink",
    "refrigerator",  "book",         "clock",
    "vase",          "scissors",     "teddy bear",
    "hair drier",    "toothbrush"};

struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
};

float calculateIoU(const cv::Rect &box1, const cv::Rect &box2) {
    int x1 = std::max(box1.x, box2.x);
    int y1 = std::max(box1.y, box2.y);
    int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    int y2 = std::min(box1.y + box1.height, box2.y + box2.height);

    int interArea = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    float box1Area = box1.width * box1.height;
    float box2Area = box2.width * box2.height;

    return (float)interArea / (box1Area + box2Area - interArea);
}

void preprocess_image(const cv::Mat &img, float *data) {
    cv::Mat input_image;
    // Resize to 640x640
    cv::resize(img, input_image, cv::Size(INPUT_W, INPUT_H));
    // Convert Color (OpenCV is BGR, Model is RGB)
    cv::cvtColor(input_image, input_image, cv::COLOR_BGR2RGB);

    // Normalize & Layout Transform (HWC -> CHW)
    // YOLOv8 models usually expect values 0.0 - 1.0 (float32)
    input_image.convertTo(input_image, CV_32F, 1.0 / 255.0);

    // Pointer arithmetic to fill the flat array in CHW order
    // RRRRR... GGGGG... BBBBB...
    int image_size = INPUT_W * INPUT_H;
    for (int h = 0; h < INPUT_H; ++h) {
        for (int w = 0; w < INPUT_W; ++w) {
            cv::Vec3f pixel = input_image.at<cv::Vec3f>(h, w);
            data[0 * image_size + h * INPUT_W + w] = pixel[0]; // R
            data[1 * image_size + h * INPUT_W + w] = pixel[1]; // G
            data[2 * image_size + h * INPUT_W + w] = pixel[2]; // B
        }
    }
}

std::vector<Detection> postprocess(float *output,
                                   const cv::Size &original_size) {
    std::vector<Detection> detections;

    // Scaling factors to map 640x640 box back to original image
    float x_factor = (float)original_size.width / INPUT_W;
    float y_factor = (float)original_size.height / INPUT_H;

    // The output tensor is [1, 84, 8400]
    // Stride to jump between rows (attributes)
    // row 0: x, row 1: y, row 2: w, row 3: h, row 4+: class scores
    int stride = NUM_CANDIDATES;

    for (int i = 0; i < NUM_CANDIDATES; ++i) {
        // Find the maximum class score
        float max_score = -1.0f;
        int class_id = -1;

        // Loop through class scores (Row 4 to 83)
        for (int c = 0; c < NUM_CLASSES; ++c) {
            float score = output[(4 + c) * stride + i];
            if (score > max_score) {
                max_score = score;
                class_id = c;
            }
        }

        if (max_score >= CONF_THRESHOLD) {
            // Extract box coordinates (Row 0 to 3)
            float cx = output[0 * stride + i];
            float cy = output[1 * stride + i];
            float w = output[2 * stride + i];
            float h = output[3 * stride + i];

            // Convert center-x, center-y, width, height to top-left x, y, w, h
            int left = (int)((cx - 0.5 * w) * x_factor);
            int top = (int)((cy - 0.5 * h) * y_factor);
            int width = (int)(w * x_factor);
            int height = (int)(h * y_factor);

            detections.push_back(
                {class_id, max_score, cv::Rect(left, top, width, height)});
        }
    }

    // Apply Non-Maximum Suppression (NMS)
    std::sort(detections.begin(), detections.end(),
              [](const Detection &a, const Detection &b) {
                  return a.confidence > b.confidence;
              });

    std::vector<Detection> nms_result;
    std::vector<bool> suppressed(detections.size(), false);

    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i])
            continue;
        nms_result.push_back(detections[i]);

        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (calculateIoU(detections[i].box, detections[j].box) >
                IOU_THRESHOLD) {
                suppressed[j] = true;
            }
        }
    }

    return nms_result;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <image_path> <lib_path>"
                  << std::endl;
        return 1;
    }

    std::string image_path = argv[1];
    std::string lib_path = argv[2];

    // Load input image
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Could not open image: " << image_path << std::endl;
        return 1;
    }

    // Load TVM model
    std::cout << "Loading module: " << lib_path << std::endl;
    Module mod = Module::LoadFromFile(lib_path);

    DLDevice dev = {kDLOpenCL, 0};
    Module gmod = mod.GetFunction("default")(dev);
    PackedFunc set_input = gmod.GetFunction("set_input");
    PackedFunc run = gmod.GetFunction("run");
    PackedFunc get_output = gmod.GetFunction("get_output");

    // Allocate input memory
    std::vector<int64_t> input_shape = {1, 3, INPUT_H, INPUT_W};
    DLDataType dtype{kDLFloat, 32, 1};

    DLDevice cpu_dev = {kDLCPU, 0};
    NDArray input_cpu = NDArray::Empty(input_shape, dtype, cpu_dev);
    NDArray input_gpu = NDArray::Empty(input_shape, dtype, dev);

    // Preprocess image
    float *cpu_data = static_cast<float *>(input_cpu->data);
    preprocess_image(img, cpu_data);

    // Copy to OpenCL device
    input_gpu.CopyFrom(input_cpu);

    // Run inference
    set_input("images", input_gpu);
    run();

    // Read output
    NDArray out_gpu = get_output(0);
    NDArray out_cpu = out_gpu.CopyTo(cpu_dev);
    float *output_data = static_cast<float *>(out_cpu->data);

    // Postprocess and draw bounding boxes
    std::vector<Detection> results = postprocess(output_data, img.size());

    std::cout << "Detected " << results.size() << " objects." << std::endl;

    for (const auto &det : results) {
        cv::rectangle(img, det.box, cv::Scalar(0, 255, 0), 2);

        std::string label = "Class " + COCO_CLASSES[det.class_id] + ": " +
                            std::to_string(det.confidence);
        cv::putText(img, label, cv::Point(det.box.x, det.box.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);

        std::cout << label << " " << det.box << std::endl;
    }

    // Save result image to disk
    cv::imwrite("result.jpg", img);
    std::cout << "Result saved to result.jpg" << std::endl;

    return 0;
}
