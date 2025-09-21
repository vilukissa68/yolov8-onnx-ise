// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#include <iostream>
#include <iomanip>
#include "inference.h"
#include <filesystem>
#include <fstream>
#include <random>

constexpr auto image_path = "../images/test.jpg";

void Detector(YOLO_V8*& p) {
    std::filesystem::path current_path = std::filesystem::current_path();
	cv::Mat img = cv::imread(image_path);
	std::vector<DL_RESULT> res;
	p->RunSession(img, res);

	for (auto& re : res)
	{
		cv::RNG rng(cv::getTickCount());
		cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

		cv::rectangle(img, re.box, color, 3);

		float confidence = floor(100 * re.confidence) / 100;
		std::cout << std::fixed << std::setprecision(2);
		std::string label = p->classes[re.classId] + " " +
			std::to_string(confidence).substr(0, std::to_string(confidence).size() - 4);

		cv::rectangle(
			img,
			cv::Point(re.box.x, re.box.y - 25),
			cv::Point(re.box.x + label.length() * 15, re.box.y),
			color,
			cv::FILLED
		);

		cv::putText(
			img,
			label,
			cv::Point(re.box.x, re.box.y - 5),
			cv::FONT_HERSHEY_SIMPLEX,
			0.75,
			cv::Scalar(0, 0, 0),
			2
		);
	}
	cv::imwrite("result.jpg", img);
}


int ReadCocoYaml(YOLO_V8*& p) {
    // Open the YAML file
    std::ifstream file("coco.yaml");
    if (!file.is_open())
    {
        std::cerr << "Failed to open file" << std::endl;
        return 1;
    }

    // Read the file line by line
    std::string line;
    std::vector<std::string> lines;
    while (std::getline(file, line))
    {
        lines.push_back(line);
    }

    // Find the start and end of the names section
    std::size_t start = 0;
    std::size_t end = 0;
    for (std::size_t i = 0; i < lines.size(); i++)
    {
        if (lines[i].find("names:") != std::string::npos)
        {
            start = i + 1;
        }
        else if (start > 0 && lines[i].find(':') == std::string::npos)
        {
            end = i;
            break;
        }
    }

    // Extract the names
    std::vector<std::string> names;
    for (std::size_t i = start; i < end; i++)
    {
        std::stringstream ss(lines[i]);
        std::string name;
        std::getline(ss, name, ':'); // Extract the number before the delimiter
        std::getline(ss, name); // Extract the string after the delimiter
        names.push_back(name);
    }

    p->classes = names;
    return 0;
}


void DetectTest()
{
    YOLO_V8* yoloDetector = new YOLO_V8;
    ReadCocoYaml(yoloDetector);
    DL_INIT_PARAM params;
    params.rectConfidenceThreshold = 0.1;
    params.iouThreshold = 0.5;
    params.modelPath = "../models/yolov8n.onnx";
    params.imgSize = { 640, 640 };
    params.modelType = YOLO_DETECT_V8;
    params.cudaEnable = false;
    yoloDetector->CreateSession(params);
    Detector(yoloDetector);
}


int main()
{
	DetectTest();
}
