// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#include "inference.h"
#include <iostream>
#include <regex>

#define benchmark
#define min(a, b) (((a) < (b)) ? (a) : (b))
YOLO_V8::YOLO_V8() {}

YOLO_V8::~YOLO_V8() { delete session; }

template <typename T> char *BlobFromImage(cv::Mat &iImg, T &iBlob) {
    int channels = iImg.channels();
    int imgHeight = iImg.rows;
    int imgWidth = iImg.cols;

    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < imgHeight; h++) {
            for (int w = 0; w < imgWidth; w++) {
                iBlob[c * imgWidth * imgHeight + h * imgWidth + w] =
                    typename std::remove_pointer<T>::type(
                        (iImg.at<cv::Vec3b>(h, w)[c]) / 255.0f);
            }
        }
    }
    return RET_OK;
}

char *YOLO_V8::PreProcess(cv::Mat &iImg, std::vector<int> iImgSize,
                          cv::Mat &oImg) {
    if (iImg.channels() == 3) {
        oImg = iImg.clone();
        cv::cvtColor(oImg, oImg, cv::COLOR_BGR2RGB);
    } else {
        cv::cvtColor(iImg, oImg, cv::COLOR_GRAY2RGB);
    }

    switch (modelType) {
    case YOLO_DETECT_V8:
    case YOLO_POSE:
    case YOLO_DETECT_V8_HALF:
    case YOLO_POSE_V8_HALF: // LetterBox
    {
        if (iImg.cols >= iImg.rows) {
            resizeScales = iImg.cols / (float)iImgSize.at(0);
            cv::resize(oImg, oImg,
                       cv::Size(iImgSize.at(0), int(iImg.rows / resizeScales)));
        } else {
            resizeScales = iImg.rows / (float)iImgSize.at(0);
            cv::resize(oImg, oImg,
                       cv::Size(int(iImg.cols / resizeScales), iImgSize.at(1)));
        }
        cv::Mat tempImg =
            cv::Mat::zeros(iImgSize.at(0), iImgSize.at(1), CV_8UC3);
        oImg.copyTo(tempImg(cv::Rect(0, 0, oImg.cols, oImg.rows)));
        oImg = tempImg;
        break;
    }
    }
    return RET_OK;
}

char *YOLO_V8::CreateSession(DL_INIT_PARAM &iParams) {
    char *Ret = RET_OK;
    std::regex pattern("[\u4e00-\u9fa5]");
    bool result = std::regex_search(iParams.modelPath, pattern);
    if (result) {
        Ret = "[YOLO_V8]:Your model path is error.Change your model path "
              "without chinese characters.";
        std::cout << Ret << std::endl;
        return Ret;
    }
    try {
        rectConfidenceThreshold = iParams.rectConfidenceThreshold;
        iouThreshold = iParams.iouThreshold;
        imgSize = iParams.imgSize;
        modelType = iParams.modelType;
        cudaEnable = iParams.cudaEnable;
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Yolo");
        Ort::SessionOptions sessionOption;
        if (iParams.cudaEnable) {
            OrtCUDAProviderOptions cudaOption;
            cudaOption.device_id = 0;
            sessionOption.AppendExecutionProvider_CUDA(cudaOption);
        }
        sessionOption.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);
        sessionOption.SetIntraOpNumThreads(iParams.intraOpNumThreads);
        sessionOption.SetLogSeverityLevel(iParams.logSeverityLevel);

        const char *modelPath = iParams.modelPath.c_str();

        session = new Ort::Session(env, modelPath, sessionOption);
        Ort::AllocatorWithDefaultOptions allocator;
        size_t inputNodesNum = session->GetInputCount();
        for (size_t i = 0; i < inputNodesNum; i++) {
            Ort::AllocatedStringPtr input_node_name =
                session->GetInputNameAllocated(i, allocator);
            char *temp_buf = new char[50];
            strcpy(temp_buf, input_node_name.get());
            inputNodeNames.push_back(temp_buf);
        }
        size_t OutputNodesNum = session->GetOutputCount();
        for (size_t i = 0; i < OutputNodesNum; i++) {
            Ort::AllocatedStringPtr output_node_name =
                session->GetOutputNameAllocated(i, allocator);
            char *temp_buf = new char[10];
            strcpy(temp_buf, output_node_name.get());
            outputNodeNames.push_back(temp_buf);
        }
        options = Ort::RunOptions{nullptr};
        WarmUpSession();
        return RET_OK;
    } catch (const std::exception &e) {
        const char *str1 = "[YOLO_V8]:";
        const char *str2 = e.what();
        std::string result = std::string(str1) + std::string(str2);
        char *merged = new char[result.length() + 1];
        std::strcpy(merged, result.c_str());
        std::cout << merged << std::endl;
        delete[] merged;
        return "[YOLO_V8]:Create session failed.";
    }
}

char *YOLO_V8::RunSession(cv::Mat &iImg, std::vector<DL_RESULT> &oResult) {
    char *Ret = RET_OK;
    cv::Mat processedImg;
    PreProcess(iImg, imgSize, processedImg);
    if (modelType < 4) {
        float *blob = new float[processedImg.total() * 3];
        BlobFromImage(processedImg, blob);
        std::vector<int64_t> inputNodeDims = {1, 3, imgSize.at(0),
                                              imgSize.at(1)};
        TensorProcess(iImg, blob, inputNodeDims, oResult);
    }
    return Ret;
}

template <typename N>
char *YOLO_V8::TensorProcess(cv::Mat &iImg, N &blob,
                             std::vector<int64_t> &inputNodeDims,
                             std::vector<DL_RESULT> &oResult) {
    Ort::Value inputTensor =
        Ort::Value::CreateTensor<typename std::remove_pointer<N>::type>(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob,
            3 * imgSize.at(0) * imgSize.at(1), inputNodeDims.data(),
            inputNodeDims.size());
    auto outputTensor =
        session->Run(options, inputNodeNames.data(), &inputTensor, 1,
                     outputNodeNames.data(), outputNodeNames.size());

    Ort::TypeInfo typeInfo = outputTensor.front().GetTypeInfo();
    auto tensor_info = typeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outputNodeDims = tensor_info.GetShape();
    auto output =
        outputTensor.front()
            .GetTensorMutableData<typename std::remove_pointer<N>::type>();
    delete[] blob;
    int signalResultNum = outputNodeDims[1]; // 84
    int strideNum = outputNodeDims[2];       // 8400
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    cv::Mat rawData;
    if (modelType == YOLO_DETECT_V8) {
        // FP32
        rawData = cv::Mat(signalResultNum, strideNum, CV_32F, output);
    } else {
        // FP16
        rawData = cv::Mat(signalResultNum, strideNum, CV_16F, output);
        rawData.convertTo(rawData, CV_32F);
    }
    // Note:
    // ultralytics add transpose operator to the output of yolov8 model.which
    // make yolov8/v5/v7 has same shape
    // https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
    rawData = rawData.t();

    float *data = (float *)rawData.data;

    for (int i = 0; i < strideNum; ++i) {
        float *classesScores = data + 4;
        cv::Mat scores(1, this->classes.size(), CV_32FC1, classesScores);
        cv::Point class_id;
        double maxClassScore;
        cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
        if (maxClassScore > rectConfidenceThreshold) {
            confidences.push_back(maxClassScore);
            class_ids.push_back(class_id.x);
            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];

            int left = int((x - 0.5 * w) * resizeScales);
            int top = int((y - 0.5 * h) * resizeScales);

            int width = int(w * resizeScales);
            int height = int(h * resizeScales);

            boxes.push_back(cv::Rect(left, top, width, height));
        }
        data += signalResultNum;
    }
    std::vector<int> nmsResult;
    cv::dnn::NMSBoxes(boxes, confidences, rectConfidenceThreshold, iouThreshold,
                      nmsResult);
    for (int i = 0; i < nmsResult.size(); ++i) {
        int idx = nmsResult[i];
        DL_RESULT result;
        result.classId = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        oResult.push_back(result);
    }
    return RET_OK;
}

char *YOLO_V8::WarmUpSession() {
    clock_t starttime_1 = clock();
    cv::Mat iImg = cv::Mat(cv::Size(imgSize.at(0), imgSize.at(1)), CV_8UC3);
    cv::Mat processedImg;
    PreProcess(iImg, imgSize, processedImg);
    if (modelType < 4) {
        float *blob = new float[iImg.total() * 3];
        BlobFromImage(processedImg, blob);
        std::vector<int64_t> YOLO_input_node_dims = {1, 3, imgSize.at(0),
                                                     imgSize.at(1)};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob,
            3 * imgSize.at(0) * imgSize.at(1), YOLO_input_node_dims.data(),
            YOLO_input_node_dims.size());
        auto output_tensors =
            session->Run(options, inputNodeNames.data(), &input_tensor, 1,
                         outputNodeNames.data(), outputNodeNames.size());
        delete[] blob;
        clock_t starttime_4 = clock();
        double post_process_time =
            (double)(starttime_4 - starttime_1) / CLOCKS_PER_SEC * 1000;
        if (cudaEnable) {
            std::cout << "[YOLO_V8(CUDA)]: " << "Cuda warm-up cost "
                      << post_process_time << " ms. " << std::endl;
        }
    }
    return RET_OK;
}
