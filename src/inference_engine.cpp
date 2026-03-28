#include "inference_engine.h"

#include <iostream>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <filesystem>

// ------------------------------
// Constructor
// ------------------------------
InferenceEngine::InferenceEngine(const std::string& model_path)
    : model_path_(model_path),
      env_(ORT_LOGGING_LEVEL_WARNING, "InferenceEngine"),
      session_options_(),
      running_(false)
{
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    try {
        std::filesystem::path path_fs(model_path);
        std::wstring wmodel_path = path_fs.wstring();

        session_ = std::make_unique<Ort::Session>(
            env_, wmodel_path.c_str(), session_options_
        );

        Ort::AllocatorWithDefaultOptions allocator;

        // --- INPUTS ---
        size_t num_inputs = session_->GetInputCount();
        input_names_str_.resize(num_inputs);
        input_names_.resize(num_inputs);

        for (size_t i = 0; i < num_inputs; ++i) {
            auto name = session_->GetInputNameAllocated(i, allocator);
            input_names_str_[i] =
                (!name || std::strlen(name.get()) == 0)
                    ? "images"
                    : std::string(name.get());

            input_names_[i] = input_names_str_[i].c_str();

            std::cout << "[ONNX] Input " << i << ": "
                      << input_names_[i] << std::endl;
        }

        // --- OUTPUTS ---
        size_t num_outputs = session_->GetOutputCount();
        output_names_str_.resize(num_outputs);
        output_names_.resize(num_outputs);

        for (size_t i = 0; i < num_outputs; ++i) {
            auto name = session_->GetOutputNameAllocated(i, allocator);
            output_names_str_[i] =
                (!name || std::strlen(name.get()) == 0)
                    ? "output0"
                    : std::string(name.get());

            output_names_[i] = output_names_str_[i].c_str();

            std::cout << "[ONNX] Output " << i << ": "
                      << output_names_[i] << std::endl;
        }

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime exception: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Init error: " << e.what() << std::endl;
    }
}

// ------------------------------
InferenceEngine::~InferenceEngine() {
    stop();
}

// ------------------------------
void InferenceEngine::start() {
    running_ = true;
    inference_thread_ = std::thread(&InferenceEngine::inferenceLoop, this);
}

// ------------------------------
void InferenceEngine::stop() {
    running_ = false;
    queue_cv_.notify_all();

    if (inference_thread_.joinable())
        inference_thread_.join();
}

// ------------------------------
void InferenceEngine::processFrame(std::shared_ptr<Frame> frame) {
    if (!frame) return;

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        input_queue_.push(frame);
    }

    queue_cv_.notify_one();
}

// ------------------------------
std::shared_ptr<Frame> InferenceEngine::getResult() {
    std::lock_guard<std::mutex> lock(queue_mutex_);

    if (output_queue_.empty())
        return nullptr;

    auto frame = output_queue_.front();
    output_queue_.pop();

    return frame;
}

// ------------------------------
void InferenceEngine::inferenceLoop() {
    while (running_) {
        std::shared_ptr<Frame> frame;

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this]() {
                return !input_queue_.empty() || !running_;
            });

            if (!running_) break;

            frame = input_queue_.front();
            input_queue_.pop();
        }

        processFrameImpl(frame);

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            output_queue_.push(frame);
        }
    }
}

// ------------------------------
void InferenceEngine::processFrameImpl(std::shared_ptr<Frame> frame) {
    if (!frame || frame->mat.empty()) return;
    frame->detections.clear();

    // --- Inference mat ---
    cv::Mat infer_mat;
    cv::resize(frame->mat, infer_mat, cv::Size(640, 640));
    infer_mat.convertTo(infer_mat, CV_32F, 1.0 / 255.0);
    cv::cvtColor(infer_mat, infer_mat, cv::COLOR_BGR2RGB);

    // --- JPEG для фронтенду ---
    cv::Mat send_mat;
    cv::resize(frame->mat, send_mat, cv::Size(640, 640)); // resize тільки для фронтенду
    cv::imencode(".jpg", send_mat, frame->jpeg);

    // --- Input tensor (CHW) ---
    Frame tmp_frame("", 0, 0, infer_mat); // створюємо тимчасовий Frame для CHW
    std::vector<float> input_tensor_values = tmp_frame.getDataCHW();
    std::vector<int64_t> input_shape = {1, infer_mat.channels(), infer_mat.rows, infer_mat.cols};

    try {
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault
        );

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_tensor_values.data(),
            input_tensor_values.size(),
            input_shape.data(),
            input_shape.size()
        );

        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_.data(),
            &input_tensor,
            input_names_.size(),
            output_names_.data(),
            output_names_.size()
        );

        if (!output_tensors.empty() && output_tensors[0].IsTensor()) {
            auto& tensor = output_tensors[0];
            float* output_data = tensor.GetTensorMutableData<float>();
            frame->inference_result.assign(output_data, output_data + tensor.GetTensorTypeAndShapeInfo().GetElementCount());

            float scale_x = static_cast<float>(frame->mat.cols) / infer_mat.cols; // коефіцієнт по ширині
            float scale_y = static_cast<float>(frame->mat.rows) / infer_mat.rows; // коефіцієнт по висоті

            parseYOLO(frame); // detections
        }

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime inference error: " << e.what() << std::endl;
    }
}

// ------------------------------
void InferenceEngine::parseYOLO(std::shared_ptr<Frame> frame) {
    const float CONF_THRESHOLD = 0.25f;
    frame->detections.clear();

    int num_classes = 80;
    int elements_per_det = 84; // bbox+classes
    int num_det = frame->inference_result.size() / elements_per_det; // 8400

    float* data = frame->inference_result.data();

    // --- Оголошуємо розміри кадру
    int img_w = frame->mat.cols;
    int img_h = frame->mat.rows;

    for (int i = 0; i < num_det; i++) {
        float* ptr = data + i * elements_per_det;

        float x = ptr[0];
        float y = ptr[1];
        float w = ptr[2];
        float h = ptr[3];

        // знайти клас з max confidence
        float max_conf = 0;
        int class_id = -1;
        for (int c = 0; c < num_classes; c++) {
            float conf = ptr[4 + c];
            if (conf > max_conf) {
                max_conf = conf;
                class_id = c;
            }
        }

        if (max_conf > CONF_THRESHOLD) {
            float scale_x = static_cast<float>(frame->mat.cols) / 640.0f; // бо infer_mat 640x640
            float scale_y = static_cast<float>(frame->mat.rows) / 640.0f;
            frame->detections.push_back({
                std::to_string(class_id),
                max_conf,
                BBox(
                    static_cast<int>(x * scale_x),
                    static_cast<int>(y * scale_y),
                    static_cast<int>(w * scale_x),
                    static_cast<int>(h * scale_y)
                )
            });
        }
    }

    std::cout << "[DEBUG] detections: " << frame->detections.size() << std::endl;
}