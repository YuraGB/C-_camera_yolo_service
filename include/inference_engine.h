#pragma once

#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <onnxruntime_cxx_api.h>
#include "frame.h"

class InferenceEngine {
public:
    // --------------------------
    // Конструктор/деструктор
    // --------------------------
    explicit InferenceEngine(const std::string& model_path);
    ~InferenceEngine();

    // Забороняємо копіювання та присвоєння
    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;

    // --------------------------
    // Запуск/зупинка потоку інференсу
    // --------------------------
    void start();
    void stop();

    // --------------------------
    // Обробка кадру
    // --------------------------
    void processFrame(std::shared_ptr<Frame> frame); // додає кадр у чергу
    std::shared_ptr<Frame> getResult();             // повертає оброблений кадр з детекціями

private:
    // --------------------------
    // Власні методи
    // --------------------------
    void inferenceLoop();                                // головний цикл потоку
    void processFrameImpl(std::shared_ptr<Frame> frame); // реальна обробка кадру ONNX
    void parseYOLO(std::shared_ptr<Frame> frame, const std::vector<int64_t>& output_shape);
    void parseDetectionsFromYOLO(); // конвертує inference_result у detections

    // --------------------------
    // Модель та ONNX Runtime
    // --------------------------
    std::string model_path_;
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;

    // Імена входів та виходів моделі
    std::vector<std::string> input_names_str_;
    std::vector<const char*> input_names_;
    std::vector<std::string> output_names_str_;
    std::vector<const char*> output_names_;

    // --------------------------
    // Потік обробки та стан
    // --------------------------
    std::thread inference_thread_;
    std::atomic<bool> running_{false};

    // --------------------------
    // Черги кадрів
    // --------------------------
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::queue<std::shared_ptr<Frame>> input_queue_;
    std::queue<std::shared_ptr<Frame>> output_queue_;
};
