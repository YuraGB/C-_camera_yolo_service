#include "grpc_server.h"
#include <thread>
#include <chrono>
#include <iostream>

// ------------------------------
// Конструктор / Деструктор DetectionServiceImpl
// ------------------------------
DetectionServiceImpl::DetectionServiceImpl()
    : MAX_QUEUE(30) // обмеження черги
{
}

DetectionServiceImpl::~DetectionServiceImpl() {
    notifyAll();
}

// ------------------------------
// Реалізація StreamDetections
// ------------------------------
grpc::Status DetectionServiceImpl::StreamDetections(
    grpc::ServerContext* context,
    const google::protobuf::Empty* /*request*/,
    grpc::ServerWriter<detection::Frame>* writer)
{
    std::cout << "[gRPC] StreamDetections запущено" << std::endl;

    while (!context->IsCancelled()) {
        std::shared_ptr<Frame> frame_ptr;

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (frame_queue_.empty()) {
                queue_cv_.wait_for(lock, std::chrono::milliseconds(100), [this, context]{
                    return !frame_queue_.empty() || context->IsCancelled();
                });
            }

            if (frame_queue_.empty() || context->IsCancelled()) continue;

            frame_ptr = frame_queue_.front();
            frame_queue_.pop();
        }

        if (!frame_ptr) continue;

        detection::Frame proto_frame;
        proto_frame.set_frame_id(static_cast<int32_t>(frame_ptr->frame_id));
        proto_frame.set_timestamp(frame_ptr->timestamp);
        proto_frame.set_camera_id(frame_ptr->camera_id);

        if (!frame_ptr->jpeg.empty()) {
            proto_frame.set_image(frame_ptr->jpeg.data(), frame_ptr->jpeg.size());
        }

        for (const auto& det : frame_ptr->detections) {
            auto* proto_det = proto_frame.add_detections();
            proto_det->set_label(det.label);
            proto_det->set_confidence(det.confidence);

            auto* proto_bbox = proto_det->mutable_bbox();
            proto_bbox->set_x(det.bbox.x);
            proto_bbox->set_y(det.bbox.y);
            proto_bbox->set_width(det.bbox.width);
            proto_bbox->set_height(det.bbox.height);
        }

        // Регулюємо FPS стріму, щоб не перевантажувати клієнта
        static auto last_send = std::chrono::steady_clock::now();
        const int TARGET_FPS = 5;
        const auto frame_delay = std::chrono::milliseconds(1000 / TARGET_FPS);

        auto now = std::chrono::steady_clock::now();
        auto elapsed = now - last_send;

        if (elapsed < frame_delay) {
            std::this_thread::sleep_for(frame_delay - elapsed);
        }

        last_send = std::chrono::steady_clock::now();

        if (!writer->Write(proto_frame)) {
            std::cout << "[gRPC] Client disconnected while sending frame_id=" << frame_ptr->frame_id << std::endl;
            break;
        }
    }

    std::cout << "[gRPC] StreamDetections завершено" << std::endl;
    return grpc::Status::OK;
}

// ------------------------------
// Додавання кадру в чергу
// ------------------------------
void DetectionServiceImpl::enqueueFrame(std::shared_ptr<Frame> frame)
{
    if (!frame) return;

    std::unique_lock<std::mutex> lock(queue_mutex_);
    if (frame_queue_.size() >= MAX_QUEUE) {
        frame_queue_.pop(); // видаляємо найстаріший
    }

    frame_queue_.push(frame);
    queue_cv_.notify_one();
}

// ------------------------------
// Геттер для розміру черги
// ------------------------------
size_t DetectionServiceImpl::getQueueSize()
{
    std::unique_lock<std::mutex> lock(queue_mutex_);
    return frame_queue_.size();
}

// ------------------------------
// Пробудження всіх потоків
// ------------------------------
void DetectionServiceImpl::notifyAll()
{
    queue_cv_.notify_all();
}

// ------------------------------
// GRPCServer реалізація
// ------------------------------
GRPCServer::GRPCServer(const std::string& server_address)
    : server_address_(server_address), running_(false)
{
    service_ = std::make_unique<DetectionServiceImpl>();
}

GRPCServer::~GRPCServer()
{
    stop();
}

void GRPCServer::start()
{
    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address_, grpc::InsecureServerCredentials());
    builder.RegisterService(service_.get());

    server_ = builder.BuildAndStart();
    if (!server_) throw std::runtime_error("Failed to start gRPC server");
    running_ = true;

    std::cout << "[gRPC] Сервер запущено на " << server_address_ << std::endl;

    server_thread_ = std::thread([this]() {
        if (server_) server_->Wait();
    });
}

void GRPCServer::stop()
{
    if (running_) {
        if (server_) server_->Shutdown();
        if (service_) service_->notifyAll();

        if (server_thread_.joinable())
            server_thread_.join();

        running_ = false;
        std::cout << "[gRPC] Сервер зупинено, кадрів у черзі="
                  << (service_ ? service_->getQueueSize() : 0) << std::endl;
    }
}

void GRPCServer::sendDetectionResult(std::shared_ptr<Frame> frame)
{
    if (service_ && !frame->mat.empty()) {
        frame->parseDetectionsFromYOLO(); // конвертуємо inference_result у detections
        frame->encodeJPEG(90);            // конвертуємо кадр у JPEG
        service_->enqueueFrame(frame);    // додаємо в чергу
    }
}