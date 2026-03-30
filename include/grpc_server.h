#pragma once
#include <grpcpp/grpcpp.h>
#include <memory>
#include <string>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cstdint>
#include "frame.h"

// Підключаємо згенеровані файли protobuf/grpc
#include "generated_models/detection.pb.h"
#include "generated_models/detection.grpc.pb.h"
#include <google/protobuf/empty.pb.h> 

// ------------------------------
// Сервіс gRPC для стрімінгу кадрів + детекцій
// ------------------------------
class DetectionServiceImpl final : public detection::DetectionService::Service {
public:
    DetectionServiceImpl();
    ~DetectionServiceImpl() override;

    DetectionServiceImpl(const DetectionServiceImpl&) = delete;
    DetectionServiceImpl& operator=(const DetectionServiceImpl&) = delete;

    grpc::Status StreamDetections(
        grpc::ServerContext* context,
        const google::protobuf::Empty* request,
        grpc::ServerWriter<detection::Frame>* writer
    ) override;

    void enqueueFrame(std::shared_ptr<Frame> frame);
    size_t getQueueSize();
    void notifyAll();

private:
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::shared_ptr<Frame> latest_frame_;
    uint64_t latest_frame_sequence_ = 0;
};

// ------------------------------
// GRPC сервер
// ------------------------------
class GRPCServer {
public:
    explicit GRPCServer(const std::string& server_address);
    ~GRPCServer();

    GRPCServer(const GRPCServer&) = delete;
    GRPCServer& operator=(const GRPCServer&) = delete;

    void start();
    void stop();
    void sendDetectionResult(std::shared_ptr<Frame> frame);

private:
    std::string server_address_;
    std::unique_ptr<DetectionServiceImpl> service_;
    std::unique_ptr<grpc::Server> server_;
    std::thread server_thread_;
    std::atomic<bool> running_;
};
