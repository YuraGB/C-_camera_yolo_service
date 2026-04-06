#pragma once

#include <grpcpp/grpcpp.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <google/protobuf/empty.pb.h>

#include "frame.h"
#include "generated_models/detection.grpc.pb.h"
#include "generated_models/detection.pb.h"

class DetectionServiceImpl final : public detection::DetectionService::Service {
 public:
  DetectionServiceImpl();
  ~DetectionServiceImpl() override;

  DetectionServiceImpl(const DetectionServiceImpl&) = delete;
  DetectionServiceImpl& operator=(const DetectionServiceImpl&) = delete;

  grpc::Status StreamLiveFrames(
      grpc::ServerContext* context,
      const google::protobuf::Empty* request,
      grpc::ServerWriter<detection::Frame>* writer) override;

  grpc::Status StreamDetectionFrames(
      grpc::ServerContext* context,
      const google::protobuf::Empty* request,
      grpc::ServerWriter<detection::Frame>* writer) override;

  void publishLiveFrame(std::shared_ptr<detection::Frame> frame);
  void publishDetectionFrame(std::shared_ptr<detection::Frame> frame);
  void notifyAll();
  bool hasLiveSubscribers() const;
  bool hasDetectionSubscribers() const;

 private:
  struct StreamState {
    std::mutex mutex;
    std::condition_variable cv;
    std::shared_ptr<detection::Frame> latest_frame;
    uint64_t sequence = 0;
    std::atomic<int> subscriber_count{0};
  };

  grpc::Status streamFrames(
      const char* stream_name,
      StreamState& state,
      grpc::ServerContext* context,
      grpc::ServerWriter<detection::Frame>* writer,
      std::chrono::milliseconds min_frame_interval);

  void publishFrame(StreamState& state, std::shared_ptr<detection::Frame> frame);
  bool hasSubscribers(const StreamState& state) const;

  StreamState live_stream_;
  StreamState detection_stream_;
};

class GRPCServer {
 public:
  explicit GRPCServer(const std::string& server_address);
  ~GRPCServer();

  GRPCServer(const GRPCServer&) = delete;
  GRPCServer& operator=(const GRPCServer&) = delete;

  void start();
  void stop();
  void sendLiveFrame(const std::shared_ptr<Frame>& frame);
  void sendDetectionResult(const std::shared_ptr<Frame>& frame);

 private:
  std::shared_ptr<detection::Frame> buildProtoFrame(
      Frame& frame,
      bool reuse_encoded_jpeg,
      bool include_detections) const;

  std::string server_address_;
  std::unique_ptr<DetectionServiceImpl> service_;
  std::unique_ptr<grpc::Server> server_;
  std::thread server_thread_;
  std::atomic<bool> running_;
};
