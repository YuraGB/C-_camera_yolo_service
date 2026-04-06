#include "grpc_server.h"

#include <chrono>
#include <iostream>
#include <thread>
#include <utility>
#include <vector>

namespace {
constexpr auto kDetectionFrameInterval = std::chrono::milliseconds(1000 / 60);
constexpr int kLiveJpegQuality = 75;
constexpr int kDetectionJpegQuality = 90;
}

DetectionServiceImpl::DetectionServiceImpl() = default;

DetectionServiceImpl::~DetectionServiceImpl() {
  notifyAll();
}

grpc::Status DetectionServiceImpl::StreamLiveFrames(
    grpc::ServerContext* context,
    const google::protobuf::Empty* /*request*/,
    grpc::ServerWriter<detection::Frame>* writer) {
  return streamFrames("StreamLiveFrames", live_stream_, context, writer, std::chrono::milliseconds::zero());
}

grpc::Status DetectionServiceImpl::StreamDetectionFrames(
    grpc::ServerContext* context,
    const google::protobuf::Empty* /*request*/,
    grpc::ServerWriter<detection::Frame>* writer) {
  return streamFrames("StreamDetectionFrames", detection_stream_, context, writer, kDetectionFrameInterval);
}

void DetectionServiceImpl::publishLiveFrame(std::shared_ptr<detection::Frame> frame) {
  publishFrame(live_stream_, std::move(frame));
}

void DetectionServiceImpl::publishDetectionFrame(std::shared_ptr<detection::Frame> frame) {
  publishFrame(detection_stream_, std::move(frame));
}

void DetectionServiceImpl::notifyAll() {
  live_stream_.cv.notify_all();
  detection_stream_.cv.notify_all();
}

grpc::Status DetectionServiceImpl::streamFrames(
    const char* stream_name,
    StreamState& state,
    grpc::ServerContext* context,
    grpc::ServerWriter<detection::Frame>* writer,
    std::chrono::milliseconds min_frame_interval) {
  std::cout << "[gRPC] " << stream_name << " started" << std::endl;

  uint64_t last_seen_sequence = 0;
  auto last_send = std::chrono::steady_clock::now();
  while (!context->IsCancelled()) {
    std::shared_ptr<detection::Frame> frame;

    {
      std::unique_lock<std::mutex> lock(state.mutex);
      state.cv.wait_for(lock, std::chrono::milliseconds(100), [&state, context, last_seen_sequence] {
        return context->IsCancelled() || state.sequence > last_seen_sequence;
      });

      if (context->IsCancelled() || state.sequence <= last_seen_sequence || !state.latest_frame) {
        continue;
      }

      frame = state.latest_frame;
      last_seen_sequence = state.sequence;
    }

    if (min_frame_interval.count() > 0) {
      const auto now = std::chrono::steady_clock::now();
      const auto elapsed = now - last_send;
      if (elapsed < min_frame_interval) {
        std::this_thread::sleep_for(min_frame_interval - elapsed);
      }
      last_send = std::chrono::steady_clock::now();
    }

    if (!writer->Write(*frame)) {
      std::cout << "[gRPC] Client disconnected from " << stream_name
                << " while sending frame_id=" << frame->frame_id() << std::endl;
      break;
    }
  }

  std::cout << "[gRPC] " << stream_name << " finished" << std::endl;
  return grpc::Status::OK;
}

void DetectionServiceImpl::publishFrame(StreamState& state, std::shared_ptr<detection::Frame> frame) {
  if (!frame) {
    return;
  }

  std::lock_guard<std::mutex> lock(state.mutex);
  state.latest_frame = std::move(frame);
  ++state.sequence;
  state.cv.notify_all();
}

GRPCServer::GRPCServer(const std::string& server_address)
    : server_address_(server_address), running_(false) {
  service_ = std::make_unique<DetectionServiceImpl>();
}

GRPCServer::~GRPCServer() {
  stop();
}

void GRPCServer::start() {
  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address_, grpc::InsecureServerCredentials());
  builder.RegisterService(service_.get());

  server_ = builder.BuildAndStart();
  if (!server_) {
    throw std::runtime_error("Failed to start gRPC server");
  }
  running_ = true;

  std::cout << "[gRPC] Server started on " << server_address_ << std::endl;

  server_thread_ = std::thread([this]() {
    if (server_) {
      server_->Wait();
    }
  });
}

void GRPCServer::stop() {
  if (running_) {
    if (server_) {
      server_->Shutdown();
    }
    if (service_) {
      service_->notifyAll();
    }

    if (server_thread_.joinable()) {
      server_thread_.join();
    }

    running_ = false;
    std::cout << "[gRPC] Server stopped" << std::endl;
  }
}

void GRPCServer::sendLiveFrame(const std::shared_ptr<Frame>& frame) {
  if (!service_ || !frame || frame->mat.empty()) {
    return;
  }

  auto proto_frame = buildProtoFrame(*frame, kLiveJpegQuality, false);
  if (!proto_frame) {
    return;
  }

  service_->publishLiveFrame(std::move(proto_frame));
}

void GRPCServer::sendDetectionResult(const std::shared_ptr<Frame>& frame) {
  if (!service_ || !frame || frame->mat.empty()) {
    return;
  }

  auto proto_frame = buildProtoFrame(*frame, kDetectionJpegQuality, true);
  if (!proto_frame) {
    return;
  }

  service_->publishDetectionFrame(std::move(proto_frame));
}

std::shared_ptr<detection::Frame> GRPCServer::buildProtoFrame(
    const Frame& frame,
    int jpeg_quality,
    bool include_detections) const {
  if (frame.camera_id.empty()) {
    std::cerr << "[gRPC] Dropping frame with missing camera_id. frame_id=" << frame.frame_id << std::endl;
    return nullptr;
  }

  if (frame.frame_id < 0) {
    std::cerr << "[gRPC] Dropping frame with invalid frame_id=" << frame.frame_id
              << " camera_id=" << frame.camera_id << std::endl;
    return nullptr;
  }

  auto proto_frame = std::make_shared<detection::Frame>();
  proto_frame->set_frame_id(frame.frame_id);
  proto_frame->set_timestamp(frame.timestamp);
  proto_frame->set_camera_id(frame.camera_id);

  if (!frame.mat.empty()) {
    std::vector<unsigned char> jpeg;
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, jpeg_quality};
    cv::imencode(".jpg", frame.mat, jpeg, params);
    proto_frame->set_image(jpeg.data(), static_cast<int>(jpeg.size()));
  }

  if (include_detections) {
    for (const auto& det : frame.detections) {
      auto* proto_det = proto_frame->add_detections();
      proto_det->set_label(det.label);
      proto_det->set_confidence(det.confidence);

      auto* proto_bbox = proto_det->mutable_bbox();
      proto_bbox->set_x(det.bbox.x);
      proto_bbox->set_y(det.bbox.y);
      proto_bbox->set_width(det.bbox.width);
      proto_bbox->set_height(det.bbox.height);
    }
  }

  return proto_frame;
}
