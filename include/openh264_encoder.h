#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

class ISVCEncoder;

class OpenH264Encoder {
 public:
  explicit OpenH264Encoder(const std::string& dll_path);
  ~OpenH264Encoder();

  OpenH264Encoder(const OpenH264Encoder&) = delete;
  OpenH264Encoder& operator=(const OpenH264Encoder&) = delete;

  bool isReady() const;
  void setTargetFrameRate(double fps);
  std::vector<uint8_t> encode(const cv::Mat& bgr_frame, int64_t timestamp_ms, bool force_idr);

 private:
  using CreateEncoderFn = int(__cdecl*)(ISVCEncoder**);
  using DestroyEncoderFn = void(__cdecl*)(ISVCEncoder*);

  void loadLibrary();
  void initializeEncoder(int width, int height);
  void ensureInitialized(int width, int height);
  static cv::Mat convertToI420(const cv::Mat& bgr_frame);

  std::string dll_path_;
  void* dll_handle_ = nullptr;
  CreateEncoderFn create_encoder_ = nullptr;
  DestroyEncoderFn destroy_encoder_ = nullptr;
  ISVCEncoder* encoder_ = nullptr;
  int width_ = 0;
  int height_ = 0;
  double fps_ = 30.0;
  int bitrate_bps_ = 2'500'000;
  bool reconfigure_pending_ = false;
};
