#include "openh264_encoder.h"

#include <algorithm>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

#include "codec_api.h"

namespace {
constexpr int kTargetBitrateBps = 2'500'000;
constexpr double kMinTargetFps = 5.0;
constexpr double kMaxTargetFps = 60.0;

void* loadDynamicLibrary(const std::string& path) {
#ifdef _WIN32
  return LoadLibraryA(path.c_str());
#else
  return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}

void* loadSymbol(void* handle, const char* name) {
#ifdef _WIN32
  return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle), name));
#else
  return dlsym(handle, name);
#endif
}

void closeDynamicLibrary(void* handle) {
  if (!handle) {
    return;
  }

#ifdef _WIN32
  FreeLibrary(static_cast<HMODULE>(handle));
#else
  dlclose(handle);
#endif
}

std::string dynamicLibraryError() {
#ifdef _WIN32
  return {};
#else
  const char* error = dlerror();
  return error ? std::string(error) : std::string();
#endif
}

int readEnvInt(const char* name, int fallback) {
  if (const char* raw = std::getenv(name)) {
    try {
      return std::max(1, std::stoi(raw));
    } catch (...) {
    }
  }
  return fallback;
}

std::vector<uint8_t> collectBitstream(const SFrameBSInfo& info) {
  std::vector<uint8_t> output;
  output.reserve(static_cast<size_t>(info.iFrameSizeInBytes));

  for (int layer = 0; layer < info.iLayerNum; ++layer) {
    const auto& layer_info = info.sLayerInfo[layer];
    int layer_size = 0;
    for (int nal = 0; nal < layer_info.iNalCount; ++nal) {
      layer_size += layer_info.pNalLengthInByte[nal];
    }

    const uint8_t* cursor = layer_info.pBsBuf;
    output.insert(output.end(), cursor, cursor + layer_size);
  }

  return output;
}
}  // namespace

OpenH264Encoder::OpenH264Encoder(const std::string& dll_path)
    : dll_path_(dll_path),
      bitrate_bps_(readEnvInt("CAMERA_H264_BITRATE_BPS", kTargetBitrateBps)) {
  loadLibrary();
}

OpenH264Encoder::~OpenH264Encoder() {
  if (encoder_) {
    encoder_->Uninitialize();
    destroy_encoder_(encoder_);
    encoder_ = nullptr;
  }

  if (dll_handle_) {
    closeDynamicLibrary(dll_handle_);
    dll_handle_ = nullptr;
  }
}

bool OpenH264Encoder::isReady() const {
  return create_encoder_ != nullptr && destroy_encoder_ != nullptr;
}

void OpenH264Encoder::setTargetFrameRate(double fps) {
  if (!(fps > 0.0)) {
    return;
  }

  const double clamped_fps = std::clamp(fps, kMinTargetFps, kMaxTargetFps);
  if (std::abs(clamped_fps - fps_) < 1.0) {
    return;
  }

  fps_ = clamped_fps;
  reconfigure_pending_ = true;
}

std::vector<uint8_t> OpenH264Encoder::encode(const cv::Mat& bgr_frame, int64_t timestamp_ms, bool force_idr) {
  if (bgr_frame.empty()) {
    return {};
  }

  ensureInitialized(bgr_frame.cols, bgr_frame.rows);
  if (!encoder_) {
    return {};
  }

  if (force_idr) {
    encoder_->ForceIntraFrame(true);
  }

  const cv::Mat& i420 = convertToI420(bgr_frame);

  SSourcePicture picture{};
  picture.iColorFormat = videoFormatI420;
  picture.iPicWidth = width_;
  picture.iPicHeight = height_;
  picture.iStride[0] = width_;
  picture.iStride[1] = width_ / 2;
  picture.iStride[2] = width_ / 2;
  picture.uiTimeStamp = timestamp_ms;

  picture.pData[0] = i420.data;
  picture.pData[1] = picture.pData[0] + (width_ * height_);
  picture.pData[2] = picture.pData[1] + (width_ * height_ / 4);

  SFrameBSInfo info{};
  const int result = encoder_->EncodeFrame(&picture, &info);
  if (result != 0 || info.eFrameType == videoFrameTypeSkip) {
    return {};
  }

  return collectBitstream(info);
}

void OpenH264Encoder::loadLibrary() {
  dll_handle_ = loadDynamicLibrary(dll_path_);
  if (!dll_handle_) {
    const std::string error = dynamicLibraryError();
    throw std::runtime_error(
        "Failed to load OpenH264 library: " + dll_path_ + (error.empty() ? "" : " (" + error + ")"));
  }

  create_encoder_ =
      reinterpret_cast<CreateEncoderFn>(loadSymbol(dll_handle_, "WelsCreateSVCEncoder"));
  destroy_encoder_ =
      reinterpret_cast<DestroyEncoderFn>(loadSymbol(dll_handle_, "WelsDestroySVCEncoder"));

  if (!create_encoder_ || !destroy_encoder_) {
    closeDynamicLibrary(dll_handle_);
    dll_handle_ = nullptr;
    throw std::runtime_error("OpenH264 library is missing required encoder exports");
  }
}

void OpenH264Encoder::initializeEncoder(int width, int height) {
  if (encoder_) {
    encoder_->Uninitialize();
    destroy_encoder_(encoder_);
    encoder_ = nullptr;
  }

  if (create_encoder_(&encoder_) != 0 || !encoder_) {
    throw std::runtime_error("WelsCreateSVCEncoder failed");
  }

  const int keyframe_interval_frames = std::max(15, static_cast<int>(std::round(fps_ * 2.0)));
  SEncParamExt params{};
  if (encoder_->GetDefaultParams(&params) != 0) {
    throw std::runtime_error("OpenH264 GetDefaultParams failed");
  }

  params.iUsageType = CAMERA_VIDEO_REAL_TIME;
  params.fMaxFrameRate = static_cast<float>(fps_);
  params.iPicWidth = width;
  params.iPicHeight = height;
  params.iTargetBitrate = bitrate_bps_;
  params.iMaxBitrate = bitrate_bps_;
  params.iRCMode = RC_BITRATE_MODE;
  params.iTemporalLayerNum = 1;
  params.iSpatialLayerNum = 1;
  params.bEnableFrameSkip = true;
  params.uiIntraPeriod = static_cast<unsigned int>(keyframe_interval_frames);
  params.sSpatialLayers[0].iVideoWidth = width;
  params.sSpatialLayers[0].iVideoHeight = height;
  params.sSpatialLayers[0].fFrameRate = static_cast<float>(fps_);
  params.sSpatialLayers[0].iSpatialBitrate = bitrate_bps_;
  params.sSpatialLayers[0].iMaxSpatialBitrate = bitrate_bps_;

  if (encoder_->InitializeExt(&params) != 0) {
    throw std::runtime_error("OpenH264 InitializeExt failed");
  }

  int video_format = videoFormatI420;
  encoder_->SetOption(ENCODER_OPTION_DATAFORMAT, &video_format);

  width_ = width;
  height_ = height;
  reconfigure_pending_ = false;
}

void OpenH264Encoder::ensureInitialized(int width, int height) {
  if (!encoder_ || width != width_ || height != height_ || reconfigure_pending_) {
    initializeEncoder(width, height);
  }
}

const cv::Mat& OpenH264Encoder::convertToI420(const cv::Mat& bgr_frame) {
  cv::cvtColor(bgr_frame, i420_buffer_, cv::COLOR_BGR2YUV_I420);
  return i420_buffer_;
}
