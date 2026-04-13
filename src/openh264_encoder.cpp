#include "openh264_encoder.h"

#include <cstring>
#include <stdexcept>

#include <windows.h>

#include "codec_api.h"

namespace {
constexpr int kTargetFps = 30;
constexpr int kTargetBitrateBps = 2'500'000;
constexpr int kKeyframeIntervalFrames = 60;

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

OpenH264Encoder::OpenH264Encoder(const std::string& dll_path) : dll_path_(dll_path) {
  loadLibrary();
}

OpenH264Encoder::~OpenH264Encoder() {
  if (encoder_) {
    encoder_->Uninitialize();
    destroy_encoder_(encoder_);
    encoder_ = nullptr;
  }

  if (dll_handle_) {
    FreeLibrary(static_cast<HMODULE>(dll_handle_));
    dll_handle_ = nullptr;
  }
}

bool OpenH264Encoder::isReady() const {
  return create_encoder_ != nullptr && destroy_encoder_ != nullptr;
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

  cv::Mat i420 = convertToI420(bgr_frame);

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
  dll_handle_ = LoadLibraryA(dll_path_.c_str());
  if (!dll_handle_) {
    throw std::runtime_error("Failed to load OpenH264 DLL: " + dll_path_);
  }

  create_encoder_ =
      reinterpret_cast<CreateEncoderFn>(GetProcAddress(static_cast<HMODULE>(dll_handle_), "WelsCreateSVCEncoder"));
  destroy_encoder_ =
      reinterpret_cast<DestroyEncoderFn>(GetProcAddress(static_cast<HMODULE>(dll_handle_), "WelsDestroySVCEncoder"));

  if (!create_encoder_ || !destroy_encoder_) {
    throw std::runtime_error("OpenH264 DLL is missing required encoder exports");
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
  params.uiIntraPeriod = kKeyframeIntervalFrames;
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
}

void OpenH264Encoder::ensureInitialized(int width, int height) {
  if (!encoder_ || width != width_ || height != height_) {
    initializeEncoder(width, height);
  }
}

cv::Mat OpenH264Encoder::convertToI420(const cv::Mat& bgr_frame) {
  cv::Mat yuv_i420;
  cv::cvtColor(bgr_frame, yuv_i420, cv::COLOR_BGR2YUV_I420);
  return yuv_i420;
}
