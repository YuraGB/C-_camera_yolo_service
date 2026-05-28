#pragma once

#include "capture/video_source.h"

namespace capture {

class OpenCvVideoSource final : public VideoSource {
 public:
  bool open(const std::string& source) override;
  bool read(cv::Mat& frame) override;
  bool grab() override;
  void release() override;
  bool isFileSource() const override;
  double fps() const override;
  double positionMs() const override;

 private:
  bool is_file_source_ = false;
  cv::VideoCapture capture_;
};

}  // namespace capture
