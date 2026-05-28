#pragma once

#include <string>

#include <opencv2/opencv.hpp>

namespace capture {

class VideoSource {
 public:
  virtual ~VideoSource() = default;

  virtual bool open(const std::string& source) = 0;
  virtual bool read(cv::Mat& frame) = 0;
  virtual bool grab() = 0;
  virtual void release() = 0;
  virtual bool isFileSource() const = 0;
  virtual double fps() const = 0;
  virtual double positionMs() const = 0;
};

bool isNumericSource(const std::string& source);

}  // namespace capture
