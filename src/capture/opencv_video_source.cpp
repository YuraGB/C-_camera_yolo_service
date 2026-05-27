#include "capture/opencv_video_source.h"

#include <cctype>

namespace capture {

bool isNumericSource(const std::string& source) {
  if (source.empty()) {
    return false;
  }

  const size_t start = (source[0] == '-' || source[0] == '+') ? 1 : 0;
  if (start >= source.size()) {
    return false;
  }

  for (size_t index = start; index < source.size(); ++index) {
    if (!std::isdigit(static_cast<unsigned char>(source[index]))) {
      return false;
    }
  }

  return true;
}

bool OpenCvVideoSource::open(const std::string& source) {
  source_ = source;
  is_file_source_ = !isNumericSource(source);

  if (isNumericSource(source)) {
    try {
      is_file_source_ = false;
      return capture_.open(std::stoi(source));
    } catch (...) {
      is_file_source_ = true;
    }
  }

  return capture_.open(source);
}

bool OpenCvVideoSource::read(cv::Mat& frame) {
  return capture_.read(frame);
}

bool OpenCvVideoSource::grab() {
  return capture_.grab();
}

void OpenCvVideoSource::release() {
  capture_.release();
}

bool OpenCvVideoSource::isOpened() const {
  return capture_.isOpened();
}

bool OpenCvVideoSource::isFileSource() const {
  return is_file_source_;
}

double OpenCvVideoSource::fps() const {
  return capture_.get(cv::CAP_PROP_FPS);
}

double OpenCvVideoSource::positionMs() const {
  return capture_.get(cv::CAP_PROP_POS_MSEC);
}

std::string OpenCvVideoSource::source() const {
  return source_;
}

}  // namespace capture
