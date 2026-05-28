#include "platform/windows/windows_platform_services.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <algorithm>
#include <iostream>

#include <opencv2/opencv.hpp>

namespace platform::windows {
namespace {

class WindowsPlatformServices final : public PlatformServices {
 public:
  std::string name() const override {
    return "windows";
  }

  std::filesystem::path executableDir() const override {
    std::wstring buffer(MAX_PATH, L'\0');
    const DWORD length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (length == 0) {
      return {};
    }

    buffer.resize(length);
    return std::filesystem::path(buffer).parent_path();
  }

  std::filesystem::path sourceRootHint() const override {
    return std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
  }

  std::vector<int> enumerateCameraIndices(int max_cameras) const override {
    std::vector<int> indices;
    if (max_cameras <= 0) {
      return indices;
    }

    for (int index = 0; index < max_cameras; ++index) {
      cv::VideoCapture capture;
      if (!capture.open(index, cv::CAP_DSHOW)) {
        capture.open(index);
      }

      if (capture.isOpened()) {
        indices.push_back(index);
      }
    }

    return indices;
  }

  std::string defaultSignalingUrl() const override {
    return "ws://127.0.0.1:3002/ws";
  }

  std::string defaultOpenH264LibraryName() const override {
    return "openh264-2.6.0-win64.dll";
  }
};

}  // namespace

std::unique_ptr<PlatformServices> createWindowsPlatformServices() {
  return std::make_unique<WindowsPlatformServices>();
}

}  // namespace platform::windows

namespace platform {

std::unique_ptr<PlatformServices> createPlatformServices() {
  return windows::createWindowsPlatformServices();
}

}  // namespace platform
