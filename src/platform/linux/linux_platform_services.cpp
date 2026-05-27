#include "platform/linux/linux_platform_services.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <set>
#include <string>
#include <vector>

#include <unistd.h>
#include <opencv2/opencv.hpp>

namespace platform::linux {
namespace {

bool parseVideoDeviceIndex(const std::filesystem::path& path, int& index) {
  const std::string name = path.filename().string();
  constexpr char prefix[] = "video";
  if (name.rfind(prefix, 0) != 0 || name.size() == std::strlen(prefix)) {
    return false;
  }

  const std::string suffix = name.substr(std::strlen(prefix));
  if (!std::all_of(suffix.begin(), suffix.end(), [](unsigned char c) { return std::isdigit(c) != 0; })) {
    return false;
  }

  try {
    index = std::stoi(suffix);
    return true;
  } catch (...) {
    return false;
  }
}

class LinuxPlatformServices final : public PlatformServices {
 public:
  std::string name() const override {
    return "linux";
  }

  std::filesystem::path executableDir() const override {
    std::vector<char> buffer(4096, '\0');
    const ssize_t length = readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
    if (length <= 0) {
      return {};
    }

    return std::filesystem::path(std::string(buffer.data(), static_cast<size_t>(length))).parent_path();
  }

  std::filesystem::path sourceRootHint() const override {
    return std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
  }

  std::vector<int> enumerateCameraIndices(int max_cameras) const override {
    std::set<int> discovered;
    std::error_code ec;
    const std::filesystem::path dev_dir("/dev");
    if (std::filesystem::exists(dev_dir, ec) && !ec) {
      for (const auto& entry : std::filesystem::directory_iterator(dev_dir, ec)) {
        if (ec) {
          break;
        }

        int index = -1;
        if (parseVideoDeviceIndex(entry.path(), index)) {
          discovered.insert(index);
        }
      }
    }

    for (int index = 0; index < max_cameras; ++index) {
      discovered.insert(index);
    }

    std::vector<int> indices;
    for (int index : discovered) {
      if (max_cameras >= 0 && static_cast<int>(indices.size()) >= max_cameras) {
        break;
      }

      cv::VideoCapture capture;
      if (!capture.open(index, cv::CAP_V4L2)) {
        capture.open(index);
      }

      if (capture.isOpened()) {
        indices.push_back(index);
      }
    }

    return indices;
  }

  std::string defaultSignalingUrl() const override {
    return "ws://127.0.0.1:3001/ws";
  }

  std::string defaultOpenH264LibraryName() const override {
    return "libopenh264.so.2";
  }
};

}  // namespace

std::unique_ptr<PlatformServices> createLinuxPlatformServices() {
  return std::make_unique<LinuxPlatformServices>();
}

}  // namespace platform::linux

namespace platform {

std::unique_ptr<PlatformServices> createPlatformServices() {
  return linux::createLinuxPlatformServices();
}

}  // namespace platform
