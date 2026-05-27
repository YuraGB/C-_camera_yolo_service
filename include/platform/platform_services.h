#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace platform {

class PlatformServices {
 public:
  virtual ~PlatformServices() = default;

  virtual std::string name() const = 0;
  virtual std::filesystem::path executableDir() const = 0;
  virtual std::filesystem::path sourceRootHint() const = 0;
  virtual std::vector<int> enumerateCameraIndices(int max_cameras) const = 0;
  virtual std::string defaultSignalingUrl() const = 0;
  virtual std::string defaultOpenH264LibraryName() const = 0;
};

std::unique_ptr<PlatformServices> createPlatformServices();

}  // namespace platform
