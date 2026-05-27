#pragma once

#include "platform/platform_services.h"

namespace platform::linux {

std::unique_ptr<PlatformServices> createLinuxPlatformServices();

}  // namespace platform::linux
