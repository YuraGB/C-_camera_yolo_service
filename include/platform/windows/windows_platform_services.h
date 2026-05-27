#pragma once

#include "platform/platform_services.h"

namespace platform::windows {

std::unique_ptr<PlatformServices> createWindowsPlatformServices();

}  // namespace platform::windows
