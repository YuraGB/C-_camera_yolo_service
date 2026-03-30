#pragma once

#include <vector>

namespace byte_tracker_internal {
double exec_lapjv(const std::vector<std::vector<float>>& cost,
                  std::vector<int>& rowsol,
                  std::vector<int>& colsol,
                  bool extend_cost,
                  float cost_limit);
}
