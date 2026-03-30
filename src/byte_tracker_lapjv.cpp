#include "byte_tracker_lapjv.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
constexpr double kLargeCost = 1000000.0;

int ccrrtDense(const size_t n, double* cost[], int* free_rows, int* x, int* y, double* v) {
    std::vector<bool> unique(n, true);

    for (size_t i = 0; i < n; ++i) {
        x[i] = -1;
        v[i] = kLargeCost;
        y[i] = 0;
    }

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (cost[i][j] < v[j]) {
                v[j] = cost[i][j];
                y[j] = static_cast<int>(i);
            }
        }
    }

    for (int j = static_cast<int>(n) - 1; j >= 0; --j) {
        const int i = y[j];
        if (x[i] < 0) {
            x[i] = j;
        } else {
            unique[i] = false;
            y[j] = -1;
        }
    }

    int free_count = 0;
    for (size_t i = 0; i < n; ++i) {
        if (x[i] < 0) {
            free_rows[free_count++] = static_cast<int>(i);
        } else if (unique[i]) {
            const int assigned_j = x[i];
            double min_value = kLargeCost;
            for (size_t j = 0; j < n; ++j) {
                if (static_cast<int>(j) == assigned_j) {
                    continue;
                }
                min_value = std::min(min_value, cost[i][j] - v[j]);
            }
            v[assigned_j] -= min_value;
        }
    }

    return free_count;
}

int carrDense(const size_t n, double* cost[], const size_t free_count, int* free_rows, int* x, int* y, double* v) {
    size_t current = 0;
    int new_free_rows = 0;
    size_t reduction_count = 0;

    while (current < free_count) {
        ++reduction_count;
        const int free_i = free_rows[current++];

        int j1 = 0;
        double v1 = cost[free_i][0] - v[0];
        int j2 = -1;
        double v2 = kLargeCost;

        for (size_t j = 1; j < n; ++j) {
            const double candidate = cost[free_i][j] - v[j];
            if (candidate < v2) {
                if (candidate >= v1) {
                    v2 = candidate;
                    j2 = static_cast<int>(j);
                } else {
                    v2 = v1;
                    v1 = candidate;
                    j2 = j1;
                    j1 = static_cast<int>(j);
                }
            }
        }

        int i0 = y[j1];
        const double updated_v1 = v[j1] - (v2 - v1);
        const bool lowers = updated_v1 < v[j1];

        if (reduction_count < current * n) {
            if (lowers) {
                v[j1] = updated_v1;
            } else if (i0 >= 0 && j2 >= 0) {
                j1 = j2;
                i0 = y[j2];
            }

            if (i0 >= 0) {
                if (lowers) {
                    free_rows[--current] = i0;
                } else {
                    free_rows[new_free_rows++] = i0;
                }
            }
        } else if (i0 >= 0) {
            free_rows[new_free_rows++] = i0;
        }

        x[free_i] = j1;
        y[j1] = free_i;
    }

    return new_free_rows;
}

size_t findDense(const size_t n, size_t lo, double* d, int* cols) {
    size_t hi = lo + 1;
    double mind = d[cols[lo]];
    for (size_t k = hi; k < n; ++k) {
        const int j = cols[k];
        if (d[j] <= mind) {
            if (d[j] < mind) {
                hi = lo;
                mind = d[j];
            }
            cols[k] = cols[hi];
            cols[hi++] = j;
        }
    }
    return hi;
}

int scanDense(const size_t n,
              double* cost[],
              size_t* lo,
              size_t* hi,
              double* d,
              int* cols,
              int* pred,
              int* y,
              double* v) {
    while (*lo != *hi) {
        int j = cols[(*lo)++];
        const int i = y[j];
        const double mind = d[j];
        const double h = cost[i][j] - v[j] - mind;

        for (size_t k = *hi; k < n; ++k) {
            j = cols[k];
            const double reduced_cost = cost[i][j] - v[j] - h;
            if (reduced_cost < d[j]) {
                d[j] = reduced_cost;
                pred[j] = i;
                if (reduced_cost == mind) {
                    if (y[j] < 0) {
                        return j;
                    }
                    cols[k] = cols[*hi];
                    cols[(*hi)++] = j;
                }
            }
        }
    }
    return -1;
}

int findPathDense(const size_t n, double* cost[], const int start_i, int* y, double* v, int* pred) {
    size_t lo = 0;
    size_t hi = 0;
    size_t ready = 0;
    int final_j = -1;
    std::vector<int> cols(n);
    std::vector<double> d(n);

    for (size_t i = 0; i < n; ++i) {
        cols[i] = static_cast<int>(i);
        pred[i] = start_i;
        d[i] = cost[start_i][i] - v[i];
    }

    while (final_j == -1) {
        if (lo == hi) {
            ready = lo;
            hi = findDense(n, lo, d.data(), cols.data());
            for (size_t k = lo; k < hi; ++k) {
                if (y[cols[k]] < 0) {
                    final_j = cols[k];
                }
            }
        }

        if (final_j == -1) {
            final_j = scanDense(n, cost, &lo, &hi, d.data(), cols.data(), pred, y, v);
        }
    }

    const double mind = d[cols[lo]];
    for (size_t k = 0; k < ready; ++k) {
        const int j = cols[k];
        v[j] += d[j] - mind;
    }

    return final_j;
}

int caDense(const size_t n, double* cost[], const size_t free_count, int* free_rows, int* x, int* y, double* v) {
    std::vector<int> pred(n, 0);
    for (int* free_i = free_rows; free_i < free_rows + free_count; ++free_i) {
        int i = -1;
        int j = findPathDense(n, cost, *free_i, y, v, pred.data());
        size_t k = 0;
        while (i != *free_i) {
            i = pred[j];
            y[j] = i;
            std::swap(j, x[i]);
            if (++k >= n) {
                throw std::runtime_error("lapjv augment path exceeded matrix size");
            }
        }
    }
    return 0;
}

int lapjvInternal(const size_t n, double* cost[], int* x, int* y) {
    std::vector<int> free_rows(n, 0);
    std::vector<double> v(n, 0.0);

    int result = ccrrtDense(n, cost, free_rows.data(), x, y, v.data());
    int rounds = 0;
    while (result > 0 && rounds < 2) {
        result = carrDense(n, cost, static_cast<size_t>(result), free_rows.data(), x, y, v.data());
        ++rounds;
    }
    if (result > 0) {
        result = caDense(n, cost, static_cast<size_t>(result), free_rows.data(), x, y, v.data());
    }
    return result;
}
}

// Adapted from the MIT-licensed ByteTrack-cpp LAPJV implementation.
double byte_tracker_internal::exec_lapjv(const std::vector<std::vector<float>>& cost,
                                         std::vector<int>& rowsol,
                                         std::vector<int>& colsol,
                                         bool extend_cost,
                                         float cost_limit) {
    if (cost.empty() || cost[0].empty()) {
        rowsol.clear();
        colsol.clear();
        return 0.0;
    }

    std::vector<std::vector<float>> expanded = cost;
    const int rows = static_cast<int>(cost.size());
    const int cols = static_cast<int>(cost[0].size());
    rowsol.resize(rows);
    colsol.resize(cols);

    int n = 0;
    if (rows == cols) {
        n = rows;
    } else if (!extend_cost) {
        throw std::runtime_error("lapjv requires a square cost matrix unless extend_cost is enabled");
    }

    if (extend_cost || cost_limit < std::numeric_limits<float>::max()) {
        n = rows + cols;
        std::vector<std::vector<float>> extended(
            static_cast<size_t>(n),
            std::vector<float>(static_cast<size_t>(n), 0.0f));

        const float fill_value =
            (cost_limit < std::numeric_limits<float>::max()) ? (cost_limit / 2.0f) : 1.0f;
        for (auto& row : extended) {
            std::fill(row.begin(), row.end(), fill_value);
        }

        for (int i = rows; i < n; ++i) {
            for (int j = cols; j < n; ++j) {
                extended[i][j] = 0.0f;
            }
        }

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                extended[i][j] = cost[i][j];
            }
        }

        expanded = std::move(extended);
    }

    std::vector<std::vector<double>> cost_storage(
        expanded.size(),
        std::vector<double>(expanded.empty() ? 0 : expanded[0].size(), 0.0));
    for (size_t i = 0; i < expanded.size(); ++i) {
        for (size_t j = 0; j < expanded[i].size(); ++j) {
            cost_storage[i][j] = expanded[i][j];
        }
    }

    std::vector<double*> pointers(cost_storage.size(), nullptr);
    for (size_t i = 0; i < cost_storage.size(); ++i) {
        pointers[i] = cost_storage[i].data();
    }

    std::vector<int> row_solution(static_cast<size_t>(n), -1);
    std::vector<int> col_solution(static_cast<size_t>(n), -1);
    const int result = lapjvInternal(static_cast<size_t>(n), pointers.data(), row_solution.data(), col_solution.data());
    if (result != 0) {
        throw std::runtime_error("lapjv failed");
    }

    if (n != rows) {
        for (int i = 0; i < n; ++i) {
            if (row_solution[i] >= cols) {
                row_solution[i] = -1;
            }
            if (col_solution[i] >= rows) {
                col_solution[i] = -1;
            }
        }
    }

    for (int i = 0; i < rows; ++i) {
        rowsol[i] = row_solution[i];
    }
    for (int i = 0; i < cols; ++i) {
        colsol[i] = col_solution[i];
    }

    double optimum = 0.0;
    for (int i = 0; i < rows; ++i) {
        if (rowsol[i] >= 0) {
            optimum += cost_storage[static_cast<size_t>(i)][static_cast<size_t>(rowsol[i])];
        }
    }
    return optimum;
}
