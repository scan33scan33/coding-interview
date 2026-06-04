/*
 * Semantic Costmap Generation via BFS Inflation
 * -----------------------------------------------
 * Interview problem: given a 2D grid, lidar obstacle points, and semantic
 * objects (e.g. "person", "dog"), inflate each obstacle outward up to its
 * configured radius. Each cell's cost decays linearly from 100 (lethal) at
 * the source to 0 at the radius boundary.
 *
 * Two algorithms are implemented and benchmarked:
 *
 *   Algorithm 1 — Multi-Source FIFO BFS
 *     Simple queue-based wave expansion. Revisits cells if a higher cost
 *     arrives later, so it isn't strictly Dijkstra. Works well in practice
 *     because obstacles tend to be sparse.
 *
 *   Algorithm 2 — Priority Bucket Queue (Dijkstra-style)
 *     101 discrete cost buckets (0–100). Processes cells in descending cost
 *     order so each cell is settled on first visit (lazy deletion handles
 *     stale entries). Avoids redundant revisits in theory, but the overhead
 *     of maintaining buckets makes it slower on typical sparse maps.
 *
 * Key bugs fixed from the original implementation (semantic_cost_map.cc):
 *   1. Dangling reference after q.pop(): must copy BFSState by value.
 *   2. Missing bounds check on neighbor cells before array access.
 *   3. Cost comparison was inverted (should skip if existing cost >= new cost).
 *   4. Typo: output[new_j][new_j] instead of output[new_i][new_j].
 *
 * Build:
 *   g++ -O3 -std=c++17 semantic_costmap_bfs.cpp -o semantic_costmap_bfs
 *
 * Run:
 *   ./semantic_costmap_bfs
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <queue>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// ─────────────────────────────────────────────
// Constants & core types
// ─────────────────────────────────────────────

constexpr float kLethalCell = 100.0f;
constexpr float kWallRadius  = 2.0f;

struct SemanticObject {
    std::string name;
    int row;
    int col;
};

struct BFSState {
    int source_row, source_col;
    int curr_row,   curr_col;
    float radius;
};

// ─────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────

inline float ComputeCost(float distance, float radius) {
    return std::max(0.0f, kLethalCell - std::floor(kLethalCell * distance / radius));
}

inline float L2Dist(int r1, int c1, int r2, int c2) {
    float dr = static_cast<float>(r1 - r2);
    float dc = static_cast<float>(c1 - c2);
    return std::sqrt(dr * dr + dc * dc);
}

static constexpr int kDr[] = {-1, -1, -1,  0, 0,  1, 1, 1};
static constexpr int kDc[] = {-1,  0,  1, -1, 1, -1, 0, 1};

// ─────────────────────────────────────────────
// Algorithm 1: Multi-Source FIFO BFS
// ─────────────────────────────────────────────
//
// Seed all obstacle sources at cost=100, then fan out wave-by-wave.
// A cell can be re-enqueued if a higher cost arrives later, so this
// isn't loop-free in the worst case — but empirically fast on sparse maps.

std::vector<float> CostmapFIFO(
    int rows, int cols,
    const std::vector<std::pair<int, int>>& lidar_points,
    const std::vector<SemanticObject>& objects,
    const std::unordered_map<std::string, float>& inflation_config
) {
    std::vector<float> costmap(rows * cols, 0.0f);
    std::queue<BFSState> q;

    auto is_valid = [rows, cols](int r, int c) {
        return r >= 0 && c >= 0 && r < rows && c < cols;
    };

    for (const auto& [r, c] : lidar_points) {
        if (is_valid(r, c)) {
            costmap[r * cols + c] = kLethalCell;
            q.push({r, c, r, c, kWallRadius});
        }
    }

    for (const auto& obj : objects) {
        if (is_valid(obj.row, obj.col)) {
            costmap[obj.row * cols + obj.col] = kLethalCell;
            float radius = kWallRadius;
            if (auto it = inflation_config.find(obj.name); it != inflation_config.end())
                radius = it->second;
            q.push({obj.row, obj.col, obj.row, obj.col, radius});
        }
    }

    while (!q.empty()) {
        BFSState curr = q.front(); // copy — do NOT keep a reference across pop()
        q.pop();

        for (int i = 0; i < 8; ++i) {
            int nr = curr.curr_row + kDr[i];
            int nc = curr.curr_col + kDc[i];
            if (!is_valid(nr, nc)) continue;

            float dist     = L2Dist(nr, nc, curr.source_row, curr.source_col);
            if (dist > curr.radius) continue;

            float new_cost = ComputeCost(dist, curr.radius);
            int   idx      = nr * cols + nc;

            if (new_cost > costmap[idx]) {
                costmap[idx] = new_cost;
                q.push({curr.source_row, curr.source_col, nr, nc, curr.radius});
            }
        }
    }

    return costmap;
}

// ─────────────────────────────────────────────
// Algorithm 2: Priority Bucket Queue
// ─────────────────────────────────────────────
//
// 101 buckets for discrete costs 0–100. Sweep from cost=100 down to 1.
// Each cell is settled on first visit (lazy deletion skips stale entries),
// eliminating the redundant re-expansions of the FIFO approach.
// In practice the bucket overhead outweighs the savings on typical maps.

std::vector<float> CostmapPriorityBucket(
    int rows, int cols,
    const std::vector<std::pair<int, int>>& lidar_points,
    const std::vector<SemanticObject>& objects,
    const std::unordered_map<std::string, float>& inflation_config
) {
    std::vector<float> costmap(rows * cols, 0.0f);
    std::array<std::vector<BFSState>, 101> buckets;

    auto is_valid = [rows, cols](int r, int c) {
        return r >= 0 && c >= 0 && r < rows && c < cols;
    };

    for (const auto& [r, c] : lidar_points) {
        if (is_valid(r, c)) {
            costmap[r * cols + c] = kLethalCell;
            buckets[100].push_back({r, c, r, c, kWallRadius});
        }
    }

    for (const auto& obj : objects) {
        if (is_valid(obj.row, obj.col)) {
            costmap[obj.row * cols + obj.col] = kLethalCell;
            float radius = kWallRadius;
            if (auto it = inflation_config.find(obj.name); it != inflation_config.end())
                radius = it->second;
            buckets[100].push_back({obj.row, obj.col, obj.row, obj.col, radius});
        }
    }

    for (int cost_level = 100; cost_level > 0; --cost_level) {
        while (!buckets[cost_level].empty()) {
            BFSState curr = buckets[cost_level].back();
            buckets[cost_level].pop_back();

            int curr_idx = curr.curr_row * cols + curr.curr_col;
            // Lazy deletion: a higher-cost wave already settled this cell.
            if (costmap[curr_idx] > static_cast<float>(cost_level)) continue;

            for (int i = 0; i < 8; ++i) {
                int nr = curr.curr_row + kDr[i];
                int nc = curr.curr_col + kDc[i];
                if (!is_valid(nr, nc)) continue;

                float dist     = L2Dist(nr, nc, curr.source_row, curr.source_col);
                if (dist > curr.radius) continue;

                float new_cost_f = ComputeCost(dist, curr.radius);
                int   new_cost_i = static_cast<int>(new_cost_f);
                if (new_cost_i <= 0) continue;

                int idx = nr * cols + nc;
                if (new_cost_f > costmap[idx]) {
                    costmap[idx] = new_cost_f;
                    buckets[new_cost_i].push_back(
                        {curr.source_row, curr.source_col, nr, nc, curr.radius});
                }
            }
        }
    }

    return costmap;
}

// ─────────────────────────────────────────────
// Benchmarking harness
// ─────────────────────────────────────────────

struct MapDataSet {
    std::vector<std::pair<int, int>> lidar_points;
    std::vector<SemanticObject>      semantic_objects;
};

MapDataSet GenerateMockEnvironment(
    int rows, int cols, int num_lidar, int num_objects, bool cluster
) {
    MapDataSet data;
    std::mt19937 rng(42); // fixed seed for determinism
    std::uniform_int_distribution<int> dist_r(0, rows - 1);
    std::uniform_int_distribution<int> dist_c(0, cols - 1);

    if (cluster) {
        std::normal_distribution<double> gauss_r(rows / 2.0, rows * 0.05);
        std::normal_distribution<double> gauss_c(cols / 2.0, cols * 0.05);
        for (int i = 0; i < num_lidar; ++i)
            data.lidar_points.emplace_back(
                std::clamp((int)gauss_r(rng), 0, rows - 1),
                std::clamp((int)gauss_c(rng), 0, cols - 1));
        for (int i = 0; i < num_objects; ++i)
            data.semantic_objects.push_back({"person",
                std::clamp((int)gauss_r(rng), 0, rows - 1),
                std::clamp((int)gauss_c(rng), 0, cols - 1)});
    } else {
        for (int i = 0; i < num_lidar; ++i)
            data.lidar_points.emplace_back(dist_r(rng), dist_c(rng));
        for (int i = 0; i < num_objects; ++i)
            data.semantic_objects.push_back({"person", dist_r(rng), dist_c(rng)});
    }
    return data;
}

using CostmapFn = std::vector<float>(*)(
    int, int,
    const std::vector<std::pair<int,int>>&,
    const std::vector<SemanticObject>&,
    const std::unordered_map<std::string, float>&);

void RunBenchmark(
    const char* label,
    CostmapFn fn,
    int rows, int cols,
    const MapDataSet& data,
    const std::unordered_map<std::string, float>& config
) {
    auto t0 = std::chrono::high_resolution_clock::now();
    auto costmap = fn(rows, cols, data.lidar_points, data.semantic_objects, config);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms       = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
    double checksum = 0.0;
    for (float v : costmap) checksum += v;

    std::cout << "    [" << label << "] Time: " << ms << " ms | Checksum: " << checksum << "\n";
}

int main() {
    const int kRows = 400, kCols = 400;
    const std::unordered_map<std::string, float> config = {{"person", 25.0f}};

    auto sparse    = GenerateMockEnvironment(kRows, kCols,   50,  5, false);
    auto dense     = GenerateMockEnvironment(kRows, kCols, 1000, 40, false);
    auto clustered = GenerateMockEnvironment(kRows, kCols, 1000, 40, true);

    std::cout << "--- Scenario 1: Sparse Indoor Space (50 Lidar, 5 Objects) ---\n";
    RunBenchmark("Standard FIFO  ", CostmapFIFO,           kRows, kCols, sparse,    config);
    RunBenchmark("Priority Bucket", CostmapPriorityBucket, kRows, kCols, sparse,    config);

    std::cout << "\n--- Scenario 2: Dense Uniformly Cluttered Space (1000 Lidar, 40 Objects) ---\n";
    RunBenchmark("Standard FIFO  ", CostmapFIFO,           kRows, kCols, dense,     config);
    RunBenchmark("Priority Bucket", CostmapPriorityBucket, kRows, kCols, dense,     config);

    std::cout << "\n--- Scenario 3: Highly Overlapping Clustered Bottleneck (1000 Lidar, 40 Objects) ---\n";
    RunBenchmark("Standard FIFO  ", CostmapFIFO,           kRows, kCols, clustered, config);
    RunBenchmark("Priority Bucket", CostmapPriorityBucket, kRows, kCols, clustered, config);

    std::cout << "\n";
    return 0;
}
