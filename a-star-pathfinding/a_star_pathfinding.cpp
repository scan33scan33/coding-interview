#include <algorithm>
#include <cmath>
#include <iostream>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

// A cell whose cost reaches LETHAL_COST is treated as an impassable obstacle.
// Costs in [0, LETHAL_COST) are traversal penalties (e.g. inflation zones).
constexpr int kLethalCost = 100;

// Octile distance: the cheapest possible movement cost on an 8-connected grid,
// ignoring per-cell penalties (which only add to the true cost). This keeps the
// heuristic admissible *and* consistent, so A* stays optimal.
float Heuristic(int i, int j, int goal_i, int goal_j) {
    const int di = std::abs(i - goal_i);
    const int dj = std::abs(j - goal_j);
    const float kDiagonal = std::sqrt(2.0f);
    return (di + dj) + (kDiagonal - 2.0f) * std::min(di, dj);
}

struct State {
    int i, j;
    int from_i, from_j;
    float g;  // accumulated path cost from start to this cell
    float f;  // g + heuristic; the priority-queue ordering key
};

/**
 * Executes an A* Search on a pre-computed semantic costmap.
 *
 * @param rows The number of rows in the grid.
 * @param cols The number of columns in the grid.
 * @param master_costmap A 1D flat vector representing the 2D costmap (0-100 costs).
 *                       Cells at kLethalCost are impassable.
 * @param start The (row, col) starting coordinate.
 * @param goal The (row, col) destination coordinate.
 * @return A vector of (row, col) coordinates representing the optimal path from
 *         start to goal, inclusive of both endpoints. Empty if no path exists.
 */
std::vector<std::pair<int, int>> AStarSearch(
    int rows,
    int cols,
    const std::vector<int>& master_costmap,
    std::pair<int, int> start,
    std::pair<int, int> goal
) {
    std::vector<std::pair<int, int>> path;

    auto in_bounds = [&](int i, int j) {
        return i >= 0 && j >= 0 && i < rows && j < cols;
    };
    auto is_lethal = [&](int i, int j) {
        return master_costmap[i * cols + j] >= kLethalCost;
    };

    // Reject degenerate inputs up front.
    if (!in_bounds(start.first, start.second) || !in_bounds(goal.first, goal.second) ||
        is_lethal(start.first, start.second) || is_lethal(goal.first, goal.second)) {
        return path;
    }

    // 8-connected neighbourhood. Diagonal steps cost sqrt(2), orthogonal cost 1.
    const int di[] = {-1, -1, -1, 0, 0, 1, 1, 1};
    const int dj[] = {-1, 0, 1, -1, 1, -1, 0, 1};
    const float kDiagonal = std::sqrt(2.0f);
    const float step_cost[] = {kDiagonal, 1.0f, kDiagonal, 1.0f,
                               1.0f, kDiagonal, 1.0f, kDiagonal};

    auto cmp = [](const State& a, const State& b) { return a.f > b.f; };
    std::priority_queue<State, std::vector<State>, decltype(cmp)> pq(cmp);

    // Maps a flattened cell index to the settled State we reached it through.
    std::unordered_map<int, State> visited;

    State start_state{start.first, start.second, -1, -1, 0.0f,
                      Heuristic(start.first, start.second, goal.first, goal.second)};
    pq.push(start_state);

    while (!pq.empty()) {
        const State current = pq.top();
        pq.pop();

        const int flat = current.i * cols + current.j;
        // The first time we pop a cell we have its optimal cost (consistent
        // heuristic); stale duplicates left in the queue are skipped here.
        if (visited.count(flat)) {
            continue;
        }
        visited[flat] = current;

        if (current.i == goal.first && current.j == goal.second) {
            for (State s = current; ; s = visited[s.from_i * cols + s.from_j]) {
                path.emplace_back(s.i, s.j);
                if (s.from_i == -1) break;
            }
            std::reverse(path.begin(), path.end());
            return path;
        }

        for (int k = 0; k < 8; ++k) {
            const int ni = current.i + di[k];
            const int nj = current.j + dj[k];
            if (!in_bounds(ni, nj) || is_lethal(ni, nj)) {
                continue;
            }
            // Forbid cutting the corner of an obstacle on diagonal moves: both
            // orthogonally-adjacent cells must be clear for the robot to fit.
            const bool diagonal = (di[k] != 0 && dj[k] != 0);
            if (diagonal && (is_lethal(current.i, nj) || is_lethal(ni, current.j))) {
                continue;
            }
            if (visited.count(ni * cols + nj)) {
                continue;
            }

            const float g = current.g + step_cost[k] + master_costmap[ni * cols + nj];
            State next{ni, nj, current.i, current.j, g,
                       g + Heuristic(ni, nj, goal.first, goal.second)};
            pq.push(next);
        }
    }

    return path;  // empty: goal unreachable
}

void RunTestCase(
    const std::string& test_name,
    int rows, int cols,
    const std::vector<int>& grid,
    std::pair<int, int> start,
    std::pair<int, int> goal
) {
    std::cout << "--- " << test_name << " ---\n";
    auto path = AStarSearch(rows, cols, grid, start, goal);

    if (path.empty()) {
        std::cout << "Result: NO PATH FOUND\n\n";
        return;
    }

    std::cout << "Path (Start -> Goal):\n";
    for (const auto& node : path) {
        std::cout << "(" << node.first << ", " << node.second << ")\n";
    }
    std::cout << "\n";
}

int main() {
    // -----------------------------------------------------------------
    // TEST 1: Open room — diagonal efficiency.
    // With no obstacles, A* should take the straight diagonal (four
    // sqrt(2) steps) rather than orthogonal zig-zags.
    // -----------------------------------------------------------------
    RunTestCase(
        "Test 1: Open Space Diagonal",
        4, 4,
        {0, 0, 0, 0,
         0, 0, 0, 0,
         0, 0, 0, 0,
         0, 0, 0, 0},
        {0, 0},
        {3, 3}
    );

    // -----------------------------------------------------------------
    // TEST 2: U-shaped trap — escaping a local minimum.
    // The start (2,1) is walled on top, right, and bottom; the only
    // opening faces left, away from the goal. The greedy pull is to the
    // right, so the search must back out left and route around.
    // -----------------------------------------------------------------
    RunTestCase(
        "Test 2: U-Shaped Trap Escape",
        5, 5,
        {0,   0,   0,   0,  0,
         0, 100, 100, 100, 0,
         0,   0, 100,   0, 0,   // start at (2,1), opening to the left
         0, 100, 100, 100, 0,
         0,   0,   0,   0,  0},
        {2, 1},
        {2, 4}
    );

    // -----------------------------------------------------------------
    // TEST 3: Impossible map — a lethal wall bisects the grid.
    // A solid column of obstacles separates start from goal. The search
    // must exhaust the queue and report NO PATH without looping forever.
    // -----------------------------------------------------------------
    RunTestCase(
        "Test 3: Impossible Path",
        3, 4,
        {0, 100, 0, 0,
         0, 100, 0, 0,
         0, 100, 0, 0},
        {1, 0},
        {1, 3}
    );

    // -----------------------------------------------------------------
    // TEST 4: Penalty detour — semantic avoidance.
    // The direct route passes through an 80-penalty inflation zone; a
    // longer route stays in free space. A* should prefer the safer,
    // physically longer path because its total cost is lower.
    // -----------------------------------------------------------------
    RunTestCase(
        "Test 4: Penalty Detour",
        4, 4,
        {0,   0,   0, 0,
         0,  80,  80, 0,    // the short, dangerous gap
         0, 100, 100, 0,
         0,   0,   0, 0},   // the safe detour
        {1, 0},
        {1, 3}
    );

    return 0;
}
