# A* Search on a Semantic Costmap

**Problem:** Given a 2D costmap (a flat row-major grid of integer costs `0–100`), find the lowest-cost path from a `start` cell to a `goal` cell on an **8-connected** grid. Cells at cost **100** are lethal/impassable; costs in `[0, 99]` are traversal penalties (e.g. the inflation zones produced by [semantic-costmap-bfs](../semantic-costmap-bfs/README.md)).

This is the planning half of a robot navigation stack: the costmap layer paints obstacles and safety buffers, then the planner routes around them, trading physical distance for safety.

---

## Algorithm — A* Search (`AStarSearch`)

Standard A* over the grid:

- **Movement:** 8 neighbors. Orthogonal steps cost `1`, diagonal steps cost `sqrt(2)`.
- **Edge cost:** `step_cost + master_costmap[neighbor]` — entering a high-cost cell is expensive, so the planner prefers longer routes through free space.
- **g / f split:** `g` is the accumulated path cost; `f = g + h` orders the priority queue. (Keeping these separate is essential — see Bug #1.)
- **Heuristic:** **octile distance** — the cheapest possible movement cost on an 8-connected grid ignoring penalties. It is admissible *and* consistent, so the first time a cell is popped it is optimal and can be settled permanently.
- **Lazy deletion:** stale duplicates left in the queue are skipped via a `visited` map on pop.
- **Corner-cutting prevention:** a diagonal move is rejected if either orthogonally-adjacent cell is lethal, so the robot can't squeeze through the corner of an obstacle.

Returns the path **start → goal inclusive**, or an empty vector if the goal is unreachable.

---

## Key Bugs Fixed (from the original)

| # | Bug | Fix |
|---|-----|-----|
| 1 | **Not optimal A\*.** `cost` stored `g + h`, then each step did `cost + step + h` again — compounding the heuristic on every expansion | Track `g` (path cost) and `f = g + h` separately; order the queue by `f` |
| 2 | **Obstacle semantics.** Cost `100` was treated as a passable high-cost cell, so the "impossible" map routed straight through the wall | Treat `cost >= 100` as impassable (`kLethalCost`) |
| 3 | **Typo** `1.141` for a diagonal step cost | All diagonals use `sqrt(2)` |
| 4 | **Weak heuristic** (Euclidean) | Octile distance — tighter, admissible, and consistent on 8-connected grids |
| 5 | **No corner-cutting guard** — diagonals could clip obstacle corners | Reject diagonal if either adjacent orthogonal cell is lethal |
| 6 | **Contract mismatch** — docstring promised start→goal, code returned goal→start | Reconstruct and `reverse` to start→goal, inclusive |
| 7 | Missing bounds/lethal input guards; missing `<cmath>`; debug `printf`s | Added guards and include; removed debug output |
| 8 | **Test 2 was a sealed box** (no escape possible regardless of semantics) | Rebuilt as a real escapable U-trap with a left-facing opening |

---

## Build & Run

```bash
g++ -O3 -std=c++17 a_star_pathfinding.cpp -o a.out
./a.out
```

> Note: an IDE language server may flag `constexpr` / `> >` — that's clang parsing the file as C / pre-C++11. It compiles cleanly under `-std=c++17 -Wall -Wextra`.

---

## Test Cases

| Test | Validates | Expected |
|------|-----------|----------|
| 1 — Open Space Diagonal | Prefers `sqrt(2)` diagonals over orthogonal zig-zags | Straight diagonal `(0,0)→(3,3)` |
| 2 — U-Shaped Trap Escape | Escapes a local minimum: greedy pull is right, but the only opening faces left | Backs out left, routes around the walls |
| 3 — Impossible Path | A lethal column bisects the grid; queue exhausts without looping | `NO PATH FOUND` |
| 4 — Penalty Detour | Chooses a longer free-space route over a short high-penalty (`80`) gap | Detours through cost-`0` cells |
