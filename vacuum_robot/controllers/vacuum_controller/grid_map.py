# =============================================================================
# grid_map.py — Occupancy Grid Map Module
# =============================================================================
# MAP layer of the Sense-Think-Act architecture.
#
# Responsibilities:
#   - 49x49 occupancy grid covering the 10m x 10m arena (20cm per cell)
#   - World-to-grid and grid-to-world coordinate conversion
#   - Visited / obstacle / wall cell tracking
#   - Coverage percentage calculation for task completion detection
#
# Cell States:
#   UNVISITED (0) — floor not yet cleaned
#   VISITED   (1) — floor cleaned by robot
#   OBSTACLE  (2) — confirmed obstacle (robot physically hit it)
#   WALL      (3) — arena boundary
#
# Obstacle marking strategy:
#   Obstacles marked only when robot enters AVOID state. The actual
#   measured sonar distance is used to place the mark at the obstacle
#   face rather than a fixed offset. Arena walls are never marked as
#   obstacles — they are already WALL cells.
# =============================================================================

import numpy as np
import math

ARENA_MIN_X = -4.9
ARENA_MAX_X =  4.9
ARENA_MIN_Y = -4.9
ARENA_MAX_Y =  4.9

CELL_SIZE   = 0.20
GRID_W      = int((ARENA_MAX_X - ARENA_MIN_X) / CELL_SIZE)  # 49
GRID_H      = int((ARENA_MAX_Y - ARENA_MIN_Y) / CELL_SIZE)  # 49

UNVISITED   = 0
VISITED     = 1
OBSTACLE    = 2
WALL        = 3


class GridMap:
    """
    49x49 occupancy grid representing the arena floor.
    Each cell is 20cm x 20cm. Outer border pre-marked as WALL.
    Covers the full 10m x 10m arena (walls at +-5.01m).
    """

    def __init__(self):
        self.grid = np.zeros((GRID_H, GRID_W), dtype=np.uint8)
        self.grid[0, :]          = WALL
        self.grid[GRID_H - 1, :] = WALL
        self.grid[:, 0]          = WALL
        self.grid[:, GRID_W - 1] = WALL
        print(f"  [grid] {GRID_W}x{GRID_H} cells "
              f"({GRID_W * CELL_SIZE:.1f}m x {GRID_H * CELL_SIZE:.1f}m) "
              f"cell={CELL_SIZE * 100:.0f}cm")

    def world_to_grid(self, wx, wy):
        """Convert GPS world coordinates (metres) to grid (row, col)."""
        col = int((wx - ARENA_MIN_X) / CELL_SIZE)
        row = int((wy - ARENA_MIN_Y) / CELL_SIZE)
        col = max(0, min(GRID_W - 1, col))
        row = max(0, min(GRID_H - 1, row))
        return row, col

    def grid_to_world(self, row, col):
        """Convert grid (row, col) back to world coordinates (metres)."""
        wx = ARENA_MIN_X + col * CELL_SIZE + CELL_SIZE / 2
        wy = ARENA_MIN_Y + row * CELL_SIZE + CELL_SIZE / 2
        return wx, wy

    def in_bounds(self, row, col):
        return 0 <= row < GRID_H and 0 <= col < GRID_W

    def mark_visited(self, row, col):
        """Mark a cell as cleaned. Will not overwrite WALL or OBSTACLE."""
        if self.in_bounds(row, col) and self.grid[row][col] == UNVISITED:
            self.grid[row][col] = VISITED

    def mark_obstacle(self, row, col):
        """Mark a cell as an obstacle. Will not overwrite WALL cells."""
        if self.in_bounds(row, col) and self.grid[row][col] != WALL:
            self.grid[row][col] = OBSTACLE

    def mark_obstacle_ahead(self, robot_x, robot_y, robot_theta,
                             zones, front_dist=0.35):
        """
        Mark the obstacle face using the actual measured sonar distance.

        Uses the minimum front sonar reading as the real distance to
        the obstacle, placing the mark at the obstacle face rather than
        a fixed offset. Arena walls (already WALL cells) are never
        overwritten — this prevents false obstacle marks at boundaries.

        Three cells are marked: directly ahead, and slightly left/right
        to capture the width of the obstacle face.
        """
        # Use actual measured sonar distance for accurate placement
        measured = min(zones.get('front_wide', front_dist),
                       zones.get('front',      front_dist))
        # Small buffer (5cm) places mark just past the obstacle face
        dist = min(measured + 0.05, front_dist)

        for angle_offset in [0, -0.25, 0.25]:
            hx = robot_x + dist * math.cos(robot_theta + angle_offset)
            hy = robot_y + dist * math.sin(robot_theta + angle_offset)
            r, c = self.world_to_grid(hx, hy)
            # Never mark arena walls as obstacles
            if self.in_bounds(r, c) and self.grid[r][c] != WALL:
                self.mark_obstacle(r, c)

    def coverage_percent(self):
        """
        Calculate percentage of cleanable floor that has been visited.
        Excludes WALL and OBSTACLE cells from the denominator — these
        are physically inaccessible and cannot be cleaned.
        """
        obstacle_cells = np.sum(
            (self.grid == OBSTACLE) | (self.grid == WALL))
        cleanable = GRID_W * GRID_H - obstacle_cells
        visited   = np.sum(self.grid == VISITED)
        return (visited / cleanable * 100) if cleanable > 0 else 0.0