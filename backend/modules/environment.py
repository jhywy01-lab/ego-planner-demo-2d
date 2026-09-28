"""ESDF-free local occupancy environment for the 2D EGO-Planner demo.

This module deliberately does *not* build or store a distance field.  During
optimization, a trajectory sample asks the local occupancy grid for nearby
occupied cells; the closest occupied cell is then used to evaluate the local
obstacle penalty and its analytic direction.

The implementation is a 2D educational analogue of EGO-Planner's local map
and obstacle-cost evaluation.  It is not a replacement for the project's ROS
map interface or for the original 3D planner.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from config import MapConfig, ObstacleConfig, grid_to_world, is_grid_in_bounds, world_to_grid

Point2D = Tuple[float, float]
GridIndex = Tuple[int, int]


@dataclass(frozen=True)
class ObstacleQuery:
    """Result of a local nearest-obstacle query.

    ``distance`` is the distance from the query point to the center of the
    nearest occupied cell.  No distance is cached in the map.
    """

    point: Point2D
    obstacle_cell: Optional[GridIndex]
    obstacle_point: Optional[Point2D]
    distance: float
    checked_cells: int


@dataclass(frozen=True)
class ObstaclePenalty:
    """Local obstacle cost and gradient for one trajectory sample."""

    cost: float
    gradient: Point2D
    distance: float
    active: bool
    obstacle_point: Optional[Point2D]
    checked_cells: int


class LocalOccupancyGrid:
    """A bounded 2D occupancy grid with on-demand local obstacle queries.

    Coordinates are expressed in metres and use a bottom-left origin.  The
    NumPy array is indexed as ``occupancy[y, x]``.  ``True`` means occupied.
    The grid stores only occupancy; it never stores ESDF, distance-transform,
    or gradient-field arrays.
    """

    def __init__(
        self,
        width: float = MapConfig.MAP_WIDTH,
        height: float = MapConfig.MAP_HEIGHT,
        resolution: float = MapConfig.GRID_RESOLUTION,
        origin: Point2D = (MapConfig.ORIGIN_X, MapConfig.ORIGIN_Y),
    ) -> None:
        if width <= 0 or height <= 0 or resolution <= 0:
            raise ValueError("width, height and resolution must be positive")
        self.width = float(width)
        self.height = float(height)
        self.resolution = float(resolution)
        self.origin = (float(origin[0]), float(origin[1]))
        self.cells_x = int(np.ceil(self.width / self.resolution))
        self.cells_y = int(np.ceil(self.height / self.resolution))
        self.occupancy = np.zeros((self.cells_y, self.cells_x), dtype=bool)

    @property
    def occupied_cells(self) -> int:
        return int(np.count_nonzero(self.occupancy))

    def clear(self) -> None:
        """Remove all occupied cells."""
        self.occupancy.fill(False)

    def in_bounds_grid(self, index: GridIndex) -> bool:
        x, y = index
        return 0 <= x < self.cells_x and 0 <= y < self.cells_y

    def in_bounds_world(self, point: Point2D) -> bool:
        x, y = point
        return (
            self.origin[0] <= x < self.origin[0] + self.width
            and self.origin[1] <= y < self.origin[1] + self.height
        )

    def world_to_grid(self, point: Point2D) -> GridIndex:
        """Convert a world point to its containing cell."""
        gx = int(np.floor((point[0] - self.origin[0]) / self.resolution))
        gy = int(np.floor((point[1] - self.origin[1]) / self.resolution))
        return gx, gy

    def grid_to_world(self, index: GridIndex) -> Point2D:
        """Return the center of a grid cell in world coordinates."""
        gx, gy = index
        return (
            self.origin[0] + (gx + 0.5) * self.resolution,
            self.origin[1] + (gy + 0.5) * self.resolution,
        )

    def set_occupied(self, index: GridIndex, occupied: bool = True) -> None:
        if not self.in_bounds_grid(index):
            raise IndexError(f"grid index outside map: {index}")
        gx, gy = index
        self.occupancy[gy, gx] = bool(occupied)

    def is_occupied(self, point: Point2D, outside_is_occupied: bool = True) -> bool:
        """Check the cell containing ``point`` without constructing a field."""
        if not self.in_bounds_world(point):
            return outside_is_occupied
        gx, gy = self.world_to_grid(point)
        return bool(self.occupancy[gy, gx])

    def set_obstacle_rect(self, x_min: float, y_min: float, x_max: float, y_max: float) -> None:
        """Insert an axis-aligned rectangle using cell-center intersection."""
        if x_min > x_max or y_min > y_max:
            raise ValueError("rectangle minimum must not exceed maximum")
        for gy in range(self.cells_y):
            for gx in range(self.cells_x):
                x, y = self.grid_to_world((gx, gy))
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    self.occupancy[gy, gx] = True

    def occupied_points(self) -> List[Point2D]:
        """Return occupied cell centers for visualization/debugging."""
        ys, xs = np.nonzero(self.occupancy)
        return [self.grid_to_world((int(x), int(y))) for x, y in zip(xs, ys)]

    def query_nearest_obstacle(
        self,
        point: Point2D,
        search_radius: float = ObstacleConfig.LOCAL_SEARCH_RADIUS,
        outside_is_occupied: bool = True,
    ) -> ObstacleQuery:
        """Search occupied cells locally around a trajectory sample.

        The search scans the occupancy array in a bounded window and computes
        distances only for this query.  This is intentionally not a nearest
        distance lookup table and therefore remains ESDF-free.
        """
        if search_radius < 0:
            raise ValueError("search_radius must be non-negative")

        gx, gy = self.world_to_grid(point)
        radius_cells = int(np.ceil(search_radius / self.resolution))
        candidates: List[Tuple[float, GridIndex, Point2D]] = []
        checked = 0

        for cy in range(gy - radius_cells, gy + radius_cells + 1):
            for cx in range(gx - radius_cells, gx + radius_cells + 1):
                if not self.in_bounds_grid((cx, cy)):
                    continue
                cell_point = self.grid_to_world((cx, cy))
                if hypot(cell_point[0] - point[0], cell_point[1] - point[1]) > search_radius:
                    continue
                checked += 1
                if self.occupancy[cy, cx]:
                    d = hypot(cell_point[0] - point[0], cell_point[1] - point[1])
                    candidates.append((d, (cx, cy), cell_point))

        if candidates:
            d, cell, obstacle_point = min(candidates, key=lambda item: item[0])
            return ObstacleQuery(point, cell, obstacle_point, d, checked)

        # Outside the local map is treated as unknown/unsafe by default.
        if outside_is_occupied and not self.in_bounds_world(point):
            return ObstacleQuery(point, None, None, 0.0, checked)
        return ObstacleQuery(point, None, None, float("inf"), checked)

    def obstacle_penalty(
        self,
        point: Point2D,
        safety_distance: float = ObstacleConfig.SAFETY_DISTANCE,
        search_radius: float = ObstacleConfig.LOCAL_SEARCH_RADIUS,
    ) -> ObstaclePenalty:
        """Evaluate the EGO-Planner-style piecewise quadratic local penalty.

        f_obs(d) = 0.5 * (d - r)^2, d < r
                 = 0,                 d >= r

        For a finite nearest obstacle, the gradient is computed from the
        local vector to that obstacle.  At zero distance the direction is
        undefined; a zero gradient is returned and the positive collision
        cost is still reported.  The caller can use a small finite-difference
        fallback if a non-zero direction is required at that exact point.
        """
        if safety_distance < 0:
            raise ValueError("safety_distance must be non-negative")
        query = self.query_nearest_obstacle(point, search_radius)
        if query.distance >= safety_distance:
            return ObstaclePenalty(0.0, (0.0, 0.0), query.distance, False, query.obstacle_point, query.checked_cells)

        d = query.distance
        cost = 0.5 * (d - safety_distance) ** 2
        if query.obstacle_point is None or d <= 1e-12:
            gradient = (0.0, 0.0)
        else:
            # grad[0.5(d-r)^2] = (d-r) * grad(d)
            direction = ((point[0] - query.obstacle_point[0]) / d, (point[1] - query.obstacle_point[1]) / d)
            gradient = ((d - safety_distance) * direction[0], (d - safety_distance) * direction[1])
        return ObstaclePenalty(cost, gradient, d, True, query.obstacle_point, query.checked_cells)

    def trajectory_obstacle_cost(
        self,
        points: Sequence[Point2D],
        dt: float = 1.0,
        safety_distance: float = ObstacleConfig.SAFETY_DISTANCE,
        search_radius: float = ObstacleConfig.LOCAL_SEARCH_RADIUS,
    ) -> Tuple[float, List[ObstaclePenalty]]:
        """Numerically integrate local obstacle cost over sampled trajectory points."""
        if dt <= 0:
            raise ValueError("dt must be positive")
        penalties = [self.obstacle_penalty(p, safety_distance, search_radius) for p in points]
        return float(sum(item.cost for item in penalties) * dt), penalties

    def export(self) -> dict:
        """Serialize occupancy for the frontend; no distance field is exported."""
        return {
            "width": self.width,
            "height": self.height,
            "resolution": self.resolution,
            "origin": list(self.origin),
            "cells_x": self.cells_x,
            "cells_y": self.cells_y,
            "occupied": self.occupancy.astype(np.uint8).tolist(),
            "occupied_cells": self.occupied_cells,
            "distance_field": None,
        }


__all__ = ["LocalOccupancyGrid", "ObstacleQuery", "ObstaclePenalty"]
