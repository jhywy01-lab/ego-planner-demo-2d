"""Tests for the ESDF-free environment module."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from modules.environment import LocalOccupancyGrid


def test_grid_coordinate_round_trip_uses_cell_centres():
    grid = LocalOccupancyGrid(width=2.0, height=2.0, resolution=0.1)
    index = grid.world_to_grid((0.24, 1.76))
    assert index == (2, 17)
    assert grid.grid_to_world(index) == pytest.approx((0.25, 1.75))


def test_local_query_finds_nearest_occupied_cell_without_distance_field():
    grid = LocalOccupancyGrid(width=2.0, height=2.0, resolution=0.1)
    grid.set_occupied((10, 10))
    query = grid.query_nearest_obstacle((1.15, 1.05), search_radius=0.5)
    assert query.obstacle_cell == (10, 10)
    assert query.distance == pytest.approx(math.hypot(0.10, 0.0))
    assert not hasattr(grid, "distance_field")
    assert grid.export()["distance_field"] is None


def test_piecewise_quadratic_penalty_and_gradient():
    grid = LocalOccupancyGrid(width=2.0, height=2.0, resolution=0.1)
    grid.set_occupied((10, 10))  # center=(1.05, 1.05)
    penalty = grid.obstacle_penalty((1.25, 1.05), safety_distance=0.5, search_radius=1.0)
    expected_distance = 0.2
    assert penalty.active
    assert penalty.distance == pytest.approx(expected_distance)
    assert penalty.cost == pytest.approx(0.5 * (expected_distance - 0.5) ** 2)
    assert penalty.gradient[0] == pytest.approx(expected_distance - 0.5)
    assert penalty.gradient[1] == pytest.approx(0.0)


def test_zero_cost_outside_safety_distance():
    grid = LocalOccupancyGrid(width=2.0, height=2.0, resolution=0.1)
    grid.set_occupied((10, 10))
    penalty = grid.obstacle_penalty((1.8, 1.05), safety_distance=0.5, search_radius=1.0)
    assert penalty.cost == 0.0
    assert not penalty.active


def test_trajectory_cost_is_integrated_over_dt():
    grid = LocalOccupancyGrid(width=2.0, height=2.0, resolution=0.1)
    grid.set_occupied((10, 10))
    points = [(1.25, 1.05), (1.25, 1.05)]
    total, penalties = grid.trajectory_obstacle_cost(points, dt=0.1)
    assert len(penalties) == 2
    assert total == pytest.approx(2 * 0.5 * 0.3**2 * 0.1)


def test_rectangle_insertion_and_clear():
    grid = LocalOccupancyGrid(width=2.0, height=2.0, resolution=0.1)
    grid.set_obstacle_rect(0.5, 0.5, 0.9, 0.9)
    assert grid.occupied_cells > 0
    assert grid.is_occupied((0.55, 0.55))
    grid.clear()
    assert grid.occupied_cells == 0
