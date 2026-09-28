"""Tests for piecewise quintic trajectory initialization."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from modules.trajectory import TrajectoryGenerator


def test_quintic_trajectory_interpolates_waypoints_and_rest_boundaries():
    trajectory = TrajectoryGenerator().generate([(1.0, 1.0), (3.0, 1.0), (4.0, 4.0)])
    start = trajectory.evaluate(0.0)
    end = trajectory.evaluate(trajectory.total_time)
    assert start.position == pytest.approx((1.0, 1.0))
    assert end.position == pytest.approx((4.0, 4.0))
    assert start.velocity == pytest.approx((0.0, 0.0))
    assert end.velocity == pytest.approx((0.0, 0.0))
    assert start.acceleration == pytest.approx((0.0, 0.0))
    assert end.acceleration == pytest.approx((0.0, 0.0))


def test_internal_waypoint_is_continuous():
    trajectory = TrajectoryGenerator().generate([(0.0, 0.0), (2.0, 1.0), (4.0, 0.0)])
    first = trajectory.segments[0].evaluate(trajectory.segments[0].duration)
    second = trajectory.segments[1].evaluate(0.0)
    assert first.position == pytest.approx(second.position)
    assert first.velocity == pytest.approx(second.velocity)
    assert first.acceleration == pytest.approx(second.acceleration)


def test_report_checks_dynamic_limits():
    trajectory = TrajectoryGenerator().generate([(0.0, 0.0), (5.0, 0.0)])
    report = trajectory.report(dt=0.02)
    assert report.total_time >= 0.5
    assert report.max_speed <= 3.0 + 1e-6
    assert report.max_acceleration <= 2.0 + 1e-6
    assert report.feasible


def test_invalid_waypoints_are_rejected():
    with pytest.raises(ValueError):
        TrajectoryGenerator().generate([(0.0, 0.0)])
