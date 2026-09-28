"""Piecewise quintic trajectory initialization for the 2D demo.

The planner is represented as a piecewise polynomial curve.  Each segment
matches position, velocity and acceleration at both ends.  The default
boundary conditions are rest-to-rest, while internal derivatives are estimated
from neighbouring path points to avoid sharp stops at every waypoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from config import DynamicsConfig, TrajectoryConfig

Point2D = Tuple[float, float]


@dataclass(frozen=True)
class TrajectorySample:
    """State of the trajectory at one time instant."""

    t: float
    position: Point2D
    velocity: Point2D
    acceleration: Point2D

    @property
    def speed(self) -> float:
        return hypot(*self.velocity)

    @property
    def acceleration_magnitude(self) -> float:
        return hypot(*self.acceleration)


@dataclass(frozen=True)
class PolynomialSegment:
    """One local-time quintic segment, coefficients low-order first."""

    duration: float
    x_coefficients: Tuple[float, ...]
    y_coefficients: Tuple[float, ...]

    def evaluate(self, local_t: float) -> TrajectorySample:
        t = float(np.clip(local_t, 0.0, self.duration))
        x = _evaluate(self.x_coefficients, t)
        y = _evaluate(self.y_coefficients, t)
        vx = _evaluate_derivative(self.x_coefficients, t, 1)
        vy = _evaluate_derivative(self.y_coefficients, t, 1)
        ax = _evaluate_derivative(self.x_coefficients, t, 2)
        ay = _evaluate_derivative(self.y_coefficients, t, 2)
        return TrajectorySample(t, (x, y), (vx, vy), (ax, ay))


@dataclass(frozen=True)
class TrajectoryReport:
    """Feasibility and geometry metrics for a generated trajectory."""

    total_time: float
    total_length: float
    max_speed: float
    max_acceleration: float
    velocity_feasible: bool
    acceleration_feasible: bool

    @property
    def feasible(self) -> bool:
        return self.velocity_feasible and self.acceleration_feasible


@dataclass
class PiecewiseQuinticTrajectory:
    """A time-parameterized, piecewise quintic trajectory."""

    segments: List[PolynomialSegment]
    start_time: float = 0.0

    @property
    def total_time(self) -> float:
        return sum(segment.duration for segment in self.segments)

    def evaluate(self, t: float) -> TrajectorySample:
        if not self.segments:
            raise ValueError("trajectory has no segments")
        relative_t = float(np.clip(t - self.start_time, 0.0, self.total_time))
        elapsed = 0.0
        for segment in self.segments:
            if relative_t <= elapsed + segment.duration:
                sample = segment.evaluate(relative_t - elapsed)
                return TrajectorySample(
                    float(np.clip(t, self.start_time, self.start_time + self.total_time)),
                    sample.position,
                    sample.velocity,
                    sample.acceleration,
                )
            elapsed += segment.duration
        return self.segments[-1].evaluate(self.segments[-1].duration)

    def sample(self, dt: float = DynamicsConfig.TRAJECTORY_DT) -> List[TrajectorySample]:
        if dt <= 0:
            raise ValueError("dt must be positive")
        if not self.segments:
            return []
        count = max(1, int(np.ceil(self.total_time / dt)))
        times = np.linspace(0.0, self.total_time, count + 1)
        return [self.evaluate(self.start_time + float(t)) for t in times]

    def report(self, dt: float = DynamicsConfig.TRAJECTORY_DT) -> TrajectoryReport:
        samples = self.sample(dt)
        if not samples:
            return TrajectoryReport(0.0, 0.0, 0.0, 0.0, True, True)
        length = sum(
            hypot(samples[i].position[0] - samples[i - 1].position[0],
                  samples[i].position[1] - samples[i - 1].position[1])
            for i in range(1, len(samples))
        )
        max_speed = max(item.speed for item in samples)
        max_acceleration = max(item.acceleration_magnitude for item in samples)
        return TrajectoryReport(
            self.total_time,
            length,
            max_speed,
            max_acceleration,
            max_speed <= DynamicsConfig.MAX_VELOCITY + 1e-9,
            max_acceleration <= DynamicsConfig.MAX_ACCELERATION + 1e-9,
        )

    def to_dict(self, dt: float = DynamicsConfig.TRAJECTORY_DT) -> dict:
        """Serialize samples and coefficients for the UI."""
        report = self.report(dt)
        return {
            "total_time": report.total_time,
            "segments": [
                {
                    "duration": segment.duration,
                    "x_coefficients": list(segment.x_coefficients),
                    "y_coefficients": list(segment.y_coefficients),
                }
                for segment in self.segments
            ],
            "samples": [
                {
                    "t": sample.t,
                    "position": list(sample.position),
                    "velocity": list(sample.velocity),
                    "acceleration": list(sample.acceleration),
                }
                for sample in self.sample(dt)
            ],
            "report": {
                "total_length": report.total_length,
                "max_speed": report.max_speed,
                "max_acceleration": report.max_acceleration,
                "velocity_feasible": report.velocity_feasible,
                "acceleration_feasible": report.acceleration_feasible,
                "feasible": report.feasible,
            },
        }


class TrajectoryGenerator:
    """Generate a piecewise quintic trajectory from path waypoints."""

    def __init__(
        self,
        max_velocity: float = DynamicsConfig.MAX_VELOCITY,
        max_acceleration: float = DynamicsConfig.MAX_ACCELERATION,
        min_segment_duration: float = TrajectoryConfig.MIN_SEGMENT_DURATION,
    ) -> None:
        if max_velocity <= 0 or max_acceleration <= 0 or min_segment_duration <= 0:
            raise ValueError("trajectory limits and minimum duration must be positive")
        self.max_velocity = float(max_velocity)
        self.max_acceleration = float(max_acceleration)
        self.min_segment_duration = float(min_segment_duration)

    def generate(self, waypoints: Sequence[Point2D]) -> PiecewiseQuinticTrajectory:
        """Create rest-to-rest or internally smooth quintic segments.

        Waypoint positions are interpolated exactly.  Internal velocities are
        estimated by centred differences and then scaled to respect the
        configured velocity limit.  Accelerations are estimated from adjacent
        velocities and are conservatively scaled when necessary.
        """
        points = [(float(x), float(y)) for x, y in waypoints]
        if len(points) < 2:
            raise ValueError("at least two waypoints are required")
        durations = self._durations(points)
        velocities = self._estimate_velocities(points, durations)
        accelerations = self._estimate_accelerations(velocities, durations)
        segments = []
        for i, duration in enumerate(durations):
            coefficients_x = _quintic_coefficients(
                points[i][0], velocities[i][0], accelerations[i][0],
                points[i + 1][0], velocities[i + 1][0], accelerations[i + 1][0], duration,
            )
            coefficients_y = _quintic_coefficients(
                points[i][1], velocities[i][1], accelerations[i][1],
                points[i + 1][1], velocities[i + 1][1], accelerations[i + 1][1], duration,
            )
            segments.append(PolynomialSegment(duration, coefficients_x, coefficients_y))
        return PiecewiseQuinticTrajectory(segments)

    def _durations(self, points: Sequence[Point2D]) -> List[float]:
        # A conservative time assignment; optimization can later refine timing.
        result = []
        for first, second in zip(points, points[1:]):
            distance = hypot(second[0] - first[0], second[1] - first[1])
            velocity_time = distance / self.max_velocity
            acceleration_time = 2.0 * np.sqrt(distance / self.max_acceleration) if distance else 0.0
            result.append(max(self.min_segment_duration, velocity_time, acceleration_time))
        return result

    def _estimate_velocities(self, points: Sequence[Point2D], durations: Sequence[float]) -> List[Point2D]:
        velocities: List[Point2D] = [(0.0, 0.0)]
        for i in range(1, len(points) - 1):
            dt = durations[i - 1] + durations[i]
            velocity = (
                (points[i + 1][0] - points[i - 1][0]) / dt,
                (points[i + 1][1] - points[i - 1][1]) / dt,
            )
            speed = hypot(*velocity)
            if speed > self.max_velocity:
                scale = self.max_velocity / speed
                velocity = (velocity[0] * scale, velocity[1] * scale)
            velocities.append(velocity)
        velocities.append((0.0, 0.0))
        return velocities

    def _estimate_accelerations(self, velocities: Sequence[Point2D], durations: Sequence[float]) -> List[Point2D]:
        accelerations: List[Point2D] = [(0.0, 0.0)]
        for i in range(1, len(velocities) - 1):
            dt = max(durations[i - 1], 1e-9)
            acceleration = (
                (velocities[i][0] - velocities[i - 1][0]) / dt,
                (velocities[i][1] - velocities[i - 1][1]) / dt,
            )
            magnitude = hypot(*acceleration)
            if magnitude > self.max_acceleration:
                scale = self.max_acceleration / magnitude
                acceleration = (acceleration[0] * scale, acceleration[1] * scale)
            accelerations.append(acceleration)
        accelerations.append((0.0, 0.0))
        return accelerations


def _quintic_coefficients(p0: float, v0: float, a0: float, pf: float, vf: float, af: float, duration: float) -> Tuple[float, ...]:
    """Solve the six endpoint constraints for low-order coefficients."""
    if duration <= 0:
        raise ValueError("segment duration must be positive")
    t = duration
    matrix = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0],
        [1, t, t**2, t**3, t**4, t**5],
        [0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4],
        [0, 0, 2, 6*t, 12*t**2, 20*t**3],
    ], dtype=float)
    values = np.array([p0, v0, a0, pf, vf, af], dtype=float)
    return tuple(float(value) for value in np.linalg.solve(matrix, values))


def _evaluate(coefficients: Sequence[float], t: float) -> float:
    return float(sum(coefficient * t** power for power, coefficient in enumerate(coefficients)))


def _evaluate_derivative(coefficients: Sequence[float], t: float, order: int) -> float:
    values = list(coefficients)
    for _ in range(order):
        values = [index * value for index, value in enumerate(values)][1:]
    return _evaluate(values, t) if values else 0.0


__all__ = [
    "PiecewiseQuinticTrajectory",
    "PolynomialSegment",
    "TrajectoryGenerator",
    "TrajectoryReport",
    "TrajectorySample",
]
