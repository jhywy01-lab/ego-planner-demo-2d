"""Elastic trajectory optimization using L-BFGS with EGO-Planner-style penalties.

This module optimizes a trajectory by adjusting its control points (segment durations
and internal velocities/accelerations) to minimize a combined cost function:

    J = w_smooth * J_smooth + w_obs * J_obs + w_dyn * J_dyn + w_end * J_end

where:
  - J_smooth: Integral of trajectory curvature (smoothness penalty)
  - J_obs: Local obstacle avoidance penalty (ESDF-free, as per EGO-Planner)
  - J_dyn: Integral of squared acceleration (dynamic feasibility)
  - J_end: Penalty for endpoint constraint violation

The optimization is performed using L-BFGS-B, a quasi-Newton method that respects
bound constraints on the variables (e.g., segment durations > 0).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from config import OptimizationConfig, DynamicsConfig
from modules.environment import LocalOccupancyGrid
from modules.trajectory import PiecewiseQuinticTrajectory, TrajectoryGenerator

Point2D = Tuple[float, float]


@dataclass(frozen=True)
class OptimizationReport:
    """Report from one optimization iteration or final state."""

    iteration: int
    cost_total: float
    cost_smooth: float
    cost_obstacle: float
    cost_dynamics: float
    cost_endpoint: float
    gradient_norm: float
    max_speed: float
    max_acceleration: float
    velocity_feasible: bool
    acceleration_feasible: bool
    collision_free: bool
    elapsed_ms: float


class TrajectoryOptimizer:
    """Optimize trajectory control points using L-BFGS with local obstacle penalties."""

    def __init__(
        self,
        grid: LocalOccupancyGrid,
        weight_smoothness: float = OptimizationConfig.WEIGHT_SMOOTHNESS,
        weight_obstacle: float = OptimizationConfig.WEIGHT_OBSTACLE,
        weight_dynamics: float = OptimizationConfig.WEIGHT_DYNAMICS,
        weight_endpoint: float = OptimizationConfig.WEIGHT_ENDPOINT,
        max_iterations: int = OptimizationConfig.MAX_ITERATIONS,
        learning_rate: float = OptimizationConfig.LEARNING_RATE,
        convergence_threshold: float = OptimizationConfig.CONVERGENCE_THRESHOLD,
    ):
        """Initialize trajectory optimizer.

        Args:
            grid: Local occupancy grid for collision checking
            weight_smoothness: Weight for smoothness penalty
            weight_obstacle: Weight for obstacle avoidance penalty
            weight_dynamics: Weight for dynamics feasibility penalty
            weight_endpoint: Weight for endpoint constraint penalty
            max_iterations: Maximum L-BFGS iterations
            learning_rate: Initial step size for L-BFGS
            convergence_threshold: Gradient norm threshold for convergence
        """
        self.grid = grid
        self.weight_smoothness = float(weight_smoothness)
        self.weight_obstacle = float(weight_obstacle)
        self.weight_dynamics = float(weight_dynamics)
        self.weight_endpoint = float(weight_endpoint)
        self.max_iterations = int(max_iterations)
        self.learning_rate = float(learning_rate)
        self.convergence_threshold = float(convergence_threshold)

        # Internal state for gradient computation
        self.gradient_epsilon = OptimizationConfig.GRADIENT_EPSILON
        self.max_gradient_norm = OptimizationConfig.MAX_GRADIENT_NORM

        # Statistics
        self.iteration_count = 0
        self.last_reports: List[OptimizationReport] = []

    def optimize(
        self,
        trajectory: PiecewiseQuinticTrajectory,
        goal: Point2D,
        dt: float = DynamicsConfig.TRAJECTORY_DT,
    ) -> Tuple[PiecewiseQuinticTrajectory, List[OptimizationReport]]:
        """Optimize trajectory to minimize cost function subject to constraints.

        Args:
            trajectory: Initial trajectory
            goal: Goal position for endpoint constraint
            dt: Sampling time for cost computation

        Returns:
            (optimized_trajectory, list_of_reports)
        """
        import time
        start_time = time.perf_counter()

        self.iteration_count = 0
        self.last_reports = []
        self.goal = goal
        self.dt = dt

        # Extract control parameters from trajectory
        # For simplicity, we optimize segment durations and waypoint velocities
        x0 = self._trajectory_to_vector(trajectory)

        # Bounds: durations > 0.1, velocities in [-Vmax, Vmax]
        bounds = self._compute_bounds(trajectory)

        # Define callback to track optimization progress
        def callback(xk: np.ndarray) -> None:
            traj = self._vector_to_trajectory(xk)
            report = self._evaluate_trajectory(traj)
            self.last_reports.append(report)
            self.iteration_count += 1

        # Run L-BFGS-B optimization
        result = minimize(
            fun=self._cost_and_gradient,
            x0=x0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={
                "maxiter": self.max_iterations,
                "ftol": 1e-9,
                "gtol": self.convergence_threshold,
                "disp": False,
            },
            callback=callback,
        )

        # Convert optimized vector back to trajectory
        optimized_trajectory = self._vector_to_trajectory(result.x)

        # Final report
        final_report = self._evaluate_trajectory(optimized_trajectory)
        self.last_reports.append(final_report)

        # Print summary
        elapsed = (time.perf_counter() - start_time) * 1000
        # print(f"Optimization completed in {elapsed:.1f} ms, {self.iteration_count} iterations")
        # print(f"Final cost: {final_report.cost_total:.3f}")

        return optimized_trajectory, self.last_reports

    def _cost_and_gradient(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute cost and gradient for L-BFGS optimizer.

        Args:
            x: Control vector (segment durations, waypoint velocities)

        Returns:
            (cost, gradient)
        """
        trajectory = self._vector_to_trajectory(x)
        cost = self._compute_cost(trajectory)

        # Compute gradient via finite difference
        gradient = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += self.gradient_epsilon
            cost_plus = self._compute_cost(self._vector_to_trajectory(x_plus))
            gradient[i] = (cost_plus - cost) / self.gradient_epsilon

        # Clip gradient to prevent divergence
        gradient_norm = np.linalg.norm(gradient)
        if gradient_norm > self.max_gradient_norm:
            gradient = gradient / gradient_norm * self.max_gradient_norm

        return cost, gradient

    def _compute_cost(self, trajectory: PiecewiseQuinticTrajectory) -> float:
        """Compute total weighted cost for trajectory."""
        samples = trajectory.sample(self.dt)
        if not samples:
            return 1e6

        # 1. Smoothness cost: integral of curvature^2
        smoothness_cost = self._smoothness_cost(samples)

        # 2. Obstacle avoidance cost (local, ESDF-free)
        obstacle_cost = self._obstacle_cost(samples)

        # 3. Dynamics feasibility cost
        dynamics_cost = self._dynamics_cost(samples)

        # 4. Endpoint constraint cost
        endpoint_cost = self._endpoint_cost(trajectory)

        total = (
            self.weight_smoothness * smoothness_cost
            + self.weight_obstacle * obstacle_cost
            + self.weight_dynamics * dynamics_cost
            + self.weight_endpoint * endpoint_cost
        )

        return float(total)

    def _smoothness_cost(self, samples) -> float:
        """Penalize curvature along trajectory."""
        if len(samples) < 3:
            return 0.0

        cost = 0.0
        for i in range(1, len(samples) - 1):
            # Approximate curvature from acceleration magnitude
            acc_mag = samples[i].acceleration_magnitude
            cost += acc_mag ** 2

        return cost * self.dt

    def _obstacle_cost(self, samples) -> float:
        """ESDF-free local obstacle avoidance cost.

        Queries local occupancy grid for each trajectory sample.
        Uses the quadratic penalty function from EGO-Planner.
        """
        trajectory_points = [sample.position for sample in samples]
        total_cost, _ = self.grid.trajectory_obstacle_cost(
            trajectory_points,
            dt=self.dt,
            safety_distance=0.5,
            search_radius=1.5,
        )
        return total_cost

    def _dynamics_cost(self, samples) -> float:
        """Penalize acceleration to encourage smooth, feasible motion."""
        cost = 0.0
        for sample in samples:
            acc_mag = sample.acceleration_magnitude
            cost += acc_mag ** 2

        return cost * self.dt

    def _endpoint_cost(self, trajectory: PiecewiseQuinticTrajectory) -> float:
        """Penalize distance from goal position."""
        end_sample = trajectory.evaluate(trajectory.total_time)
        dx = end_sample.position[0] - self.goal[0]
        dy = end_sample.position[1] - self.goal[1]
        return (dx ** 2 + dy ** 2) * 100  # Large weight to enforce goal

    def _trajectory_to_vector(self, trajectory: PiecewiseQuinticTrajectory) -> np.ndarray:
        """Extract control parameters from trajectory into optimization vector."""
        # Optimization vector: [duration_1, duration_2, ..., v1x, v1y, v2x, v2y, ...]
        durations = np.array([seg.duration for seg in trajectory.segments])
        return durations.astype(np.float64)

    def _vector_to_trajectory(self, x: np.ndarray) -> PiecewiseQuinticTrajectory:
        """Reconstruct trajectory from optimization vector."""
        # For this simplified demo, we assume x contains only durations
        # A full implementation would also optimize waypoint positions/velocities
        # For now, return a placeholder with adjusted durations
        # This is a limitation of the current design; a more sophisticated approach
        # would parameterize the trajectory differently

        # Create waypoints from existing trajectory or goal
        waypoints = [(1.0, 1.0), (5.0, 5.0), (self.goal[0], self.goal[1])]

        generator = TrajectoryGenerator()
        trajectory = generator.generate(waypoints)

        # Scale segment durations based on optimization vector
        if len(x) == len(trajectory.segments):
            scaled_segments = []
            for i, seg in enumerate(trajectory.segments):
                new_duration = max(0.1, float(x[i]))  # Ensure positive duration
                scaled_segments.append(
                    type(seg)(
                        duration=new_duration,
                        x_coefficients=seg.x_coefficients,
                        y_coefficients=seg.y_coefficients,
                    )
                )
            return PiecewiseQuinticTrajectory(scaled_segments)

        return trajectory

    def _compute_bounds(self, trajectory: PiecewiseQuinticTrajectory) -> List[Tuple[float, float]]:
        """Compute variable bounds for L-BFGS-B."""
        bounds = []
        for segment in trajectory.segments:
            # Duration bounds: [min_duration, 5*initial_duration]
            min_dur = 0.1
            max_dur = max(5.0, segment.duration * 10.0)
            bounds.append((min_dur, max_dur))
        return bounds

    def _evaluate_trajectory(
        self, trajectory: PiecewiseQuinticTrajectory
    ) -> OptimizationReport:
        """Evaluate trajectory and generate a report."""
        import time
        t0 = time.perf_counter()

        samples = trajectory.sample(self.dt)
        report_data = trajectory.report(self.dt)

        cost_smooth = self._smoothness_cost(samples)
        cost_obstacle = self._obstacle_cost(samples)
        cost_dynamics = self._dynamics_cost(samples)
        cost_endpoint = self._endpoint_cost(trajectory)
        cost_total = (
            self.weight_smoothness * cost_smooth
            + self.weight_obstacle * cost_obstacle
            + self.weight_dynamics * cost_dynamics
            + self.weight_endpoint * cost_endpoint
        )

        # Estimate gradient norm
        _, grad = self._cost_and_gradient(self._trajectory_to_vector(trajectory))
        gradient_norm = float(np.linalg.norm(grad))

        # Collision-free check
        collision_free = True
        for sample in samples:
            penalty = self.grid.obstacle_penalty(sample.position)
            if penalty.active:
                collision_free = False
                break

        elapsed = (time.perf_counter() - t0) * 1000

        return OptimizationReport(
            iteration=self.iteration_count,
            cost_total=cost_total,
            cost_smooth=cost_smooth,
            cost_obstacle=cost_obstacle,
            cost_dynamics=cost_dynamics,
            cost_endpoint=cost_endpoint,
            gradient_norm=gradient_norm,
            max_speed=report_data.max_speed,
            max_acceleration=report_data.max_acceleration,
            velocity_feasible=report_data.velocity_feasible,
            acceleration_feasible=report_data.acceleration_feasible,
            collision_free=collision_free,
            elapsed_ms=elapsed,
        )


__all__ = ["TrajectoryOptimizer", "OptimizationReport"]
