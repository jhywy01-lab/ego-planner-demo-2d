"""Utility functions for EGO-Planner Demo."""

import numpy as np
from typing import List, Tuple, Optional
from config import MapConfig, DynamicsConfig, ObstacleConfig
import time
from functools import wraps


# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.elapsed_ms = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        # print(f"{self.name}: {self.elapsed_ms:.2f} ms")
        return False


def timing_decorator(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with PerformanceTimer(func.__name__) as timer:
            result = func(*args, **kwargs)
        return result, timer.elapsed_ms
    return wrapper


# ============================================================================
# GEOMETRIC UTILITIES
# ============================================================================

def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def distance_point_to_segment(
    point: Tuple[float, float],
    seg_start: Tuple[float, float],
    seg_end: Tuple[float, float]
) -> float:
    """Minimum distance from point to line segment."""
    x, y = point
    x1, y1 = seg_start
    x2, y2 = seg_end
    
    # Vector from start to end
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0 and dy == 0:
        return distance(point, seg_start)
    
    # Parameter t for closest point on line (clamped to [0, 1])
    t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / (dx**2 + dy**2)))
    
    # Closest point on segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    return distance(point, (closest_x, closest_y))


def point_to_point_direction(
    from_point: Tuple[float, float],
    to_point: Tuple[float, float]
) -> Tuple[float, float]:
    """Normalized direction vector from one point to another."""
    dx = to_point[0] - from_point[0]
    dy = to_point[1] - from_point[1]
    dist = np.sqrt(dx**2 + dy**2)
    
    if dist < 1e-9:
        return 0.0, 0.0
    
    return dx / dist, dy / dist


# ============================================================================
# TRAJECTORY UTILITIES
# ============================================================================

def discretize_trajectory(
    trajectory_func,
    t_start: float,
    t_end: float,
    dt: float
) -> List[Tuple[float, float]]:
    """Discretize a continuous trajectory function into waypoints.
    
    Args:
        trajectory_func: Function that returns (x, y) given time t
        t_start: Start time
        t_end: End time
        dt: Time step
    
    Returns:
        List of (x, y) waypoints
    """
    waypoints = []
    t = t_start
    while t <= t_end:
        waypoints.append(trajectory_func(t))
        t += dt
    return waypoints


def trajectory_length(waypoints: List[Tuple[float, float]]) -> float:
    """Calculate total path length from waypoints."""
    total = 0.0
    for i in range(len(waypoints) - 1):
        total += distance(waypoints[i], waypoints[i + 1])
    return total


def simplify_waypoints(
    waypoints: List[Tuple[float, float]],
    merge_distance: float
) -> List[Tuple[float, float]]:
    """Simplify waypoints by merging nearby ones.
    
    Args:
        waypoints: Original waypoint list
        merge_distance: Threshold distance for merging
    
    Returns:
        Simplified waypoint list
    """
    if len(waypoints) <= 2:
        return waypoints
    
    simplified = [waypoints[0]]
    
    for wp in waypoints[1:-1]:
        if distance(simplified[-1], wp) >= merge_distance:
            simplified.append(wp)
    
    simplified.append(waypoints[-1])
    return simplified


# ============================================================================
# CONSTRAINT CHECKING
# ============================================================================

def check_velocity_constraint(
    velocities: np.ndarray
) -> Tuple[bool, float]:
    """Check if all velocities satisfy constraints.
    
    Args:
        velocities: Array of velocity magnitudes
    
    Returns:
        (is_satisfied, max_violation) - max_violation is the excess over limit
    """
    max_vel = np.max(np.abs(velocities))
    violated = max_vel > DynamicsConfig.MAX_VELOCITY
    violation = max(0, max_vel - DynamicsConfig.MAX_VELOCITY)
    return not violated, violation


def check_acceleration_constraint(
    accelerations: np.ndarray
) -> Tuple[bool, float]:
    """Check if all accelerations satisfy constraints.
    
    Args:
        accelerations: Array of acceleration magnitudes
    
    Returns:
        (is_satisfied, max_violation)
    """
    max_acc = np.max(np.abs(accelerations))
    violated = max_acc > DynamicsConfig.MAX_ACCELERATION
    violation = max(0, max_acc - DynamicsConfig.MAX_ACCELERATION)
    return not violated, violation


def clamp_velocity(vel: float) -> float:
    """Clamp velocity to allowed range."""
    return np.clip(vel, DynamicsConfig.MIN_VELOCITY, DynamicsConfig.MAX_VELOCITY)


def clamp_acceleration(acc: float) -> float:
    """Clamp acceleration to allowed range."""
    return np.clip(acc, DynamicsConfig.MIN_ACCELERATION, DynamicsConfig.MAX_ACCELERATION)


# ============================================================================
# NUMERICAL UTILITIES
# ============================================================================

def normalize_vector(v: Tuple[float, float]) -> Tuple[float, float]:
    """Normalize a 2D vector."""
    norm = np.sqrt(v[0]**2 + v[1]**2)
    if norm < 1e-9:
        return 0.0, 0.0
    return v[0] / norm, v[1] / norm


def vector_magnitude(v: Tuple[float, float]) -> float:
    """Get magnitude of a 2D vector."""
    return np.sqrt(v[0]**2 + v[1]**2)


def dot_product(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    """Dot product of two 2D vectors."""
    return v1[0] * v2[0] + v1[1] * v2[1]


def cross_product_2d(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    """2D cross product (returns scalar)."""
    return v1[0] * v2[1] - v1[1] * v2[0]


# ============================================================================
# POLYNOMIAL UTILITIES
# ============================================================================

def evaluate_polynomial(coeffs: List[float], t: float) -> float:
    """Evaluate polynomial at time t.
    
    Args:
        coeffs: Polynomial coefficients [a0, a1, a2, ...] for a0 + a1*t + a2*t^2 + ...
        t: Time point
    
    Returns:
        Polynomial value at t
    """
    result = 0.0
    for i, coeff in enumerate(coeffs):
        result += coeff * (t ** i)
    return result


def evaluate_polynomial_derivative(coeffs: List[float], t: float, order: int = 1) -> float:
    """Evaluate derivative of polynomial at time t.
    
    Args:
        coeffs: Original polynomial coefficients
        t: Time point
        order: Derivative order (1 = velocity, 2 = acceleration)
    
    Returns:
        Derivative value at t
    """
    if order == 0:
        return evaluate_polynomial(coeffs, t)
    
    # Compute derivative coefficients
    deriv_coeffs = []
    for i in range(1, len(coeffs)):
        deriv_coeffs.append(i * coeffs[i])
    
    if not deriv_coeffs:
        return 0.0
    
    if order == 1:
        return evaluate_polynomial(deriv_coeffs, t)
    else:
        return evaluate_polynomial_derivative(deriv_coeffs, t, order - 1)


def polynomial_to_string(coeffs: List[float], var: str = 't') -> str:
    """Convert polynomial coefficients to readable string.
    
    Args:
        coeffs: Polynomial coefficients
        var: Variable name
    
    Returns:
        String representation
    """
    terms = []
    for i, coeff in enumerate(coeffs):
        if abs(coeff) < 1e-9:
            continue
        
        if i == 0:
            terms.append(f"{coeff:.3f}")
        elif i == 1:
            terms.append(f"{coeff:.3f}*{var}")
        else:
            terms.append(f"{coeff:.3f}*{var}^{i}")
    
    if not terms:
        return "0"
    
    return " + ".join(terms).replace("+ -", "- ")
