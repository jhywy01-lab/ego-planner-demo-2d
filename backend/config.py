"""Configuration and constants for EGO-Planner Demo 2D."""

import numpy as np
from typing import Tuple

# ============================================================================
# COORDINATE SYSTEM & MAP PARAMETERS
# ============================================================================

class MapConfig:
    """Map and coordinate system configuration."""
    
    # Map boundaries (in meters)
    MAP_WIDTH = 10.0  # meters
    MAP_HEIGHT = 10.0  # meters
    
    # Grid resolution
    GRID_RESOLUTION = 0.1  # meters per cell
    
    # Number of cells
    GRID_CELLS_X = int(MAP_WIDTH / GRID_RESOLUTION)  # 100
    GRID_CELLS_Y = int(MAP_HEIGHT / GRID_RESOLUTION)  # 100
    
    # Origin (bottom-left corner in world coordinates)
    ORIGIN_X = 0.0
    ORIGIN_Y = 0.0


# ============================================================================
# OBSTACLE & SAFETY PARAMETERS
# ============================================================================

class ObstacleConfig:
    """Obstacle representation and safety parameters."""
    
    # Safety distance (inflation radius around obstacles)
    SAFETY_DISTANCE = 0.5  # meters
    
    # Obstacle representation
    OBSTACLE_THRESHOLD = 0.5  # occupancy probability threshold (0-1)
    
    # Collision detection method: 'local_search' (ESDF-free)
    COLLISION_METHOD = 'local_search'
    
    # Local search radius for obstacle detection
    LOCAL_SEARCH_RADIUS = 1.5  # meters


# ============================================================================
# DYNAMICS CONSTRAINTS
# ============================================================================

class DynamicsConfig:
    """Quadrotor dynamics constraints."""
    
    # Velocity constraints
    MAX_VELOCITY = 3.0  # m/s
    MIN_VELOCITY = 0.0  # m/s
    
    # Acceleration constraints
    MAX_ACCELERATION = 2.0  # m/s²
    MIN_ACCELERATION = -2.0  # m/s²
    
    # Jerk constraints (derivative of acceleration)
    MAX_JERK = 5.0  # m/s³
    
    # Time step for trajectory discretization
    TRAJECTORY_DT = 0.1  # seconds


# ============================================================================
# TRAJECTORY GENERATION PARAMETERS
# ============================================================================

class TrajectoryConfig:
    """Trajectory generation parameters."""
    
    # Polynomial degree (5 = quintic)
    POLYNOMIAL_DEGREE = 5
    
    # Minimum segment duration
    MIN_SEGMENT_DURATION = 0.5  # seconds
    
    # Maximum segment duration
    MAX_SEGMENT_DURATION = 5.0  # seconds
    
    # Initial/final velocity and acceleration
    START_VELOCITY = 0.0  # m/s
    START_ACCELERATION = 0.0  # m/s²
    END_VELOCITY = 0.0  # m/s
    END_ACCELERATION = 0.0  # m/s²


# ============================================================================
# OPTIMIZATION PARAMETERS
# ============================================================================

class OptimizationConfig:
    """Elastic optimization parameters."""
    
    # Cost function weights
    WEIGHT_SMOOTHNESS = 1.0  # w_smooth
    WEIGHT_OBSTACLE = 2.0  # w_obs
    WEIGHT_DYNAMICS = 0.5  # w_dyn
    WEIGHT_ENDPOINT = 1.0  # w_end (endpoint constraint)
    
    # L-BFGS optimizer parameters
    OPTIMIZER_TYPE = 'L-BFGS-B'  # or 'SLSQP'
    LEARNING_RATE = 0.01  # Initial step size
    MAX_ITERATIONS = 50  # Maximum optimization iterations
    CONVERGENCE_THRESHOLD = 1e-4  # Convergence criterion
    
    # Finite difference for gradient computation
    GRADIENT_EPSILON = 1e-6  # Perturbation for numerical gradient
    
    # Gradient clipping to prevent divergence
    MAX_GRADIENT_NORM = 100.0  # Clip if norm exceeds this


# ============================================================================
# SEARCH (A*) PARAMETERS
# ============================================================================

class SearchConfig:
    """Topology-guided A* search parameters."""
    
    # Heuristic type
    HEURISTIC = 'manhattan'  # or 'euclidean'
    HEURISTIC_WEIGHT = 1.0  # Weight for heuristic (1.0 = optimal A*)
    
    # Number of topologically distinct paths to generate
    NUM_CANDIDATE_PATHS = 3  # Generate up to 3 different paths
    
    # Path simplification
    SIMPLIFY_WAYPOINTS = True
    WAYPOINT_MERGE_DISTANCE = 0.2  # meters


# ============================================================================
# EXECUTION & REPLANNING PARAMETERS
# ============================================================================

class ExecutionConfig:
    """Trajectory execution and replanning parameters."""
    
    # Replanning trigger conditions
    REPLANNING_TRIGGER_DISTANCE = 0.3  # meters (trigger if deviation exceeds)
    REPLANNING_TRIGGER_TIME_INTERVAL = 0.5  # seconds (trigger every N seconds)
    
    # Trajectory tracking controller (PID-like)
    CONTROL_DT = 0.01  # seconds (control loop frequency: 100 Hz)
    
    # Tracking error monitoring
    MAX_TRACKING_ERROR = 0.5  # meters (warning threshold)


# ============================================================================
# VISUALIZATION & UI PARAMETERS
# ============================================================================

class UIConfig:
    """UI and visualization parameters."""
    
    # Canvas size in pixels
    CANVAS_WIDTH = 800
    CANVAS_HEIGHT = 800
    
    # Grid visualization
    SHOW_GRID = True
    SHOW_DISTANCE_FIELD = False  # For comparison only (ESDF)
    
    # Color scheme (RGB values 0-255)
    COLOR_BACKGROUND = (255, 255, 255)  # White
    COLOR_GRID = (200, 200, 200)  # Light gray
    COLOR_OBSTACLE = (50, 50, 50)  # Dark gray
    COLOR_START = (0, 200, 0)  # Green
    COLOR_GOAL = (0, 0, 200)  # Blue
    COLOR_PATH_RAW = (255, 165, 0)  # Orange
    COLOR_TRAJECTORY = (0, 200, 200)  # Cyan
    COLOR_VELOCITY_VECTOR = (255, 0, 0)  # Red
    
    # Animation
    ANIMATION_SPEED = 1.0  # 1.0 = real-time
    AUTO_PLAY = False  # Auto-play trajectory after planning


# ============================================================================
# PERFORMANCE & MONITORING
# ============================================================================

class PerformanceConfig:
    """Performance monitoring and profiling."""
    
    # Enable detailed timing
    ENABLE_PROFILING = True
    
    # Timing thresholds for warnings (milliseconds)
    WARNING_TIME_SEARCH = 100.0  # A* search should be < 100ms
    WARNING_TIME_OPTIMIZATION = 500.0  # Optimization should be < 500ms
    WARNING_TIME_TOTAL = 1000.0  # Total planning should be < 1s


# ============================================================================
# SERVER CONFIGURATION
# ============================================================================

class ServerConfig:
    """FastAPI server configuration."""
    
    HOST = "0.0.0.0"
    PORT = 8000
    DEBUG = True
    WORKERS = 1
    
    # CORS settings
    CORS_ORIGINS = ["*"]
    
    # WebSocket settings
    WEBSOCKET_PING_INTERVAL = 30  # seconds
    WEBSOCKET_PING_TIMEOUT = 10  # seconds


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def world_to_grid(x: float, y: float) -> Tuple[int, int]:
    """Convert world coordinates to grid indices.
    
    Args:
        x, y: World coordinates in meters
    
    Returns:
        (grid_x, grid_y): Grid cell indices
    """
    grid_x = int((x - MapConfig.ORIGIN_X) / MapConfig.GRID_RESOLUTION)
    grid_y = int((y - MapConfig.ORIGIN_Y) / MapConfig.GRID_RESOLUTION)
    return grid_x, grid_y


def grid_to_world(grid_x: int, grid_y: int) -> Tuple[float, float]:
    """Convert grid indices to world coordinates.
    
    Args:
        grid_x, grid_y: Grid cell indices
    
    Returns:
        (x, y): World coordinates in meters
    """
    x = MapConfig.ORIGIN_X + (grid_x + 0.5) * MapConfig.GRID_RESOLUTION
    y = MapConfig.ORIGIN_Y + (grid_y + 0.5) * MapConfig.GRID_RESOLUTION
    return x, y


def is_in_bounds(x: float, y: float) -> bool:
    """Check if a point is within map bounds.
    
    Args:
        x, y: World coordinates in meters
    
    Returns:
        True if point is within bounds, False otherwise
    """
    return (MapConfig.ORIGIN_X <= x < MapConfig.ORIGIN_X + MapConfig.MAP_WIDTH and
            MapConfig.ORIGIN_Y <= y < MapConfig.ORIGIN_Y + MapConfig.MAP_HEIGHT)


def is_grid_in_bounds(grid_x: int, grid_y: int) -> bool:
    """Check if a grid cell is within map bounds.
    
    Args:
        grid_x, grid_y: Grid cell indices
    
    Returns:
        True if cell is within bounds, False otherwise
    """
    return (0 <= grid_x < MapConfig.GRID_CELLS_X and
            0 <= grid_y < MapConfig.GRID_CELLS_Y)
