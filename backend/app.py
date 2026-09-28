"""Main FastAPI application for EGO-Planner 2D Demo.

Integrates all four core modules:
1. LocalOccupancyGrid (environment sensing)
2. AStarPlanner (topology-guided search)
3. TrajectoryGenerator (trajectory initialization)
4. TrajectoryOptimizer (elastic optimization)

Provides REST endpoints and WebSocket for real-time visualization.
"""

from __future__ import annotations

import asyncio
import time
from typing import Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from config import MapConfig, ObstacleConfig, DynamicsConfig, ServerConfig
from modules.environment import LocalOccupancyGrid
from modules.search import AStarPlanner, simplify_waypoints, smooth_waypoints
from modules.trajectory import TrajectoryGenerator, PiecewiseQuinticTrajectory
from modules.optimization import TrajectoryOptimizer

# ============================================================================
# Initialize FastAPI application
# ============================================================================

app = FastAPI(
    title="EGO-Planner 2D Demo",
    description="Interactive ESDF-free trajectory planning visualization",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ServerConfig.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Global state (in production, use proper state management)
# ============================================================================

class PlannerState:
    """Global planner state."""
    
    def __init__(self):
        self.grid = LocalOccupancyGrid()
        self.start: Optional[tuple] = None
        self.goal: Optional[tuple] = None
        self.raw_paths: List[List[tuple]] = []
        self.trajectory: Optional[PiecewiseQuinticTrajectory] = None
        self.optimization_reports = []
        self.timestamp = time.time()
    
    def reset(self):
        """Reset planning state."""
        self.start = None
        self.goal = None
        self.raw_paths = []
        self.trajectory = None
        self.optimization_reports = []
        self.timestamp = time.time()

state = PlannerState()

# ============================================================================
# REST Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Serve main UI page."""
    return {"message": "EGO-Planner 2D Demo API", "version": "1.0.0"}


@app.get("/api/environment")
async def get_environment():
    """Get current environment map state."""
    return state.grid.export()


@app.post("/api/environment/reset")
async def reset_environment():
    """Clear all obstacles."""
    state.grid.clear()
    state.reset()
    return {"status": "success", "message": "Environment cleared"}


@app.post("/api/environment/add-obstacle")
async def add_obstacle(x_min: float, y_min: float, x_max: float, y_max: float):
    """Add rectangular obstacle to map."""
    try:
        state.grid.set_obstacle_rect(x_min, y_min, x_max, y_max)
        return {
            "status": "success",
            "occupied_cells": state.grid.occupied_cells,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/planning/set-start")
async def set_start(x: float, y: float):
    """Set start position."""
    if not state.grid.in_bounds_world((x, y)):
        return {"status": "error", "message": "Start position outside map bounds"}
    
    if state.grid.is_occupied((x, y)):
        return {"status": "error", "message": "Start position in obstacle"}
    
    state.start = (float(x), float(y))
    return {"status": "success", "start": list(state.start)}


@app.post("/api/planning/set-goal")
async def set_goal(x: float, y: float):
    """Set goal position."""
    if not state.grid.in_bounds_world((x, y)):
        return {"status": "error", "message": "Goal position outside map bounds"}
    
    if state.grid.is_occupied((x, y)):
        return {"status": "error", "message": "Goal position in obstacle"}
    
    state.goal = (float(x), float(y))
    return {"status": "success", "goal": list(state.goal)}


@app.post("/api/planning/search")
async def run_search():
    """Run A* topology search."""
    if state.start is None or state.goal is None:
        return {"status": "error", "message": "Start or goal not set"}
    
    t0 = time.perf_counter()
    
    # Run A* search
    planner = AStarPlanner(state.grid)
    paths = planner.search_multiple_paths(state.start, state.goal, num_paths=3)
    
    if not paths:
        return {"status": "error", "message": "No path found"}
    
    # Store raw paths
    state.raw_paths = [
        simplify_waypoints(path.waypoints, merge_distance=0.2)
        for path in paths
    ]
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    return {
        "status": "success",
        "num_paths": len(state.raw_paths),
        "paths": [
            {
                "waypoints": path,
                "length": sum(
                    np.hypot(path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
                    for i in range(1, len(path))
                ),
            }
            for path in state.raw_paths
        ],
        "search_time_ms": elapsed_ms,
        "cells_expanded": planner.cells_expanded,
    }


@app.post("/api/planning/initialize-trajectory")
async def initialize_trajectory(path_index: int = 0):
    """Generate initial trajectory from selected path."""
    if not state.raw_paths or path_index >= len(state.raw_paths):
        return {"status": "error", "message": "Invalid path index"}
    
    t0 = time.perf_counter()
    
    path = state.raw_paths[path_index]
    generator = TrajectoryGenerator()
    state.trajectory = generator.generate(path)
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    trajectory_dict = state.trajectory.to_dict()
    trajectory_dict["initialization_time_ms"] = elapsed_ms
    
    return {"status": "success", "trajectory": trajectory_dict}


@app.post("/api/planning/optimize")
async def optimize_trajectory():
    """Run elastic optimization on current trajectory."""
    if state.trajectory is None or state.goal is None:
        return {"status": "error", "message": "Trajectory not initialized"}
    
    t0 = time.perf_counter()
    
    optimizer = TrajectoryOptimizer(state.grid)
    optimized_traj, reports = optimizer.optimize(state.trajectory, state.goal)
    
    state.trajectory = optimized_traj
    state.optimization_reports = reports
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    # Prepare response
    final_report = reports[-1] if reports else None
    
    return {
        "status": "success",
        "num_iterations": len(reports),
        "optimization_time_ms": elapsed_ms,
        "final_report": {
            "cost_total": float(final_report.cost_total),
            "cost_smooth": float(final_report.cost_smooth),
            "cost_obstacle": float(final_report.cost_obstacle),
            "cost_dynamics": float(final_report.cost_dynamics),
            "cost_endpoint": float(final_report.cost_endpoint),
            "max_speed": float(final_report.max_speed),
            "max_acceleration": float(final_report.max_acceleration),
            "velocity_feasible": final_report.velocity_feasible,
            "acceleration_feasible": final_report.acceleration_feasible,
            "collision_free": final_report.collision_free,
        } if final_report else None,
        "trajectory": state.trajectory.to_dict(),
    }


@app.get("/api/planning/trajectory")
async def get_trajectory():
    """Get current trajectory."""
    if state.trajectory is None:
        return {"status": "error", "message": "No trajectory"}
    
    return {
        "status": "success",
        "trajectory": state.trajectory.to_dict(),
    }


@app.get("/api/planning/optimization-history")
async def get_optimization_history():
    """Get optimization iteration history."""
    return {
        "status": "success",
        "iterations": len(state.optimization_reports),
        "reports": [
            {
                "iteration": report.iteration,
                "cost_total": float(report.cost_total),
                "cost_smooth": float(report.cost_smooth),
                "cost_obstacle": float(report.cost_obstacle),
                "cost_dynamics": float(report.cost_dynamics),
                "cost_endpoint": float(report.cost_endpoint),
                "gradient_norm": float(report.gradient_norm),
                "max_speed": float(report.max_speed),
                "max_acceleration": float(report.max_acceleration),
                "velocity_feasible": report.velocity_feasible,
                "acceleration_feasible": report.acceleration_feasible,
                "collision_free": report.collision_free,
                "elapsed_ms": float(report.elapsed_ms),
            }
            for report in state.optimization_reports
        ],
    }


@app.post("/api/planning/reset")
async def reset_planning():
    """Reset planning state (keep map)."""
    state.reset()
    return {"status": "success", "message": "Planning state reset"}


# ============================================================================
# WebSocket for real-time updates
# ============================================================================

class ConnectionManager:
    """Manage WebSocket connections."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            
            # Handle incoming messages
            message_type = data.get("type")
            
            if message_type == "ping":
                await websocket.send_json({"type": "pong"})
            
            elif message_type == "get-state":
                await websocket.send_json({
                    "type": "state-update",
                    "environment": state.grid.export(),
                    "start": list(state.start) if state.start else None,
                    "goal": list(state.goal) if state.goal else None,
                    "trajectory": state.trajectory.to_dict() if state.trajectory else None,
                })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        manager.disconnect(websocket)


# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize demo obstacles on startup."""
    # Add some default obstacles for demo
    state.grid.set_obstacle_rect(2.0, 2.0, 3.0, 4.0)  # Vertical obstacle
    state.grid.set_obstacle_rect(6.0, 5.0, 8.0, 6.5)  # Horizontal obstacle
    print("✓ EGO-Planner 2D Demo initialized")
    print(f"  Map: {MapConfig.MAP_WIDTH}m × {MapConfig.MAP_HEIGHT}m")
    print(f"  Resolution: {MapConfig.GRID_RESOLUTION}m/cell")
    print(f"  Safety distance: {ObstacleConfig.SAFETY_DISTANCE}m")


@app.on_event("shutdown")
async def shutdown_event():
    print("✓ EGO-Planner 2D Demo shutdown")


# ============================================================================
# Main entry point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host=ServerConfig.HOST,
        port=ServerConfig.PORT,
        log_level="info",
    )
