"""Topology-guided A* pathfinding with multi-homotopy path generation.

This module implements A* search in a 2D grid, supporting generation of multiple
topologically distinct paths. The implementation is inspired by EGO-Planner's
approach to finding diverse candidate paths for trajectory optimization.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from enum import Enum
from math import hypot, sqrt
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from config import SearchConfig, MapConfig, ObstacleConfig
from modules.environment import LocalOccupancyGrid

Point2D = Tuple[float, float]
GridIndex = Tuple[int, int]


class HeuristicType(Enum):
    """Heuristic function types for A* search."""
    MANHATTAN = "manhattan"
    EUCLIDEAN = "euclidean"


@dataclass
class PathInfo:
    """Information about a found path."""
    waypoints: List[Point2D]  # World coordinates
    length: float
    cells_expanded: int
    search_time_ms: float
    homotopy_class: Optional[int] = None  # ID for topological class
    

@dataclass(order=True)
class Node:
    """A* search node with priority queue support."""
    f_score: float
    g_score: float = field(compare=False)
    h_score: float = field(compare=False)
    grid_pos: GridIndex = field(compare=False)
    parent: Optional[GridIndex] = field(default=None, compare=False)


class AStarPlanner:
    """A* pathfinding with support for multi-homotopy path generation."""
    
    def __init__(
        self,
        occupancy_grid: LocalOccupancyGrid,
        heuristic: str = SearchConfig.HEURISTIC,
        heuristic_weight: float = SearchConfig.HEURISTIC_WEIGHT,
    ):
        """Initialize A* planner.
        
        Args:
            occupancy_grid: The environment map
            heuristic: 'manhattan' or 'euclidean'
            heuristic_weight: Multiplier for heuristic (1.0 = optimal, >1.0 = faster but suboptimal)
        """
        self.grid = occupancy_grid
        self.heuristic_type = HeuristicType(heuristic)
        self.heuristic_weight = float(heuristic_weight)
        
        # Movement directions: 4-connectivity (no diagonals for simplicity)
        self.directions = [
            (1, 0), (-1, 0),   # East, West
            (0, 1), (0, -1),   # North, South
        ]
        
        # Statistics
        self.cells_expanded = 0
        self.search_time_ms = 0.0
    
    def heuristic(self, pos: GridIndex, goal: GridIndex) -> float:
        """Compute heuristic distance from pos to goal."""
        dx = abs(pos[0] - goal[0])
        dy = abs(pos[1] - goal[1])
        
        if self.heuristic_type == HeuristicType.MANHATTAN:
            return self.heuristic_weight * (dx + dy) * self.grid.resolution
        else:  # EUCLIDEAN
            return self.heuristic_weight * sqrt(dx**2 + dy**2) * self.grid.resolution
    
    def get_neighbors(self, pos: GridIndex) -> List[GridIndex]:
        """Get valid neighboring cells."""
        neighbors = []
        for dx, dy in self.directions:
            nx, ny = pos[0] + dx, pos[1] + dy
            if self.grid.in_bounds_grid((nx, ny)):
                # Cell is not occupied
                if not self.grid.occupancy[ny, nx]:
                    neighbors.append((nx, ny))
        return neighbors
    
    def reconstruct_path(
        self,
        came_from: Dict[GridIndex, GridIndex],
        current: GridIndex,
    ) -> List[GridIndex]:
        """Reconstruct path from start to goal via came_from map."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
    
    def search(
        self,
        start: Point2D,
        goal: Point2D,
    ) -> Optional[PathInfo]:
        """Run A* search from start to goal.
        
        Args:
            start: Start position in world coordinates
            goal: Goal position in world coordinates
        
        Returns:
            PathInfo if path found, None otherwise
        """
        import time
        start_time = time.perf_counter()
        
        # Convert to grid coordinates
        start_grid = self.grid.world_to_grid(start)
        goal_grid = self.grid.world_to_grid(goal)
        
        # Check if start/goal are in free space
        if self.grid.occupancy[start_grid[1], start_grid[0]]:
            return None  # Start in obstacle
        if self.grid.occupancy[goal_grid[1], goal_grid[0]]:
            return None  # Goal in obstacle
        
        # Initialize A* data structures
        open_set = []
        came_from: Dict[GridIndex, GridIndex] = {}
        g_score: Dict[GridIndex, float] = {start_grid: 0.0}
        f_score: Dict[GridIndex, float] = {
            start_grid: self.heuristic(start_grid, goal_grid)
        }
        
        # Add start to open set
        start_node = Node(
            f_score=f_score[start_grid],
            g_score=0.0,
            h_score=self.heuristic(start_grid, goal_grid),
            grid_pos=start_grid,
        )
        heapq.heappush(open_set, start_node)
        open_set_dict = {start_grid}  # For O(1) membership checking
        self.cells_expanded = 0
        
        while open_set:
            current_node = heapq.heappop(open_set)
            current = current_node.grid_pos
            open_set_dict.discard(current)
            
            self.cells_expanded += 1
            
            # Goal check
            if current == goal_grid:
                path_grid = self.reconstruct_path(came_from, current)
                path_world = [self.grid.grid_to_world(p) for p in path_grid]
                length = sum(
                    hypot(path_world[i][0] - path_world[i-1][0],
                          path_world[i][1] - path_world[i-1][1])
                    for i in range(1, len(path_world))
                )
                
                self.search_time_ms = (time.perf_counter() - start_time) * 1000
                
                return PathInfo(
                    waypoints=path_world,
                    length=length,
                    cells_expanded=self.cells_expanded,
                    search_time_ms=self.search_time_ms,
                )
            
            # Explore neighbors
            for neighbor in self.get_neighbors(current):
                tentative_g = g_score[current] + self.grid.resolution
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h = self.heuristic(neighbor, goal_grid)
                    f_score[neighbor] = tentative_g + h
                    
                    if neighbor not in open_set_dict:
                        node = Node(
                            f_score=f_score[neighbor],
                            g_score=tentative_g,
                            h_score=h,
                            grid_pos=neighbor,
                        )
                        heapq.heappush(open_set, node)
                        open_set_dict.add(neighbor)
        
        self.search_time_ms = (time.perf_counter() - start_time) * 1000
        return None  # No path found
    
    def search_multiple_paths(
        self,
        start: Point2D,
        goal: Point2D,
        num_paths: int = SearchConfig.NUM_CANDIDATE_PATHS,
    ) -> List[PathInfo]:
        """Generate multiple topologically distinct paths.
        
        Uses an iterative approach: find path, block some cells, find next path.
        This provides diverse candidate paths for trajectory optimization.
        
        Args:
            start: Start position
            goal: Goal position
            num_paths: Number of distinct paths to generate
        
        Returns:
            List of PathInfo objects
        """
        paths = []
        blocked_cells: Set[GridIndex] = set()
        
        for _ in range(num_paths):
            # Temporarily block cells from previous paths (except start/goal)
            for cell in blocked_cells:
                self.grid.set_occupied(cell, True)
            
            # Search for path
            path_info = self.search(start, goal)
            
            # Restore blocked cells
            for cell in blocked_cells:
                self.grid.set_occupied(cell, False)
            
            if path_info is None:
                break
            
            paths.append(path_info)
            
            # Block middle cells of this path for diversity
            waypoints_grid = [self.grid.world_to_grid(wp) for wp in path_info.waypoints]
            if len(waypoints_grid) > 2:
                # Block every other cell in the middle section
                mid_start = len(waypoints_grid) // 3
                mid_end = 2 * len(waypoints_grid) // 3
                for i in range(mid_start, mid_end, 2):
                    cell = waypoints_grid[i]
                    if cell != self.grid.world_to_grid(start) and \
                       cell != self.grid.world_to_grid(goal):
                        blocked_cells.add(cell)
        
        return paths


def simplify_waypoints(
    waypoints: List[Point2D],
    merge_distance: float = 0.2,
) -> List[Point2D]:
    """Simplify path by merging nearby waypoints.
    
    Args:
        waypoints: Original waypoints
        merge_distance: Threshold for merging adjacent waypoints
    
    Returns:
        Simplified waypoint list
    """
    if len(waypoints) <= 2:
        return waypoints
    
    simplified = [waypoints[0]]
    
    for wp in waypoints[1:-1]:
        last = simplified[-1]
        dist = hypot(wp[0] - last[0], wp[1] - last[1])
        if dist >= merge_distance:
            simplified.append(wp)
    
    simplified.append(waypoints[-1])
    return simplified


def smooth_waypoints(
    waypoints: List[Point2D],
    grid: LocalOccupancyGrid,
    safety_radius: float = ObstacleConfig.SAFETY_DISTANCE,
    iterations: int = 5,
) -> List[Point2D]:
    """Smooth path using laplacian smoothing while maintaining collision-free constraint.
    
    Args:
        waypoints: Original waypoints
        grid: Occupancy grid for collision checking
        safety_radius: Safety distance to obstacles
        iterations: Number of smoothing iterations
    
    Returns:
        Smoothed waypoint list
    """
    if len(waypoints) <= 2:
        return waypoints
    
    smoothed = [list(wp) for wp in waypoints]  # Convert to lists for in-place modification
    
    for _ in range(iterations):
        for i in range(1, len(smoothed) - 1):
            # Laplacian: move towards average of neighbors
            new_x = (smoothed[i-1][0] + smoothed[i+1][0]) / 2.0
            new_y = (smoothed[i-1][1] + smoothed[i+1][1]) / 2.0
            
            # Check if new position is collision-free
            if not grid.is_occupied((new_x, new_y)):
                penalty = grid.obstacle_penalty((new_x, new_y), safety_radius)
                # Only update if no collision risk
                if not penalty.active:
                    smoothed[i] = [new_x, new_y]
    
    return [tuple(wp) for wp in smoothed]


__all__ = [
    "AStarPlanner",
    "PathInfo",
    "HeuristicType",
    "simplify_waypoints",
    "smooth_waypoints",
]
