# EGO-Planner 2D Interactive Demonstration System

## Overview

An **ESDF-free, gradient-based trajectory planning visualization system** that demonstrates the core algorithms of EGO-Planner in an interactive 2D web interface.

### Key Features

- **ESDF-Free Environment Sensing**: Local occupancy grid with on-demand obstacle collision detection
- **Topology-Guided Search**: A* pathfinding with multi-homotopy path generation
- **Trajectory Initialization**: Polynomial trajectory from B-splines with dynamic feasibility checks
- **Elastic Optimization**: L-BFGS-based trajectory optimization with obstacle avoidance penalty
- **Real-time Performance Monitoring**: Detailed computation time statistics and constraint violation tracking

## Architecture

```
┌─────────────────────────────────────┐
│  Frontend (Vue.js + Canvas/Chart.js)│
│  - 2D Map Visualization             │
│  - Interactive Controls              │
│  - Real-time Charts                  │
└──────────────┬──────────────────────┘
               │ WebSocket
┌──────────────▼──────────────────────┐
│  Backend (Python FastAPI)           │
│  - environment.py (Local Grid)      │
│  - search.py (A* Search)            │
│  - trajectory.py (Poly Traj)        │
│  - optimization.py (L-BFGS)         │
│  - performance.py (Timing)          │
└─────────────────────────────────────┘
```

## Quick Start

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Server runs on `http://localhost:8000`

### Frontend Setup

```bash
cd frontend
python -m http.server 8080
# Or use any HTTP server
```

Open `http://localhost:8080` in your browser.

## Project Structure

```
ego-planner-demo-2d/
├── backend/
│   ├── app.py                    # FastAPI main application
│   ├── requirements.txt           # Python dependencies
│   ├── config.py                 # Configuration (constants, parameters)
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── environment.py        # Local occupancy grid (ESDF-free)
│   │   ├── search.py             # A* topology search
│   │   ├── trajectory.py         # Polynomial trajectory generation
│   │   ├── optimization.py       # Elastic optimization (L-BFGS)
│   │   ├── execution.py          # Trajectory execution & replanning
│   │   └── utils.py              # Utility functions
│   └── services/
│       ├── __init__.py
│       └── performance.py        # Performance monitoring
├── frontend/
│   ├── index.html               # Main HTML
│   ├── css/
│   │   └── style.css            # Styling
│   ├── js/
│   │   ├── main.js              # Vue.js app
│   │   ├── canvas.js            # Canvas visualization
│   │   ├── charts.js            # Chart.js integration
│   │   └── websocket.js         # WebSocket communication
│   └── lib/
│       ├── vue.js
│       ├── chart.js
│       └── axios.js
├── .gitignore
└── README.md
```

## Technical Specifications

### Coordinate System
- **Map Size**: 10m × 10m
- **Grid Resolution**: 0.1m per cell (100 × 100 cells)
- **Safety Distance**: 0.5m (obstacle inflation)
- **Max Velocity**: 3.0 m/s
- **Max Acceleration**: 2.0 m/s²
- **Max Jerk**: 5.0 m/s³

### Algorithm Parameters

#### A* Search
- Heuristic: Manhattan distance
- Multi-path generation: up to 5 topologically distinct paths

#### Trajectory Optimization
- **Polynomial Degree**: 5 (quintic)
- **Cost Function Weights**:
  - Smoothness: w_smooth = 1.0
  - Obstacle Avoidance: w_obs = 2.0
  - Dynamic Feasibility: w_dyn = 0.5

#### L-BFGS Optimizer
- Learning Rate: 0.01-0.1
- Max Iterations: 50
- Convergence Threshold: 1e-4

### Obstacle Avoidance (ESDF-Free)

Based on EGO-Planner paper, the collision penalty function is:

```
J_obs = ∫₀ᵀ f_obs(d(x(t))) dt

where:
f_obs(d) = { 0.5 * (d - r)²  if d < r
           { 0              otherwise

d(x) = distance from point x to nearest obstacle
r = safety_margin (0.5m)
```

## Development Status

- [x] Project initialization
- [ ] Phase 1: Environment module (local grid + collision detection)
- [ ] Phase 2: Search module (A* pathfinding)
- [ ] Phase 3: Trajectory module (polynomial generation)
- [ ] Phase 4: Optimization module (L-BFGS + elastic optimization)
- [ ] Phase 5: Frontend UI (Canvas + controls)
- [ ] Phase 6: Integration & testing
- [ ] Phase 7: Performance optimization

## References

- **EGO-Planner Paper**: Zhou, X., Wang, Z., Xu, C., & Gao, F. (2021). EGO-Planner: An ESDF-free Gradient-based Local Planner for Quadrotors. RA-L.
- **arXiv**: https://arxiv.org/abs/2008.08835
- **GitHub**: https://github.com/ZJU-FAST-Lab/ego-planner

## License

GNU General Public License v3.0

## Author

Developed as an educational demonstration system for trajectory planning algorithms.
