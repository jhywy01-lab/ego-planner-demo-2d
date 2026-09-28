# EGO-Planner 2D Demo

This project is a 2D interactive visualization of the EGO-Planner planning logic and the ESDF-free strategy it uses for local obstacle avoidance.

## Overview

This demo is intentionally implemented as a simplified educational 2D version of the original planner. The system follows the same design philosophy:

- No global ESDF is precomputed.
- The planner keeps a local occupancy grid.
- Obstacle avoidance is computed on demand for trajectory samples.
- A trajectory optimizer minimizes a combined cost function composed of:
  - smoothness,
  - obstacle avoidance,
  - dynamics feasibility,
  - endpoint constraints.

## Architecture

- Backend: Python + FastAPI
- Frontend: HTML + CSS + JavaScript
- Planning modules:
  - `backend/modules/environment.py`
  - `backend/modules/search.py`
  - `backend/modules/trajectory.py`
  - `backend/modules/optimization.py`

## Coordinate system and map parameters

- Map size: 10m × 10m
- Resolution: 0.1m per cell
- Safety distance: 0.5m
- Max speed: 3.0m/s
- Max acceleration: 2.0m/s²

## Core obstacle cost function

```
J_obs = ∫_0^T f_obs(d(x(t))) dt

f_obs(d) = 0.5 * (d - r)^2,  if d < r
           0,               otherwise
```

This is the exact EGO-Planner-style local penalty used in the environment module.

## Quick start

### 1. Install backend dependencies

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start the backend

```bash
python app.py
```

The API will run at:

- http://localhost:8000

### 3. Start the frontend

```bash
cd ../frontend
python -m http.server 8080
```

Open:

- http://localhost:8080

## API endpoints

- `GET /` - service status
- `GET /api/environment` - current occupancy grid
- `POST /api/environment/reset` - reset map
- `POST /api/environment/add-obstacle` - add a rectangle obstacle
- `POST /api/planning/set-start` - set the start point
- `POST /api/planning/set-goal` - set the goal point
- `POST /api/planning/search` - run A* planning
- `POST /api/planning/initialize-trajectory` - initialize a polynomial trajectory
- `POST /api/planning/optimize` - optimize the trajectory
- `GET /api/planning/trajectory` - return current trajectory
- `GET /api/planning/optimization-history` - optimization metrics

## Notes

This project is designed as a teaching/demo platform and is not the full production ROS planner from the original EGO-Planner project. It retains the same core ideas and local optimization pattern, while making the behavior easy to inspect in a browser.

## License

GNU General Public License v3.0
