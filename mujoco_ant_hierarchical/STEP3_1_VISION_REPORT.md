# Hierarchical MuJoCo Ant Step 3-1 Vision Report

Date: 2026-06-20

## Goal

Verify real MuJoCo EGL rendering for the hierarchical Ant scene before using camera frames for LLM waypoint planning.

## Scene

The script generates a custom `Ant-v4` XML at runtime with:

- static red obstacle at `(2.5, 0.0, 0.45)`
- green goal marker at `(5.0, 0.0)`
- fixed `overhead` camera
- Ant torso-mounted `ego` camera looking along local +x

This is not a matplotlib mock. Images are rendered through MuJoCo EGL via `render_mode="rgb_array"`.

## Result

Command:

```bash
source env_isaaclab/bin/activate
python mujoco_ant_hierarchical/step3_1_egl_vision_smoke.py
```

| Camera | Shape | Nonblank | Saved image |
| --- | --- | --- | --- |
| overhead | `(480, 640, 3)` | yes | `artifacts/step3_1_vision/overhead_obstacle_goal.png` |
| ego | `(480, 640, 3)` | yes | `artifacts/step3_1_vision/egocentric_obstacle_goal.png` |

10 consecutive EGL renders:

| Camera | Frames | Result |
| --- | ---: | --- |
| overhead | 10 | all nonblank |
| ego | 10 | all nonblank |

## Artifacts

- `mujoco_ant_hierarchical/artifacts/step3_1_vision/egl_vision_smoke_summary.json`
- `mujoco_ant_hierarchical/artifacts/step3_1_vision/overhead_obstacle_goal.png`
- `mujoco_ant_hierarchical/artifacts/step3_1_vision/egocentric_obstacle_goal.png`

## Decision

Step 3-1 passes. Real EGL-rendered overhead and egocentric camera frames are available for the next uncertainty and LLM stages.
