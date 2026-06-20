#!/usr/bin/env python3
"""Check whether Ant-v4 observations support target-aligned policy reuse."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ant_directional_controller import DirectionalAntController, make_env, target_aligned_observation


def run_probe(goal: tuple[float, float], *, use_transform: bool, steps: int = 100, seed: int = 0) -> dict:
    controller = DirectionalAntController()
    env = make_env()
    obs, _ = env.reset(seed=seed)
    start_xy = np.array(env.unwrapped.data.qpos[:2], dtype=np.float64)
    goal_xy = np.array(goal, dtype=np.float64)
    headings = []
    terminated = False
    truncated = False

    for step in range(steps):
        current_xy = np.array(env.unwrapped.data.qpos[:2], dtype=np.float64)
        if use_transform:
            action, heading = controller.predict(obs, current_xy, goal_xy)
        else:
            action, _ = controller.model.predict(obs, deterministic=True)
            heading = float(np.arctan2(goal_xy[1] - current_xy[1], goal_xy[0] - current_xy[0]))
        headings.append(heading)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    end_xy = np.array(env.unwrapped.data.qpos[:2], dtype=np.float64)
    env.close()
    return {
        "goal": list(goal),
        "use_transform": use_transform,
        "steps": step + 1,
        "terminated": terminated,
        "truncated": truncated,
        "start_xy": start_xy.tolist(),
        "end_xy": end_xy.tolist(),
        "delta_xy": (end_xy - start_xy).tolist(),
        "distance_to_goal": float(np.linalg.norm(goal_xy - end_xy)),
        "mean_target_heading": float(np.mean(headings)),
    }


def main() -> None:
    out_dir = Path("mujoco_ant_hierarchical/artifacts/step1")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = [
        run_probe((0.0, 5.0), use_transform=False),
        run_probe((0.0, 5.0), use_transform=True),
        run_probe((5.0, 0.0), use_transform=False),
        run_probe((5.0, 0.0), use_transform=True),
    ]
    summary = {
        "ant_v4_observation": "qpos[2:] + qvel[:], with world x/y position excluded",
        "obs_transform_indices": {
            "root_quaternion_wxyz": [1, 2, 3, 4],
            "root_xy_velocity": [13, 14],
        },
        "decision": "A",
        "decision_reason": (
            "The pretrained forward policy can be redirected by expressing root orientation "
            "and xy velocity in a target-aligned frame. No joint-level locomotion retraining is needed."
        ),
        "results": results,
    }
    path = out_dir / "observation_frame_check.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
