#!/usr/bin/env python3
"""Evaluate no-learning overshoot refinements for hierarchical Ant."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ant_directional_controller import DirectionalAntController, DirectionalControllerConfig, make_env
from step2_directional_smoke import GOALS


def run_episode(
    controller: DirectionalAntController,
    *,
    goal: tuple[float, float],
    seed: int,
    max_steps: int,
    success_radius: float,
) -> dict:
    env = make_env()
    obs, _ = env.reset(seed=seed)
    goal_xy = np.array(goal, dtype=np.float64)
    trajectory = [np.array(env.unwrapped.data.qpos[:2], dtype=np.float64).tolist()]
    action_scales = []
    success = False
    terminated = False
    truncated = False

    for step in range(max_steps):
        current_xy = np.array(env.unwrapped.data.qpos[:2], dtype=np.float64)
        distance = float(np.linalg.norm(goal_xy - current_xy))
        if distance < success_radius:
            success = True
            break
        action, _, info = controller.predict_with_info(obs, current_xy, goal_xy)
        action_scales.append(float(info["action_scale"]))
        obs, _, terminated, truncated, _ = env.step(action)
        trajectory.append(np.array(env.unwrapped.data.qpos[:2], dtype=np.float64).tolist())
        if terminated or truncated:
            break

    final_xy = np.array(env.unwrapped.data.qpos[:2], dtype=np.float64)
    env.close()
    return {
        "goal_xy": list(goal),
        "seed": seed,
        "success": success,
        "steps": step + 1,
        "terminated": terminated,
        "truncated": truncated,
        "final_xy": final_xy.tolist(),
        "final_distance": float(np.linalg.norm(goal_xy - final_xy)),
        "mean_action_scale": float(np.mean(action_scales)) if action_scales else 0.0,
        "min_action_scale_seen": float(np.min(action_scales)) if action_scales else 0.0,
        "trajectory": trajectory,
    }


def plot_results(results: list[dict], path: Path, success_radius: float, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    for result in results:
        trajectory = np.array(result["trajectory"], dtype=np.float64)
        plt.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.8, linewidth=1.8)
        color = "green" if result["success"] else "red"
        plt.scatter([result["goal_xy"][0]], [result["goal_xy"][1]], marker="*", s=75, c=color)
        circle = plt.Circle(result["goal_xy"], success_radius, color=color, fill=False, alpha=0.25)
        plt.gca().add_patch(circle)
    plt.scatter([0.0], [0.0], c="black", s=45, label="nominal start")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--success_radius", type=float, default=0.75)
    parser.add_argument("--slow_radius", type=float, default=7.0)
    parser.add_argument("--min_action_scale", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--artifact_dir", type=Path, default=Path("mujoco_ant_hierarchical/artifacts/step2_refinement"))
    args = parser.parse_args()

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    config = DirectionalControllerConfig(
        success_radius=args.success_radius,
        slow_radius=args.slow_radius,
        min_action_scale=args.min_action_scale,
    )
    controller = DirectionalAntController(config=config)
    results = []
    for episode in range(args.episodes):
        results.append(
            run_episode(
                controller,
                goal=GOALS[episode % len(GOALS)],
                seed=args.seed + episode,
                max_steps=args.max_steps,
                success_radius=args.success_radius,
            )
        )

    success_rate = sum(result["success"] for result in results) / len(results)
    summary = {
        "controller": "target_aligned_observation_with_distance_action_scale",
        "low_level_policy": "jren123/sac-ant-v4",
        "training_timesteps": 0,
        "success_radius": args.success_radius,
        "slow_radius": args.slow_radius,
        "min_action_scale": args.min_action_scale,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "success_rate": success_rate,
        "mean_final_distance": float(np.mean([result["final_distance"] for result in results])),
        "median_final_distance": float(np.median([result["final_distance"] for result in results])),
        "results": results,
    }
    (args.artifact_dir / "refinement_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_results(
        results,
        args.artifact_dir / "refinement_trajectories.png",
        args.success_radius,
        "Frozen Ant + distance action scaling",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
