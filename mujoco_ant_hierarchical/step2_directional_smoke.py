#!/usr/bin/env python3
"""Evaluate target-aligned frozen Ant locomotion on random open-field goals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ant_directional_controller import DirectionalAntController, make_env


GOALS = [
    (5.0, 0.0),
    (0.0, 5.0),
    (5.0, 5.0),
    (-3.0, 4.0),
    (4.0, -3.0),
]


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
    heading_trace = []
    success = False
    terminated = False
    truncated = False

    for step in range(max_steps):
        current_xy = np.array(env.unwrapped.data.qpos[:2], dtype=np.float64)
        distance = float(np.linalg.norm(goal_xy - current_xy))
        if distance < success_radius:
            success = True
            break
        action, target_heading = controller.predict(obs, current_xy, goal_xy)
        heading_trace.append(target_heading)
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
        "trajectory": trajectory,
        "mean_target_heading": float(np.mean(heading_trace)) if heading_trace else 0.0,
    }


def plot_results(results: list[dict], path: Path, success_radius: float) -> None:
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
    plt.title("Frozen Ant + target-aligned observation")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--success_radius", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--artifact_dir", type=Path, default=Path("mujoco_ant_hierarchical/artifacts/step2"))
    args = parser.parse_args()

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    controller = DirectionalAntController()
    results = []
    for episode in range(args.episodes):
        goal = GOALS[episode % len(GOALS)]
        results.append(
            run_episode(
                controller,
                goal=goal,
                seed=args.seed + episode,
                max_steps=args.max_steps,
                success_radius=args.success_radius,
            )
        )

    success_rate = sum(result["success"] for result in results) / len(results)
    summary = {
        "controller": "target_aligned_observation",
        "low_level_policy": "jren123/sac-ant-v4",
        "training_timesteps": 0,
        "training_note": "Way A requires no high-level RL training; the high-level controller is target-heading geometry.",
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "success_radius": args.success_radius,
        "success_rate": success_rate,
        "mean_final_distance": float(np.mean([result["final_distance"] for result in results])),
        "results": results,
    }
    (args.artifact_dir / "directional_smoke_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_results(results, args.artifact_dir / "directional_smoke_trajectories.png", args.success_radius)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
