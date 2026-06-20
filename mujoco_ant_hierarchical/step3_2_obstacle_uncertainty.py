#!/usr/bin/env python3
"""Step 3-2 obstacle and progress-uncertainty check for hierarchical Ant."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib.pyplot as plt
import numpy as np

from ant_directional_controller import DirectionalAntController, DirectionalControllerConfig
from ant_scene import DEFAULT_GOAL_XY, DEFAULT_OBSTACLE_POS, DEFAULT_OBSTACLE_SIZE, make_scene_env


def progress_uncertainty(distances: list[float], *, window: int, min_progress: float) -> list[float]:
    values = []
    for idx, distance in enumerate(distances):
        if idx < window:
            values.append(0.0)
            continue
        progress = distances[idx - window] - distance
        values.append(float(max(0.0, min_progress - progress)))
    return values


def run_episode(
    controller: DirectionalAntController,
    *,
    with_obstacle: bool,
    seed: int,
    max_steps: int,
    success_radius: float,
    uncertainty_window: int,
    min_progress: float,
) -> dict:
    env = make_scene_env(with_obstacle=with_obstacle)
    obs, _ = env.reset(seed=seed)
    goal_xy = np.array(DEFAULT_GOAL_XY, dtype=np.float64)
    trajectory = []
    distances = []
    success = False
    terminated = False
    truncated = False

    for step in range(max_steps):
        current_xy = np.array(env.unwrapped.data.qpos[:2], dtype=np.float64)
        distance = float(np.linalg.norm(goal_xy - current_xy))
        trajectory.append(current_xy.tolist())
        distances.append(distance)
        if distance < success_radius:
            success = True
            break
        action, _, _ = controller.predict_with_info(obs, current_xy, goal_xy)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    env.close()
    uncertainty = progress_uncertainty(distances, window=uncertainty_window, min_progress=min_progress)
    return {
        "seed": seed,
        "with_obstacle": with_obstacle,
        "success": success,
        "steps": step + 1,
        "terminated": terminated,
        "truncated": truncated,
        "final_xy": trajectory[-1],
        "final_distance": distances[-1],
        "trajectory": trajectory,
        "distances": distances,
        "uncertainty": uncertainty,
        "max_uncertainty": float(max(uncertainty)) if uncertainty else 0.0,
        "mean_uncertainty": float(np.mean(uncertainty)) if uncertainty else 0.0,
    }


def add_scene_annotations(ax) -> None:
    ox, oy, _ = DEFAULT_OBSTACLE_POS
    sx, sy, _ = DEFAULT_OBSTACLE_SIZE
    ax.add_patch(
        plt.Rectangle(
            (ox - sx, oy - sy),
            sx * 2,
            sy * 2,
            color="red",
            alpha=0.25,
            label="obstacle",
        )
    )
    ax.scatter([DEFAULT_GOAL_XY[0]], [DEFAULT_GOAL_XY[1]], marker="*", s=95, c="green", label="goal")
    ax.scatter([0], [0], c="black", s=40, label="start")


def plot_trajectories(results_by_scenario: dict[str, list[dict]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, scenario in zip(axes, ["no_obstacle", "obstacle"]):
        for result in results_by_scenario[scenario]:
            trajectory = np.array(result["trajectory"], dtype=np.float64)
            ax.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.75, linewidth=1.4)
        add_scene_annotations(ax)
        ax.set_title(scenario)
        ax.set_xlabel("x [m]")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("y [m]")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles[:3], labels[:3], loc="upper center", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_uncertainty(results_by_scenario: dict[str, list[dict]], threshold: float, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, scenario in zip(axes, ["no_obstacle", "obstacle"]):
        for result in results_by_scenario[scenario]:
            ax.plot(result["uncertainty"], alpha=0.75, linewidth=1.2)
        ax.axhline(threshold, color="red", linestyle="--", linewidth=1.2, label="90th percentile threshold")
        ax.set_title(scenario)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("progress uncertainty")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def summarize(results: list[dict], threshold: float) -> dict:
    all_unc = [value for result in results for value in result["uncertainty"]]
    above = [value for value in all_unc if value >= threshold and value > 0.0]
    return {
        "episodes": len(results),
        "success_rate": sum(result["success"] for result in results) / len(results),
        "termination_rate": sum(result["terminated"] for result in results) / len(results),
        "mean_final_distance": float(np.mean([result["final_distance"] for result in results])),
        "mean_uncertainty": float(np.mean(all_unc)) if all_unc else 0.0,
        "p90_uncertainty": float(np.percentile(all_unc, 90)) if all_unc else 0.0,
        "max_uncertainty": float(max(all_unc)) if all_unc else 0.0,
        "fraction_above_threshold": len(above) / len(all_unc) if all_unc else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=220)
    parser.add_argument("--success_radius", type=float, default=0.75)
    parser.add_argument("--slow_radius", type=float, default=7.0)
    parser.add_argument("--min_action_scale", type=float, default=0.3)
    parser.add_argument("--uncertainty_window", type=int, default=5)
    parser.add_argument("--min_progress", type=float, default=0.02)
    parser.add_argument("--threshold_percentile", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--artifact_dir", type=Path, default=Path("mujoco_ant_hierarchical/artifacts/step3_2_uncertainty"))
    args = parser.parse_args()

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    controller = DirectionalAntController(
        config=DirectionalControllerConfig(
            success_radius=args.success_radius,
            slow_radius=args.slow_radius,
            min_action_scale=args.min_action_scale,
        )
    )

    results_by_scenario = {"no_obstacle": [], "obstacle": []}
    for scenario, with_obstacle in [("no_obstacle", False), ("obstacle", True)]:
        for episode in range(args.episodes):
            results_by_scenario[scenario].append(
                run_episode(
                    controller,
                    with_obstacle=with_obstacle,
                    seed=args.seed + episode,
                    max_steps=args.max_steps,
                    success_radius=args.success_radius,
                    uncertainty_window=args.uncertainty_window,
                    min_progress=args.min_progress,
                )
            )

    no_obstacle_uncertainty = [
        value for result in results_by_scenario["no_obstacle"] for value in result["uncertainty"]
    ]
    threshold = float(np.percentile(no_obstacle_uncertainty, args.threshold_percentile))
    summary = {
        "goal_xy": list(DEFAULT_GOAL_XY),
        "obstacle_pos": list(DEFAULT_OBSTACLE_POS),
        "obstacle_size": list(DEFAULT_OBSTACLE_SIZE),
        "episodes_per_scenario": args.episodes,
        "max_steps": args.max_steps,
        "uncertainty_formula": "max(0, min_progress - (distance[t-window] - distance[t]))",
        "uncertainty_window": args.uncertainty_window,
        "min_progress": args.min_progress,
        "threshold_source": "no_obstacle",
        "threshold_percentile": args.threshold_percentile,
        "threshold": threshold,
        "scenario_summary": {
            scenario: summarize(results, threshold) for scenario, results in results_by_scenario.items()
        },
        "results": results_by_scenario,
    }
    (args.artifact_dir / "obstacle_uncertainty_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_trajectories(results_by_scenario, args.artifact_dir / "obstacle_uncertainty_trajectories.png")
    plot_uncertainty(results_by_scenario, threshold, args.artifact_dir / "obstacle_uncertainty_traces.png")
    print(json.dumps(summary["scenario_summary"], indent=2))
    print(json.dumps({"threshold": threshold}, indent=2))


if __name__ == "__main__":
    main()
