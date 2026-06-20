#!/usr/bin/env python3
"""Measure progress and egocentric vision uncertainty signals without LLM calls."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image

from ant_directional_controller import DirectionalAntController, DirectionalControllerConfig
from ant_scene import DEFAULT_GOAL_XY, DEFAULT_OBSTACLE_POS, DEFAULT_OBSTACLE_SIZE, make_scene_env
from step3_2_obstacle_uncertainty import progress_uncertainty


def box_distance_xy(point_xy: np.ndarray) -> float:
    ox, oy, _ = DEFAULT_OBSTACLE_POS
    sx, sy, _ = DEFAULT_OBSTACLE_SIZE
    dx = max(abs(float(point_xy[0]) - ox) - sx, 0.0)
    dy = max(abs(float(point_xy[1]) - oy) - sy, 0.0)
    return float(np.hypot(dx, dy))


def center_crop(array: np.ndarray, *, width_frac: float, height_frac: float) -> np.ndarray:
    height, width = array.shape[:2]
    crop_w = int(width * width_frac)
    crop_h = int(height * height_frac)
    x0 = (width - crop_w) // 2
    y0 = (height - crop_h) // 2
    return array[y0 : y0 + crop_h, x0 : x0 + crop_w]


def occupancy_uncertainty(depth: np.ndarray, *, depth_threshold: float) -> tuple[float, float]:
    crop = center_crop(depth, width_frac=0.40, height_frac=0.50)
    finite = crop[np.isfinite(crop)]
    if finite.size == 0:
        return 0.0, float("nan")
    mean_depth = float(np.clip(finite, 0.0, depth_threshold * 4.0).mean())
    uncertainty = max(0.0, (depth_threshold - mean_depth) / depth_threshold)
    return float(uncertainty), mean_depth


def red_occlusion(rgb: np.ndarray) -> float:
    crop = center_crop(rgb, width_frac=0.60, height_frac=0.70)
    r = crop[:, :, 0].astype(np.int16)
    g = crop[:, :, 1].astype(np.int16)
    b = crop[:, :, 2].astype(np.int16)
    mask = (r > 120) & (r > g * 1.6) & (r > b * 1.6)
    return float(mask.mean())


def render_ego(renderer: mujoco.Renderer, env) -> tuple[np.ndarray, np.ndarray]:
    renderer.disable_depth_rendering()
    renderer.update_scene(env.unwrapped.data, camera="ego")
    rgb = renderer.render()
    renderer.enable_depth_rendering()
    renderer.update_scene(env.unwrapped.data, camera="ego")
    depth = renderer.render()
    return rgb, np.asarray(depth, dtype=np.float32)


def first_threshold_crossing(values: list[float], threshold: float) -> int | None:
    for index, value in enumerate(values):
        if value >= threshold:
            return index
    return None


def first_distance_crossing(values: list[float], threshold: float) -> int | None:
    for index, value in enumerate(values):
        if value <= threshold:
            return index
    return None


def run_episode(
    *,
    episode: int,
    controller: DirectionalAntController,
    max_steps: int,
    seed: int,
    depth_threshold: float,
    min_progress: float,
    progress_window: int,
    width: int,
    height: int,
    sample_image_steps: set[int],
    image_dir: Path,
) -> dict:
    env = make_scene_env(with_obstacle=True)
    obs, _ = env.reset(seed=seed)
    goal_xy = np.array(DEFAULT_GOAL_XY, dtype=np.float64)
    renderer = mujoco.Renderer(env.unwrapped.model, height=height, width=width)

    trajectory: list[list[float]] = []
    goal_distances: list[float] = []
    obstacle_distances: list[float] = []
    occupancy: list[float] = []
    center_depth: list[float] = []
    occlusion: list[float] = []
    progress_trace: list[float] = []
    success = False
    terminated = False
    truncated = False

    for step in range(max_steps):
        current_xy = np.array(env.unwrapped.data.qpos[:2], dtype=np.float64)
        trajectory.append(current_xy.tolist())
        goal_distance = float(np.linalg.norm(goal_xy - current_xy))
        goal_distances.append(goal_distance)
        obstacle_distances.append(box_distance_xy(current_xy))

        rgb, depth = render_ego(renderer, env)
        occ, mean_depth = occupancy_uncertainty(depth, depth_threshold=depth_threshold)
        occupancy.append(occ)
        center_depth.append(mean_depth)
        occlusion.append(red_occlusion(rgb))

        progress_trace = progress_uncertainty(goal_distances, window=progress_window, min_progress=min_progress)

        if step in sample_image_steps:
            step_dir = image_dir / f"episode_{episode:03d}_step_{step:03d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(step_dir / "ego_rgb.png")
            finite = depth[np.isfinite(depth)]
            if finite.size:
                lo, hi = np.percentile(finite, [1, 99])
                scaled = (255.0 * (np.clip(depth, lo, hi) - lo) / max(1e-6, hi - lo)).astype(np.uint8)
                Image.fromarray(scaled).save(step_dir / "ego_depth_scaled.png")

        if goal_distance < 0.75:
            success = True
            break

        action, _ = controller.predict(obs, current_xy, goal_xy)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    renderer.close()
    env.close()
    return {
        "episode": episode,
        "seed": seed,
        "success": success,
        "terminated": terminated,
        "truncated": truncated,
        "steps": len(trajectory),
        "trajectory": trajectory,
        "goal_distance": goal_distances,
        "obstacle_distance": obstacle_distances,
        "uncertainty_progress": progress_trace,
        "uncertainty_occupancy": occupancy,
        "center_depth": center_depth,
        "uncertainty_occlusion": occlusion,
    }


def plot_episode_signals(result: dict, path: Path, *, obstacle_near_threshold: float) -> None:
    steps = np.arange(result["steps"])
    obstacle_distance = np.array(result["obstacle_distance"])
    near_step = first_distance_crossing(result["obstacle_distance"], obstacle_near_threshold)

    plt.figure(figsize=(10, 4.8))
    plt.plot(steps, result["uncertainty_progress"], label="progress", linewidth=1.6)
    plt.plot(steps, result["uncertainty_occupancy"], label="depth occupancy", linewidth=1.6)
    plt.plot(steps, result["uncertainty_occlusion"], label="red occlusion", linewidth=1.6)
    plt.plot(steps, np.clip(1.0 - obstacle_distance / obstacle_near_threshold, 0.0, 1.0), "--", label="gt obstacle proximity")
    if near_step is not None:
        plt.axvline(near_step, color="black", linestyle=":", linewidth=1.2, label=f"obstacle <= {obstacle_near_threshold}m")
    plt.ylim(-0.02, 1.02)
    plt.xlabel("step")
    plt.ylabel("signal")
    plt.title(f"Episode {result['episode']:02d} uncertainty signals")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_scatter(results: list[dict], path: Path) -> None:
    obstacle = np.concatenate([np.array(r["obstacle_distance"]) for r in results])
    signals = {
        "progress": np.concatenate([np.array(r["uncertainty_progress"]) for r in results]),
        "occupancy": np.concatenate([np.array(r["uncertainty_occupancy"]) for r in results]),
        "occlusion": np.concatenate([np.array(r["uncertainty_occlusion"]) for r in results]),
    }
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True)
    for axis, (name, values) in zip(axes, signals.items()):
        axis.scatter(obstacle, values, s=6, alpha=0.28)
        axis.set_title(name)
        axis.set_xlabel("distance to obstacle box [m]")
        axis.set_ylim(-0.02, 1.02)
        axis.grid(True, alpha=0.3)
    axes[0].set_ylabel("uncertainty")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def summarize_timing(
    results: list[dict],
    *,
    obstacle_near_threshold: float,
    progress_threshold: float,
    occupancy_threshold: float,
    occlusion_threshold: float,
) -> dict:
    thresholds = {
        "progress": progress_threshold,
        "occupancy": occupancy_threshold,
        "occlusion": occlusion_threshold,
    }
    keys = {
        "progress": "uncertainty_progress",
        "occupancy": "uncertainty_occupancy",
        "occlusion": "uncertainty_occlusion",
    }
    timing = {}
    for name, key in keys.items():
        offsets = []
        false_positive_steps = 0
        total_far_steps = 0
        hits = 0
        for result in results:
            near_step = first_distance_crossing(result["obstacle_distance"], obstacle_near_threshold)
            trigger_step = first_threshold_crossing(result[key], thresholds[name])
            if near_step is not None and trigger_step is not None:
                offsets.append(trigger_step - near_step)
                hits += int(trigger_step <= near_step)
            obstacle_distance = np.array(result["obstacle_distance"])
            values = np.array(result[key])
            far_mask = obstacle_distance > obstacle_near_threshold
            false_positive_steps += int(((values >= thresholds[name]) & far_mask).sum())
            total_far_steps += int(far_mask.sum())
        timing[name] = {
            "threshold": thresholds[name],
            "offsets_trigger_minus_near_step": offsets,
            "mean_offset": None if not offsets else float(np.mean(offsets)),
            "median_offset": None if not offsets else float(np.median(offsets)),
            "early_or_on_time_hits": hits,
            "episodes_with_offset": len(offsets),
            "false_positive_rate_far_steps": 0.0 if total_far_steps == 0 else false_positive_steps / total_far_steps,
        }
    return timing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=220)
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--depth_threshold", type=float, default=3.0)
    parser.add_argument("--obstacle_near_threshold", type=float, default=1.0)
    parser.add_argument("--progress_threshold", type=float, default=0.02)
    parser.add_argument("--occupancy_threshold", type=float, default=0.15)
    parser.add_argument("--occlusion_threshold", type=float, default=0.01)
    parser.add_argument("--min_progress", type=float, default=0.02)
    parser.add_argument("--progress_window", type=int, default=5)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--output_dir", type=Path, default=Path("mujoco_ant_hierarchical/artifacts/step3_4_vision_uncertainty"))
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    controller = DirectionalAntController(
        config=DirectionalControllerConfig(success_radius=0.75, slow_radius=7.0, min_action_scale=0.3)
    )
    results = []
    image_dir = output_dir / "sample_images"
    for episode in range(args.episodes):
        result = run_episode(
            episode=episode,
            controller=controller,
            max_steps=args.max_steps,
            seed=args.seed + episode,
            depth_threshold=args.depth_threshold,
            min_progress=args.min_progress,
            progress_window=args.progress_window,
            width=args.width,
            height=args.height,
            sample_image_steps={0, 40, 80, 120},
            image_dir=image_dir,
        )
        results.append(result)
        print(
            f"episode={episode} steps={result['steps']} success={result['success']} "
            f"min_obstacle_distance={min(result['obstacle_distance']):.3f}",
            flush=True,
        )

    plots_dir = output_dir / "plots"
    for result in results:
        plot_episode_signals(result, plots_dir / f"episode_{result['episode']:03d}_signals.png", obstacle_near_threshold=args.obstacle_near_threshold)
    plot_scatter(results, plots_dir / "obstacle_distance_vs_uncertainty.png")

    timing = summarize_timing(
        results,
        obstacle_near_threshold=args.obstacle_near_threshold,
        progress_threshold=args.progress_threshold,
        occupancy_threshold=args.occupancy_threshold,
        occlusion_threshold=args.occlusion_threshold,
    )
    summary = {
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "depth_rendering": "enabled",
        "depth_threshold": args.depth_threshold,
        "obstacle_near_threshold": args.obstacle_near_threshold,
        "thresholds": {
            "progress": args.progress_threshold,
            "occupancy": args.occupancy_threshold,
            "occlusion": args.occlusion_threshold,
        },
        "success_rate": sum(r["success"] for r in results) / len(results),
        "mean_min_obstacle_distance": float(np.mean([min(r["obstacle_distance"]) for r in results])),
        "timing": timing,
        "results": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
