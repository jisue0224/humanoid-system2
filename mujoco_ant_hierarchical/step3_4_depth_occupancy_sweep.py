#!/usr/bin/env python3
"""Sweep egocentric depth occupancy ROI and thresholds without LLM calls."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from ant_directional_controller import DirectionalAntController, DirectionalControllerConfig
from ant_scene import DEFAULT_GOAL_XY, DEFAULT_OBSTACLE_POS, DEFAULT_OBSTACLE_SIZE, make_scene_env


THRESHOLDS = [0.05, 0.08, 0.10, 0.15, 0.20, 0.25]


ROI_SPECS = {
    "center_40x50_mean": {"x0": 0.30, "x1": 0.70, "y0": 0.25, "y1": 0.75, "stat": "mean"},
    "center_20x30_mean": {"x0": 0.40, "x1": 0.60, "y0": 0.35, "y1": 0.65, "stat": "mean"},
    "lower_half_mean": {"x0": 0.0, "x1": 1.0, "y0": 0.50, "y1": 1.0, "stat": "mean"},
    "center_40x50_min": {"x0": 0.30, "x1": 0.70, "y0": 0.25, "y1": 0.75, "stat": "min"},
}


def box_distance_xy(point_xy: np.ndarray) -> float:
    ox, oy, _ = DEFAULT_OBSTACLE_POS
    sx, sy, _ = DEFAULT_OBSTACLE_SIZE
    dx = max(abs(float(point_xy[0]) - ox) - sx, 0.0)
    dy = max(abs(float(point_xy[1]) - oy) - sy, 0.0)
    return float(np.hypot(dx, dy))


def roi_values(depth: np.ndarray, spec: dict, *, depth_threshold: float) -> tuple[float, float]:
    height, width = depth.shape
    x0 = int(width * spec["x0"])
    x1 = int(width * spec["x1"])
    y0 = int(height * spec["y0"])
    y1 = int(height * spec["y1"])
    crop = depth[y0:y1, x0:x1]
    finite = crop[np.isfinite(crop)]
    if finite.size == 0:
        return 0.0, float("nan")
    clipped = np.clip(finite, 0.0, depth_threshold * 4.0)
    value = float(clipped.min() if spec["stat"] == "min" else clipped.mean())
    uncertainty = max(0.0, (depth_threshold - value) / depth_threshold)
    return float(uncertainty), value


def render_depth(renderer: mujoco.Renderer, env) -> np.ndarray:
    renderer.enable_depth_rendering()
    renderer.update_scene(env.unwrapped.data, camera="ego")
    return np.asarray(renderer.render(), dtype=np.float32)


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
    width: int,
    height: int,
) -> dict:
    env = make_scene_env(with_obstacle=True)
    obs, _ = env.reset(seed=seed)
    renderer = mujoco.Renderer(env.unwrapped.model, height=height, width=width)
    goal_xy = np.array(DEFAULT_GOAL_XY, dtype=np.float64)

    traces = {name: [] for name in ROI_SPECS}
    raw_values = {name: [] for name in ROI_SPECS}
    obstacle_distance = []
    goal_distance = []
    trajectory = []
    success = False
    terminated = False
    truncated = False

    for step in range(max_steps):
        current_xy = np.array(env.unwrapped.data.qpos[:2], dtype=np.float64)
        trajectory.append(current_xy.tolist())
        goal_dist = float(np.linalg.norm(goal_xy - current_xy))
        goal_distance.append(goal_dist)
        obstacle_distance.append(box_distance_xy(current_xy))

        depth = render_depth(renderer, env)
        for name, spec in ROI_SPECS.items():
            uncertainty, value = roi_values(depth, spec, depth_threshold=depth_threshold)
            traces[name].append(uncertainty)
            raw_values[name].append(value)

        if goal_dist < 0.75:
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
        "goal_distance": goal_distance,
        "obstacle_distance": obstacle_distance,
        "signals": traces,
        "raw_depth_values": raw_values,
    }


def summarize(results: list[dict], *, obstacle_near_threshold: float) -> dict:
    summary = {}
    obstacle_all = np.concatenate([np.array(result["obstacle_distance"]) for result in results])
    far_mask = obstacle_all > obstacle_near_threshold
    for roi_name in ROI_SPECS:
        values_all = np.concatenate([np.array(result["signals"][roi_name]) for result in results])
        corr = float(np.corrcoef(obstacle_all, values_all)[0, 1])
        roi_summary = {
            "corr_distance_signal": corr,
            "near_mean": float(values_all[~far_mask].mean()),
            "far_mean": float(values_all[far_mask].mean()),
            "thresholds": {},
        }
        for threshold in THRESHOLDS:
            offsets = []
            early_hits = 0
            episodes_with_trigger = 0
            for result in results:
                near_step = first_distance_crossing(result["obstacle_distance"], obstacle_near_threshold)
                trigger_step = first_threshold_crossing(result["signals"][roi_name], threshold)
                if trigger_step is not None:
                    episodes_with_trigger += 1
                if near_step is not None and trigger_step is not None:
                    offsets.append(trigger_step - near_step)
                    early_hits += int(trigger_step <= near_step)
            false_positive_rate = float(((values_all >= threshold) & far_mask).sum() / max(1, far_mask.sum()))
            near_trigger_rate = float(((values_all >= threshold) & (~far_mask)).sum() / max(1, (~far_mask).sum()))
            roi_summary["thresholds"][str(threshold)] = {
                "mean_offset": None if not offsets else float(np.mean(offsets)),
                "median_offset": None if not offsets else float(np.median(offsets)),
                "early_or_on_time_episodes": early_hits,
                "episodes_with_offset": len(offsets),
                "episodes_with_any_trigger": episodes_with_trigger,
                "false_positive_rate_far_steps": false_positive_rate,
                "near_trigger_rate": near_trigger_rate,
            }
        summary[roi_name] = roi_summary
    return summary


def threshold_rows(summary: dict) -> list[dict]:
    rows = []
    for roi_name, roi_summary in summary.items():
        for threshold, stats in roi_summary["thresholds"].items():
            rows.append(
                {
                    "roi": roi_name,
                    "threshold": float(threshold),
                    "corr_distance_signal": roi_summary["corr_distance_signal"],
                    **stats,
                }
            )
    rows.sort(key=lambda row: (row["false_positive_rate_far_steps"], row["median_offset"] if row["median_offset"] is not None else 999))
    return rows


def plot_best_episode(result: dict, roi_name: str, threshold: float, path: Path, *, obstacle_near_threshold: float) -> None:
    steps = np.arange(result["steps"])
    obstacle_distance = np.array(result["obstacle_distance"])
    proximity = np.clip(1.0 - obstacle_distance / obstacle_near_threshold, 0.0, 1.0)
    signal = np.array(result["signals"][roi_name])
    near_step = first_distance_crossing(result["obstacle_distance"], obstacle_near_threshold)
    trigger_step = first_threshold_crossing(result["signals"][roi_name], threshold)

    plt.figure(figsize=(10, 4.6))
    plt.plot(steps, signal, label=f"{roi_name} depth occupancy", linewidth=1.8)
    plt.plot(steps, proximity, "--", label="ground-truth obstacle proximity", linewidth=1.6)
    plt.axhline(threshold, color="tab:orange", linestyle=":", label=f"threshold {threshold:.2f}")
    if near_step is not None:
        plt.axvline(near_step, color="black", linestyle=":", label="obstacle <= 1m")
    if trigger_step is not None:
        plt.axvline(trigger_step, color="tab:red", linestyle="--", label="signal trigger")
    plt.ylim(-0.03, 1.03)
    plt.xlabel("step")
    plt.ylabel("normalized value")
    plt.title(f"Episode {result['episode']:02d}: depth signal vs obstacle distance")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_threshold_sweep(summary: dict, path: Path) -> None:
    plt.figure(figsize=(10, 5))
    for roi_name, roi_summary in summary.items():
        thresholds = []
        offsets = []
        false_positive = []
        for threshold in THRESHOLDS:
            stats = roi_summary["thresholds"][str(threshold)]
            if stats["median_offset"] is None:
                continue
            thresholds.append(threshold)
            offsets.append(stats["median_offset"])
            false_positive.append(stats["false_positive_rate_far_steps"])
        plt.plot(false_positive, offsets, marker="o", label=roi_name)
        for fp, offset, threshold in zip(false_positive, offsets, thresholds):
            plt.text(fp, offset, f"{threshold:.2f}", fontsize=8)
    plt.axhline(0, color="black", linewidth=1, linestyle=":")
    plt.xlabel("false positive rate on far steps")
    plt.ylabel("median trigger offset [steps]")
    plt.title("Depth occupancy threshold/ROI tradeoff")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=220)
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--depth_threshold", type=float, default=3.0)
    parser.add_argument("--obstacle_near_threshold", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--output_dir", type=Path, default=Path("mujoco_ant_hierarchical/artifacts/step3_4_depth_occupancy_sweep"))
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    controller = DirectionalAntController(
        config=DirectionalControllerConfig(success_radius=0.75, slow_radius=7.0, min_action_scale=0.3)
    )
    results = []
    for episode in range(args.episodes):
        result = run_episode(
            episode=episode,
            controller=controller,
            max_steps=args.max_steps,
            seed=args.seed + episode,
            depth_threshold=args.depth_threshold,
            width=args.width,
            height=args.height,
        )
        results.append(result)
        print(
            f"episode={episode} steps={result['steps']} success={result['success']} "
            f"min_obstacle_distance={min(result['obstacle_distance']):.3f}",
            flush=True,
        )

    sweep_summary = summarize(results, obstacle_near_threshold=args.obstacle_near_threshold)
    rows = threshold_rows(sweep_summary)
    best = min(
        rows,
        key=lambda row: (
            row["false_positive_rate_far_steps"] > 0.25,
            abs(row["median_offset"] if row["median_offset"] is not None else 999),
            row["false_positive_rate_far_steps"],
        ),
    )
    # Prefer an actual early signal when close in quality.
    early_candidates = [
        row
        for row in rows
        if row["median_offset"] is not None
        and row["median_offset"] <= 0
        and row["false_positive_rate_far_steps"] <= 0.30
    ]
    if early_candidates:
        best = min(early_candidates, key=lambda row: (row["false_positive_rate_far_steps"], abs(row["median_offset"])))

    plots_dir = output_dir / "plots"
    plot_threshold_sweep(sweep_summary, plots_dir / "threshold_roi_tradeoff.png")
    example = min(results, key=lambda result: min(result["obstacle_distance"]))
    plot_best_episode(
        example,
        best["roi"],
        best["threshold"],
        plots_dir / "best_depth_signal_example.png",
        obstacle_near_threshold=args.obstacle_near_threshold,
    )

    summary = {
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "depth_threshold": args.depth_threshold,
        "obstacle_near_threshold": args.obstacle_near_threshold,
        "roi_specs": ROI_SPECS,
        "thresholds": THRESHOLDS,
        "best_candidate": best,
        "sweep": sweep_summary,
        "rows": rows,
        "results": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
