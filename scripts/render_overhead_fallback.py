#!/usr/bin/env python3
"""Render a 512x512 matplotlib overhead view for LLM fallback input."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="Render overhead fallback image from a rollout summary.")
parser.add_argument(
    "--summary",
    type=str,
    default="experiments/obstacle_uncertainty/metrics/scenario_b_static_obstacle_summary.json",
)
parser.add_argument("--scenario_key", type=str, default="scenario_b_static_obstacle")
parser.add_argument("--episode", type=int, default=0)
parser.add_argument("--step", type=int, default=-1)
parser.add_argument("--history", type=int, default=5)
parser.add_argument("--output", type=str, default="experiments/step4_fallback/overhead_llm_input.png")
parser.add_argument("--metadata", type=str, default="experiments/step4_fallback/overhead_llm_input.json")
args = parser.parse_args()


def choose_step(result: dict) -> int:
    if args.step >= 0:
        return min(args.step, len(result["trajectory"]) - 1)
    trigger_steps = result.get("trigger_steps", [])
    if trigger_steps:
        return min(trigger_steps[0], len(result["trajectory"]) - 1)
    return len(result["trajectory"]) - 1


def main() -> None:
    summary_path = Path(args.summary)
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    scenario = data["scenarios"][args.scenario_key]
    result = scenario["results"][args.episode]
    step = choose_step(result)
    trajectory = result["trajectory"]
    goal = result["goal"]
    current = trajectory[step]
    start_idx = max(0, step - args.history + 1)
    recent = trajectory[start_idx : step + 1]

    obstacle_pos = scenario.get("obstacle_pos") or [2.5, 0.0, 0.5]
    obstacle_size = scenario.get("obstacle_size") or [1.0, 1.0, 1.0]

    fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=100)
    ax.set_facecolor("white")
    ax.grid(True, color="#d0d0d0", linewidth=0.8, alpha=0.8)
    ax.axhline(0.0, color="#555555", linewidth=1.0, alpha=0.65)
    ax.axvline(0.0, color="#555555", linewidth=1.0, alpha=0.65)

    ox, oy, _ = obstacle_pos
    sx, sy, _ = obstacle_size
    rect = plt.Rectangle((ox - sx / 2, oy - sy / 2), sx, sy, color="#808080", alpha=0.55, label="obstacle")
    ax.add_patch(rect)

    if len(recent) > 1:
        ax.plot([p[0] for p in recent], [p[1] for p in recent], color="#1f77b4", linewidth=3.0, label="recent path")
    ax.scatter([current[0]], [current[1]], s=130, color="#1f77b4", edgecolor="black", linewidth=0.8, label="H1")
    ax.scatter([goal[0]], [goal[1]], s=220, color="#2ca02c", marker="*", edgecolor="black", linewidth=0.8, label="goal")

    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-2.0, 2.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout(pad=0.6)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)

    metadata = {
        "status": "success",
        "source_summary": str(summary_path),
        "scenario_key": args.scenario_key,
        "episode": args.episode,
        "step": step,
        "image_path": str(output_path),
        "resolution": [512, 512],
        "current_pos": current,
        "goal": goal,
        "obstacle_pos": obstacle_pos,
        "obstacle_size": obstacle_size,
        "recent_trajectory": recent,
    }
    metadata_path = Path(args.metadata)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
