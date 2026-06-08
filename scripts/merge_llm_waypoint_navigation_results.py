#!/usr/bin/env python3
"""Merge LLM waypoint navigation condition summaries."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="Merge LLM waypoint navigation summaries.")
parser.add_argument("--output_dir", type=str, default="experiments/llm_waypoint_navigation")
args = parser.parse_args()


CONDITIONS = ["policy_only", "always_llm", "uncertainty_switching"]


def main() -> None:
    output_dir = Path(args.output_dir)
    summaries = {}
    for condition in CONDITIONS:
        path = output_dir / condition / "metrics" / f"{condition}_summary.json"
        if path.exists():
            summaries[condition] = json.loads(path.read_text(encoding="utf-8"))

    combined = {
        "conditions": {
            condition: {
                "success_rate": summary["success_rate"],
                "mean_episode_length": summary["mean_episode_length"],
                "mean_final_distance": summary["mean_final_distance"],
                "total_llm_calls": summary["total_llm_calls"],
                "llm_call_rate": summary["llm_call_rate"],
                "llm_errors": len(summary["llm_errors"]),
            }
            for condition, summary in summaries.items()
        },
        "summary_paths": {
            condition: str(output_dir / condition / "metrics" / f"{condition}_summary.json") for condition in summaries
        },
    }
    metrics_dir = output_dir / "metrics"
    plots_dir = output_dir / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    combined_path = metrics_dir / "llm_waypoint_navigation_summary.json"
    combined_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")

    labels = list(summaries)
    values = [combined["conditions"][label]["success_rate"] for label in labels]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=["#4c78a8", "#f58518", "#54a24b"])
    plt.ylim(0.0, 1.0)
    plt.ylabel("success rate")
    plt.title("Waypoint navigation success rate")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plot_path = plots_dir / "success_rate_bar.png"
    plt.savefig(plot_path, dpi=160)
    plt.close()

    print(json.dumps(combined, indent=2))
    print(f"[WROTE] {combined_path}")
    print(f"[WROTE] {plot_path}")


if __name__ == "__main__":
    main()
