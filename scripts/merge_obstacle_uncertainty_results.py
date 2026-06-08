#!/usr/bin/env python3
"""Merge scenario A/B obstacle uncertainty rollouts and generate comparison plots."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="Merge obstacle uncertainty scenario summaries.")
parser.add_argument("--output_dir", type=str, default="experiments/obstacle_uncertainty")
parser.add_argument("--trigger_cooldown", type=int, default=20)
args = parser.parse_args()


def load_scenario(path: Path, scenario_key: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["scenarios"][scenario_key]


def plot_trajectory_comparison(output_path: Path, scenario_a: dict, scenario_b: dict, goal: list[float]) -> None:
    plt.figure(figsize=(7, 5))
    for label, scenario in (("A: no obstacle", scenario_a), ("B: static obstacle", scenario_b)):
        result = scenario["results"][0]
        xs = [p[0] for p in result["trajectory"]]
        ys = [p[1] for p in result["trajectory"]]
        plt.plot(xs, ys, linewidth=2.0, label=label)
    plt.scatter([0.0], [0.0], c="green", s=50, label="start")
    plt.scatter([goal[0]], [goal[1]], c="red", s=70, marker="*", label="goal")
    ox, oy, _ = scenario_b["obstacle_pos"]
    sx, sy, _ = scenario_b["obstacle_size"]
    rect = plt.Rectangle((ox - sx / 2, oy - sy / 2), sx, sy, color="black", alpha=0.25, label="obstacle")
    plt.gca().add_patch(rect)
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Scenario A vs B trajectory")
    plt.legend(loc="best")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return float(ordered[index])


def cooldown_triggers(uncertainty: list[float], threshold: float, cooldown: int) -> list[int]:
    triggers = []
    next_allowed = 0
    for idx, value in enumerate(uncertainty):
        if idx >= next_allowed and value >= threshold and value > 0.0:
            triggers.append(idx)
            next_allowed = idx + cooldown
    return triggers


def mean_combined_trigger_count(scenario: dict, threshold: float) -> float:
    return sum(len(cooldown_triggers(r["uncertainty"], threshold, args.trigger_cooldown)) for r in scenario["results"]) / len(
        scenario["results"]
    )


def plot_uncertainty_comparison(output_path: Path, scenario_a: dict, scenario_b: dict, combined_threshold: float) -> None:
    plt.figure(figsize=(8, 4.5))
    for label, scenario, color in (
        ("A: no obstacle", scenario_a, "tab:blue"),
        ("B: static obstacle", scenario_b, "tab:orange"),
    ):
        result = scenario["results"][0]
        steps = list(range(len(result["uncertainty"])))
        plt.plot(steps, result["uncertainty"], linewidth=1.8, color=color, label=label)
        plt.axhline(
            scenario["uncertainty_threshold_p90"],
            color=color,
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
        )
        for trigger_step in result["trigger_steps"]:
            plt.axvline(trigger_step, color=color, alpha=0.16, linewidth=1.0)
    plt.axhline(combined_threshold, color="black", linestyle=":", linewidth=1.5, label="combined p90 threshold")
    plt.xlabel("step")
    plt.ylabel("progress uncertainty")
    plt.title("Scenario A vs B uncertainty")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def behavior_label(scenario_b: dict) -> str:
    if scenario_b["success_rate"] > 0.0 and scenario_b["mean_inside_obstacle_steps"] > 0.0:
        return "passed_through_or_over_obstacle_region"
    if scenario_b["done_rate"] > 0.0:
        return "terminated_after_obstacle_contact_or_fall"
    if scenario_b["mean_final_distance"] > 1.0 and scenario_b["mean_min_distance"] < scenario_b["mean_final_distance"] + 0.2:
        return "blocked_or_stalled_near_obstacle"
    return "continued_pushing_without_success"


def main() -> None:
    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / "metrics"
    plots_dir = output_dir / "plots"
    scenario_a = load_scenario(metrics_dir / "scenario_a_no_obstacle_summary.json", "scenario_a_no_obstacle")
    scenario_b = load_scenario(metrics_dir / "scenario_b_static_obstacle_summary.json", "scenario_b_static_obstacle")
    goal = scenario_a["results"][0]["goal"]
    all_uncertainty = []
    for scenario in (scenario_a, scenario_b):
        for result in scenario["results"]:
            all_uncertainty.extend(result["uncertainty"])
    combined_threshold = percentile(all_uncertainty, 0.9)

    plot_trajectory_comparison(plots_dir / "scenario_a_vs_b_trajectory.png", scenario_a, scenario_b, goal)
    plot_uncertainty_comparison(plots_dir / "scenario_a_vs_b_uncertainty.png", scenario_a, scenario_b, combined_threshold)

    summary = {
        "goal": goal,
        "combined_uncertainty_threshold_p90": combined_threshold,
        "scenario_a_no_obstacle": {
            "success_rate": scenario_a["success_rate"],
            "mean_final_distance": scenario_a["mean_final_distance"],
            "mean_uncertainty": scenario_a["mean_uncertainty"],
            "mean_max_uncertainty": scenario_a["mean_max_uncertainty"],
            "uncertainty_threshold_p90": scenario_a["uncertainty_threshold_p90"],
            "mean_trigger_count": sum(len(r["trigger_steps"]) for r in scenario_a["results"]) / len(scenario_a["results"]),
            "mean_trigger_count_combined_threshold": mean_combined_trigger_count(scenario_a, combined_threshold),
        },
        "scenario_b_static_obstacle": {
            "success_rate": scenario_b["success_rate"],
            "done_rate": scenario_b["done_rate"],
            "mean_final_distance": scenario_b["mean_final_distance"],
            "mean_min_distance": scenario_b["mean_min_distance"],
            "mean_uncertainty": scenario_b["mean_uncertainty"],
            "mean_max_uncertainty": scenario_b["mean_max_uncertainty"],
            "uncertainty_threshold_p90": scenario_b["uncertainty_threshold_p90"],
            "mean_trigger_count": sum(len(r["trigger_steps"]) for r in scenario_b["results"]) / len(scenario_b["results"]),
            "mean_trigger_count_combined_threshold": mean_combined_trigger_count(scenario_b, combined_threshold),
            "mean_inside_obstacle_steps": scenario_b["mean_inside_obstacle_steps"],
            "obstacle_pos": scenario_b["obstacle_pos"],
            "obstacle_size": scenario_b["obstacle_size"],
            "behavior": behavior_label(scenario_b),
        },
        "artifact_sources": {
            "scenario_a": str(metrics_dir / "scenario_a_no_obstacle_summary.json"),
            "scenario_b": str(metrics_dir / "scenario_b_static_obstacle_summary.json"),
            "trajectory_comparison": str(plots_dir / "scenario_a_vs_b_trajectory.png"),
            "uncertainty_comparison": str(plots_dir / "scenario_a_vs_b_uncertainty.png"),
        },
    }
    summary_path = metrics_dir / "obstacle_uncertainty_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[WROTE] {summary_path}")


if __name__ == "__main__":
    main()
