#!/usr/bin/env python3
"""Vision-based LLM waypoint experiment for hierarchical Ant."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from ant_directional_controller import DirectionalAntController, DirectionalControllerConfig
from ant_scene import DEFAULT_GOAL_XY, DEFAULT_OBSTACLE_POS, DEFAULT_OBSTACLE_SIZE, make_scene_env
from step3_2_obstacle_uncertainty import progress_uncertainty


SYSTEM_PROMPT = (
    "You are a navigation planner for a MuJoCo Ant robot. "
    "Choose one intermediate waypoint that helps the robot avoid the obstacle. "
    "The low-level controller can walk toward any waypoint by rotating its observation frame."
)


def encode_image(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def capture_frame(env, camera_name: str, path: Path, *, width: int = 640, height: int = 480) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(env.unwrapped.model, height=height, width=width)
    renderer.update_scene(env.unwrapped.data, camera=camera_name)
    frame = renderer.render()
    renderer.close()
    Image.fromarray(frame).save(path)


def parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def call_llm_waypoint(
    client: OpenAI,
    *,
    model: str,
    overhead_path: Path,
    ego_path: Path | None,
    current_xy: np.ndarray,
    goal_xy: np.ndarray,
    active_waypoint: np.ndarray | None,
    recent_waypoints: list[np.ndarray],
) -> tuple[np.ndarray | None, dict]:
    ox, oy, _ = DEFAULT_OBSTACLE_POS
    sx, sy, _ = DEFAULT_OBSTACLE_SIZE
    waypoint_text = "none" if active_waypoint is None else f"({active_waypoint[0]:.2f}, {active_waypoint[1]:.2f})"
    recent_text = "none"
    if recent_waypoints:
        recent_text = ", ".join(f"({point[0]:.2f}, {point[1]:.2f})" for point in recent_waypoints[-3:])
    user_text = (
        "The images are real MuJoCo RGB renders, not diagrams.\n"
        "Overhead image: red box is the obstacle, green sphere is the final goal, Ant is the robot.\n"
        "Egocentric image, if provided: robot torso front camera.\n\n"
        f"Current robot xy: ({current_xy[0]:.2f}, {current_xy[1]:.2f})\n"
        f"Final goal xy: ({goal_xy[0]:.2f}, {goal_xy[1]:.2f})\n"
        f"Obstacle center xy: ({ox:.2f}, {oy:.2f}); obstacle half-size xy: ({sx:.2f}, {sy:.2f})\n"
        f"Current active waypoint: {waypoint_text}\n\n"
        f"Recently attempted waypoints: {recent_text}\n"
        "If a recently attempted waypoint is close to the current plan and the robot did not make progress, choose a meaningfully different waypoint.\n\n"
        "Return ONE waypoint in world xy coordinates. Do not return the final goal if the obstacle blocks the direct route. "
        "Prefer a waypoint that goes clearly around either the upper or lower side of the red obstacle. "
        "Do not put the waypoint inside the obstacle. "
        "Respond as JSON only: {\"reasoning\": \"short\", \"waypoint_x\": number, \"waypoint_y\": number}"
    )
    content = [
        {"type": "text", "text": user_text},
        {"type": "image_url", "image_url": {"url": encode_image(overhead_path)}},
    ]
    if ego_path is not None:
        content.append({"type": "image_url", "image_url": {"url": encode_image(ego_path)}})

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or "{}"
    parsed = parse_json(raw)
    waypoint = None
    if "waypoint_x" in parsed and "waypoint_y" in parsed:
        waypoint = np.array([float(parsed["waypoint_x"]), float(parsed["waypoint_y"])], dtype=np.float64)
    return waypoint, {"raw": raw, "parsed": parsed}


def run_episode(
    *,
    episode: int,
    condition: str,
    goal_controller: DirectionalAntController,
    waypoint_controller: DirectionalAntController,
    client: OpenAI | None,
    model: str,
    output_dir: Path,
    max_steps: int,
    success_radius: float,
    waypoint_radius: float,
    uncertainty_threshold: float,
    uncertainty_window: int,
    min_progress: float,
    llm_cooldown: int,
    uncertainty_warmup_steps: int,
    waypoint_timeout_steps: int,
    goal_near_suppression_radius: float,
    include_ego: bool,
    seed: int,
) -> dict:
    env = make_scene_env(with_obstacle=True)
    obs, _ = env.reset(seed=seed)
    goal_xy = np.array(DEFAULT_GOAL_XY, dtype=np.float64)
    active_waypoint: np.ndarray | None = None
    active_waypoint_start: int | None = None
    next_llm_allowed = 0
    recent_waypoints: list[np.ndarray] = []
    trajectory = []
    distances = []
    llm_calls = []
    success = False
    terminated = False
    truncated = False

    for step in range(max_steps):
        current_xy = np.array(env.unwrapped.data.qpos[:2], dtype=np.float64)
        goal_distance = float(np.linalg.norm(goal_xy - current_xy))
        trajectory.append(current_xy.tolist())
        distances.append(goal_distance)
        if goal_distance < success_radius:
            success = True
            break

        if active_waypoint is not None and float(np.linalg.norm(active_waypoint - current_xy)) < waypoint_radius:
            active_waypoint = None
            active_waypoint_start = None
        elif (
            active_waypoint is not None
            and active_waypoint_start is not None
            and step - active_waypoint_start >= waypoint_timeout_steps
        ):
            active_waypoint = None
            active_waypoint_start = None

        uncertainty = progress_uncertainty(distances, window=uncertainty_window, min_progress=min_progress)[-1]
        should_call = False
        if condition == "always_llm":
            should_call = active_waypoint is None and step >= next_llm_allowed
        elif condition == "uncertainty_switching":
            should_call = (
                active_waypoint is None
                and step >= next_llm_allowed
                and step >= uncertainty_warmup_steps
                and goal_distance > goal_near_suppression_radius
                and uncertainty >= uncertainty_threshold
                and uncertainty > 0.0
            )

        if should_call:
            assert client is not None
            call_dir = output_dir / "llm_images" / f"episode_{episode:03d}_step_{step:03d}"
            overhead_path = call_dir / "overhead.png"
            ego_path = call_dir / "ego.png" if include_ego else None
            capture_frame(env, "overhead", overhead_path)
            if ego_path is not None:
                capture_frame(env, "ego", ego_path)
            waypoint, llm_info = call_llm_waypoint(
                client,
                model=model,
                overhead_path=overhead_path,
                ego_path=ego_path,
                current_xy=current_xy,
                goal_xy=goal_xy,
                active_waypoint=active_waypoint,
                recent_waypoints=recent_waypoints,
            )
            if waypoint is not None:
                active_waypoint = waypoint
                active_waypoint_start = step
                recent_waypoints.append(waypoint)
            llm_calls.append(
                {
                    "step": step,
                    "uncertainty": uncertainty,
                    "current_xy": current_xy.tolist(),
                    "waypoint": None if waypoint is None else waypoint.tolist(),
                    "overhead_path": str(overhead_path),
                    "ego_path": None if ego_path is None else str(ego_path),
                    **llm_info,
                }
            )
            next_llm_allowed = step + llm_cooldown

        target_xy = active_waypoint if active_waypoint is not None else goal_xy
        controller = waypoint_controller if active_waypoint is not None else goal_controller
        action, _, _ = controller.predict_with_info(obs, current_xy, target_xy)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    env.close()
    uncertainty_trace = progress_uncertainty(distances, window=uncertainty_window, min_progress=min_progress)
    return {
        "episode": episode,
        "condition": condition,
        "seed": seed,
        "success": success,
        "steps": step + 1,
        "terminated": terminated,
        "truncated": truncated,
        "final_xy": trajectory[-1],
        "final_distance": distances[-1],
        "trajectory": trajectory,
        "distances": distances,
        "uncertainty": uncertainty_trace,
        "llm_calls": llm_calls,
        "llm_call_count": len(llm_calls),
    }


def plot_condition(results: list[dict], path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    ox, oy, _ = DEFAULT_OBSTACLE_POS
    sx, sy, _ = DEFAULT_OBSTACLE_SIZE
    plt.gca().add_patch(
        plt.Rectangle((ox - sx, oy - sy), sx * 2, sy * 2, color="red", alpha=0.25, label="obstacle")
    )
    for result in results:
        trajectory = np.array(result["trajectory"], dtype=np.float64)
        plt.plot(trajectory[:, 0], trajectory[:, 1], linewidth=1.3, alpha=0.75)
        for call in result["llm_calls"]:
            waypoint = call.get("waypoint")
            if waypoint is not None:
                plt.scatter([waypoint[0]], [waypoint[1]], marker="D", s=35, c="purple", alpha=0.8)
    plt.scatter([0.0], [0.0], c="black", s=45, label="start")
    plt.scatter([DEFAULT_GOAL_XY[0]], [DEFAULT_GOAL_XY[1]], marker="*", s=95, c="green", label="goal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=("policy_only", "always_llm", "uncertainty_switching"), required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=220)
    parser.add_argument("--success_radius", type=float, default=0.75)
    parser.add_argument("--waypoint_radius", type=float, default=0.75)
    parser.add_argument("--slow_radius", type=float, default=7.0)
    parser.add_argument("--min_action_scale", type=float, default=0.3)
    parser.add_argument("--waypoint_slow_radius", type=float, default=0.0)
    parser.add_argument("--waypoint_min_action_scale", type=float, default=1.0)
    parser.add_argument("--uncertainty_threshold", type=float, default=0.02)
    parser.add_argument("--uncertainty_window", type=int, default=5)
    parser.add_argument("--min_progress", type=float, default=0.02)
    parser.add_argument("--llm_cooldown", type=int, default=30)
    parser.add_argument("--uncertainty_warmup_steps", type=int, default=15)
    parser.add_argument("--waypoint_timeout_steps", type=int, default=50)
    parser.add_argument("--goal_near_suppression_radius", type=float, default=1.0)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--include_ego", action="store_true")
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--output_dir", type=Path, default=Path("mujoco_ant_hierarchical/artifacts/step3_3_llm"))
    args = parser.parse_args()

    load_dotenv(Path.cwd() / ".env")
    client = None
    if args.condition != "policy_only":
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for LLM conditions.")
        client = OpenAI()

    output_dir = args.output_dir / args.condition
    output_dir.mkdir(parents=True, exist_ok=True)
    goal_controller = DirectionalAntController(
        config=DirectionalControllerConfig(
            success_radius=args.success_radius,
            slow_radius=args.slow_radius,
            min_action_scale=args.min_action_scale,
        )
    )
    waypoint_controller = DirectionalAntController(
        config=DirectionalControllerConfig(
            success_radius=args.waypoint_radius,
            slow_radius=args.waypoint_slow_radius,
            min_action_scale=args.waypoint_min_action_scale,
        )
    )
    results = []
    for episode in range(args.episodes):
        result = run_episode(
            episode=episode,
            condition=args.condition,
            goal_controller=goal_controller,
            waypoint_controller=waypoint_controller,
            client=client,
            model=args.model,
            output_dir=output_dir,
            max_steps=args.max_steps,
            success_radius=args.success_radius,
            waypoint_radius=args.waypoint_radius,
            uncertainty_threshold=args.uncertainty_threshold,
            uncertainty_window=args.uncertainty_window,
            min_progress=args.min_progress,
            llm_cooldown=args.llm_cooldown,
            uncertainty_warmup_steps=args.uncertainty_warmup_steps,
            waypoint_timeout_steps=args.waypoint_timeout_steps,
            goal_near_suppression_radius=args.goal_near_suppression_radius,
            include_ego=args.include_ego,
            seed=args.seed + episode,
        )
        results.append(result)
        print(
            f"[{args.condition}] episode={episode} success={result['success']} "
            f"final_distance={result['final_distance']:.3f} llm_calls={result['llm_call_count']}",
            flush=True,
        )

    summary = {
        "condition": args.condition,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "success_rate": sum(result["success"] for result in results) / len(results),
        "mean_final_distance": float(np.mean([result["final_distance"] for result in results])),
        "mean_llm_calls": float(np.mean([result["llm_call_count"] for result in results])),
        "total_llm_calls": int(sum(result["llm_call_count"] for result in results)),
        "postprocess": "disabled",
        "goal_action_scaling": {
            "slow_radius": args.slow_radius,
            "min_action_scale": args.min_action_scale,
        },
        "waypoint_action_scaling": {
            "slow_radius": args.waypoint_slow_radius,
            "min_action_scale": args.waypoint_min_action_scale,
        },
        "waypoint_timeout_steps": args.waypoint_timeout_steps,
        "uncertainty_warmup_steps": args.uncertainty_warmup_steps,
        "results": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_condition(results, output_dir / "trajectories.png", f"{args.condition} trajectories")
    print(json.dumps({k: summary[k] for k in summary if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
