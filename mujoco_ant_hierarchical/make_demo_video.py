#!/usr/bin/env python3
"""Render a replay video for a successful hierarchical Ant episode."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from ant_directional_controller import DirectionalAntController, DirectionalControllerConfig
from ant_scene import DEFAULT_GOAL_XY, DEFAULT_OBSTACLE_POS, DEFAULT_OBSTACLE_SIZE, make_scene_env
from step3_3_llm_waypoint_experiment import waypoint_gate_completed, waypoint_gate_valid


def load_episode(summary_path: Path, *, seed: int) -> dict:
    summary = json.loads(summary_path.read_text())
    for result in summary["results"]:
        if int(result["seed"]) == seed:
            return {**summary, "episode": result}
    raise ValueError(f"seed {seed} not found in {summary_path}")


def draw_label(canvas: Image.Image, lines: list[str]) -> None:
    draw = ImageDraw.Draw(canvas)
    box_h = 18 * len(lines) + 12
    draw.rectangle((8, 8, 560, 8 + box_h), fill=(0, 0, 0))
    y = 14
    for line in lines:
        draw.text((14, y), line, fill=(255, 255, 255))
        y += 18


def render_camera(renderer: mujoco.Renderer, env, camera_name: str) -> np.ndarray:
    renderer.update_scene(env.unwrapped.data, camera=camera_name)
    return renderer.render()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("mujoco_ant_hierarchical/artifacts/step3_4_final_40eps_depth/uncertainty_switching/summary.json"),
    )
    parser.add_argument("--seed", type=int, default=6001)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mujoco_ant_hierarchical/artifacts/step3_4_final_40eps_demo/depth_switching_seed6001_demo.mp4"),
    )
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=360)
    args = parser.parse_args()

    bundle = load_episode(args.summary, seed=args.seed)
    summary = {k: v for k, v in bundle.items() if k != "episode"}
    episode = bundle["episode"]

    goal_controller = DirectionalAntController(
        config=DirectionalControllerConfig(
            success_radius=float(summary["success_radius"]) if "success_radius" in summary else 0.75,
            slow_radius=float(summary["goal_action_scaling"]["slow_radius"]),
            min_action_scale=float(summary["goal_action_scaling"]["min_action_scale"]),
        )
    )
    waypoint_controller = DirectionalAntController(
        config=DirectionalControllerConfig(
            success_radius=float(summary["success_radius"]) if "success_radius" in summary else 0.75,
            slow_radius=float(summary["waypoint_action_scaling"]["slow_radius"]),
            min_action_scale=float(summary["waypoint_action_scaling"]["min_action_scale"]),
        )
    )

    call_schedule = {int(call["step"]): np.array(call["waypoint"], dtype=np.float64) for call in episode["llm_calls"] if call["waypoint"] is not None}
    max_steps = int(episode["steps"])
    success_radius = float(summary.get("success_radius", 0.75))
    gate_x_margin = float(summary.get("gate_x_margin", 0.05))
    gate_y_abs = float(summary.get("gate_y_abs", 0.5))
    waypoint_timeout_steps = int(summary.get("waypoint_timeout_steps", 50))

    env = make_scene_env(with_obstacle=True)
    obs, _ = env.reset(seed=args.seed)
    renderer = mujoco.Renderer(env.unwrapped.model, height=args.height, width=args.width)

    active_waypoint: np.ndarray | None = None
    active_waypoint_start: int | None = None
    active_waypoint_side: int | None = None
    active_waypoint_gate_valid = True
    next_waypoint_index = 0
    llm_steps = sorted(call_schedule)

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        (args.width * 2, args.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {output}")

    trajectory: list[np.ndarray] = []
    for step in range(max_steps):
        current_xy = np.array(env.unwrapped.data.qpos[:2], dtype=np.float64)
        trajectory.append(current_xy.copy())
        goal_distance = float(np.linalg.norm(np.array(DEFAULT_GOAL_XY, dtype=np.float64) - current_xy))

        if active_waypoint is not None:
            completed = waypoint_gate_completed(
                current_xy,
                active_waypoint_side or 1,
                valid=active_waypoint_gate_valid,
                gate_x_margin=gate_x_margin,
                gate_y_abs=gate_y_abs,
            )
            timed_out = active_waypoint_start is not None and step - active_waypoint_start >= waypoint_timeout_steps
            if completed or timed_out:
                active_waypoint = None
                active_waypoint_start = None
                active_waypoint_side = None
                active_waypoint_gate_valid = True

        if step in call_schedule and active_waypoint is None:
            active_waypoint = call_schedule[step].copy()
            active_waypoint_start = step
            active_waypoint_side = 1 if active_waypoint[1] >= DEFAULT_OBSTACLE_POS[1] else -1
            active_waypoint_gate_valid = True
            next_waypoint_index += 1

        target_xy = active_waypoint if active_waypoint is not None else np.array(DEFAULT_GOAL_XY, dtype=np.float64)
        controller = waypoint_controller if active_waypoint is not None else goal_controller
        action, target_heading, info = controller.predict_with_info(obs, current_xy, target_xy)
        obs, _, terminated, truncated, _ = env.step(action)

        overhead = render_camera(renderer, env, "overhead")
        ego = render_camera(renderer, env, "ego")
        canvas = Image.new("RGB", (args.width * 2, args.height), (255, 255, 255))
        canvas.paste(Image.fromarray(overhead), (0, 0))
        canvas.paste(Image.fromarray(ego), (args.width, 0))
        status = "goal" if active_waypoint is None else "waypoint"
        waypoint_text = "none" if active_waypoint is None else f"({active_waypoint[0]:.2f}, {active_waypoint[1]:.2f})"
        draw_label(
            canvas,
            [
                f"seed={args.seed}  step={step+1}/{max_steps}  status={status}  goal_dist={goal_distance:.2f}",
                f"target={waypoint_text}  heading={target_heading:.2f}  action_scale={info['action_scale']:.2f}",
                f"llm_calls={next_waypoint_index}  success_radius={success_radius:.2f}",
            ],
        )
        writer.write(cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR))

        if terminated or truncated or goal_distance < success_radius:
            break

    writer.release()
    renderer.close()
    env.close()

    print(json.dumps({
        "output": str(output),
        "frames": len(trajectory),
        "seed": args.seed,
        "llm_calls_used": next_waypoint_index,
        "llm_steps": llm_steps,
    }, indent=2))


if __name__ == "__main__":
    main()
