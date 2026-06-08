#!/usr/bin/env python3
"""Step 5: LLM-assisted H1 obstacle navigation with overhead images."""

import argparse
import base64
import json
import os
import re
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run H1 LLM navigation condition.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-H1-v0")
parser.add_argument("--condition", choices=("policy_only", "always_llm", "uncertainty_switching"), required=True)
parser.add_argument("--num_envs", type=int, default=20)
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--max_steps", type=int, default=500)
parser.add_argument("--success_radius", type=float, default=0.5)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--output_dir", type=str, default="experiments/llm_navigation")
parser.add_argument("--goal_x", type=float, default=5.0)
parser.add_argument("--goal_y", type=float, default=0.0)
parser.add_argument("--obstacle_pos", type=float, nargs=3, default=(2.5, 0.0, 0.5))
parser.add_argument("--obstacle_size", type=float, nargs=3, default=(1.0, 1.0, 1.0))
parser.add_argument("--max_vx", type=float, default=1.0)
parser.add_argument("--yaw_gain", type=float, default=1.5)
parser.add_argument("--yaw_limit", type=float, default=1.0)
parser.add_argument("--forward_angle_gate", type=float, default=1.2)
parser.add_argument("--uncertainty_window", type=int, default=5)
parser.add_argument("--min_progress", type=float, default=0.02)
parser.add_argument("--uncertainty_threshold", type=float, default=0.0247)
parser.add_argument("--llm_cooldown", type=int, default=20)
parser.add_argument("--llm_duration", type=int, default=20)
parser.add_argument("--llm_model", type=str, default="gpt-5.4")
parser.add_argument("--llm_timeout", type=float, default=60.0)
parser.add_argument("--max_llm_calls", type=int, default=100000)
parser.add_argument("--dry_run_llm", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from dotenv import load_dotenv
from openai import OpenAI
from rsl_rl.runners import OnPolicyRunner

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg


SYSTEM_PROMPT = (
    "You are a navigation assistant for a humanoid robot.\n"
    "You receive an overhead view of the environment\n"
    "and suggest velocity commands to avoid obstacles."
)


def configure_env(env_cfg) -> None:
    cmd = env_cfg.commands.base_velocity
    cmd.heading_command = False
    cmd.rel_heading_envs = 0.0
    cmd.rel_standing_envs = 0.0
    cmd.debug_vis = False
    cmd.resampling_time_range = (1000.0, 1000.0)
    cmd.ranges.lin_vel_x = (-1.0, 1.0)
    cmd.ranges.lin_vel_y = (-1.0, 1.0)
    cmd.ranges.ang_vel_z = (-1.0, 1.0)
    cmd.ranges.heading = None
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 0
    env_cfg.sim.device = args_cli.device
    env_cfg.observations.policy.enable_corruption = False
    if getattr(env_cfg.events, "base_external_force_torque", None) is not None:
        env_cfg.events.base_external_force_torque = None
    if getattr(env_cfg.events, "push_robot", None) is not None:
        env_cfg.events.push_robot = None
    if getattr(env_cfg.events, "reset_base", None) is not None:
        env_cfg.events.reset_base.params = {
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
    if getattr(env_cfg.terminations, "base_contact", None) is not None:
        env_cfg.terminations.base_contact = None
    env_cfg.scene.obstacle = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle",
        spawn=sim_utils.CuboidCfg(
            size=tuple(args_cli.obstacle_size),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.2, 0.15)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(args_cli.obstacle_pos)),
    )


def load_api_key() -> None:
    load_dotenv(Path.cwd() / ".env")
    if not os.environ.get("OPENAI_API_KEY") and args_cli.condition != "policy_only" and not args_cli.dry_run_llm:
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in .env or the environment.")


def wrap_pi_tensor(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def compute_goal_command(robot, goal_xy: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
    pos = robot.data.root_pos_w[:, :2]
    delta = goal_xy - pos
    distance = torch.linalg.norm(delta, dim=1)
    target_heading = torch.atan2(delta[:, 1], delta[:, 0])
    heading_error = wrap_pi_tensor(target_heading - robot.data.heading_w)
    yaw = torch.clamp(args_cli.yaw_gain * heading_error, -args_cli.yaw_limit, args_cli.yaw_limit)
    alignment = torch.clamp(torch.cos(heading_error), min=0.0, max=1.0)
    angle_gate = (torch.abs(heading_error) < args_cli.forward_angle_gate).float()
    vx = torch.clamp(distance, max=args_cli.max_vx) * alignment * angle_gate
    vy = torch.zeros_like(vx)
    cmd = torch.stack([vx, vy, yaw], dim=1)
    cmd = torch.where(active_mask.unsqueeze(1), cmd, torch.zeros_like(cmd))
    cmd = torch.where(distance.unsqueeze(1) < args_cli.success_radius, torch.zeros_like(cmd), cmd)
    return cmd


def set_command(env, cmd: torch.Tensor) -> None:
    term = env.unwrapped.command_manager.get_term("base_velocity")
    term.vel_command_b[:] = cmd
    if hasattr(term, "is_standing_env"):
        term.is_standing_env[:] = False
    if hasattr(term, "is_heading_env"):
        term.is_heading_env[:] = False


def current_uncertainty(distances: list[float]) -> float:
    window = args_cli.uncertainty_window
    if len(distances) <= window:
        return 0.0
    progress = distances[-1 - window] - distances[-1]
    return float(max(0.0, args_cli.min_progress - progress))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def render_overhead(
    output_path: Path,
    trajectory: list[list[float]],
    step: int,
    goal: tuple[float, float],
) -> None:
    current = trajectory[-1]
    recent = trajectory[max(0, len(trajectory) - 5) :]
    fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=100)
    ax.set_facecolor("white")
    ax.grid(True, color="#d0d0d0", linewidth=0.8, alpha=0.8)
    ax.axhline(0.0, color="#555555", linewidth=1.0, alpha=0.65)
    ax.axvline(0.0, color="#555555", linewidth=1.0, alpha=0.65)
    ox, oy, _ = args_cli.obstacle_pos
    sx, sy, _ = args_cli.obstacle_size
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
    ax.set_title(f"step {step}")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout(pad=0.6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def image_data_url(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def parse_llm_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def fallback_command_from_position(pos: list[float]) -> dict:
    side = -1.0 if pos[1] >= 0.0 else 1.0
    return {
        "reasoning": "Fallback command: turn away from the obstacle and move forward.",
        "vx": 0.5,
        "vy": side,
        "yaw": side,
    }


def call_llm(
    client: OpenAI | None,
    image_path: Path,
    pos: list[float],
    goal: tuple[float, float],
) -> tuple[dict, str | None]:
    if args_cli.dry_run_llm:
        return fallback_command_from_position(pos), None
    user_text = (
        "The image shows an overhead view of the environment.\n"
        f"- Blue circle: robot current position ({pos[0]:.2f}, {pos[1]:.2f})\n"
        f"- Green star: goal position ({goal[0]:.2f}, {goal[1]:.2f})\n"
        "- Gray box: obstacle\n"
        "- Blue line: recent trajectory\n"
        "- Grid shows coordinates in meters\n\n"
        "The robot is STUCK. It cannot reach the goal directly.\n"
        "The direct path is blocked by the obstacle.\n\n"
        "Look at the image carefully.\n"
        "Suggest velocity commands to navigate around the obstacle.\n\n"
        "Think step by step:\n"
        "STEP 1: Where is the robot relative to the obstacle?\n"
        "STEP 2: Which side of the obstacle is easier to go around?\n"
        "STEP 3: What velocity command moves the robot that way?\n\n"
        "Respond in JSON only:\n"
        "{\n"
        '  "reasoning": "one sentence",\n'
        '  "vx": float,\n'
        '  "vy": float,\n'
        '  "yaw": float\n'
        "}"
    )
    response = client.chat.completions.create(
        model=args_cli.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
                ],
            },
        ],
        response_format={"type": "json_object"},
        timeout=args_cli.llm_timeout,
    )
    raw = response.choices[0].message.content or "{}"
    parsed = parse_llm_json(raw)
    parsed["vx"] = clamp(float(parsed.get("vx", 0.0)), 0.0, 1.0)
    parsed["vy"] = clamp(float(parsed.get("vy", 0.0)), -1.0, 1.0)
    parsed["yaw"] = clamp(float(parsed.get("yaw", 0.0)), -1.0, 1.0)
    parsed["reasoning"] = str(parsed.get("reasoning", ""))
    return parsed, raw


def llm_cmd_to_tensor(command: dict, device: torch.device) -> torch.Tensor:
    # Prompt convention: left is negative, right is positive.
    # Isaac velocity command convention: +y and +yaw are left/CCW.
    return torch.tensor([command["vx"], -command["vy"], -command["yaw"]], device=device, dtype=torch.float32)


def plot_success_bars(output_path: Path, summary: dict) -> None:
    labels = list(summary["conditions"].keys())
    values = [summary["conditions"][label]["success_rate"] for label in labels]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=["#4c78a8", "#f58518", "#54a24b"])
    plt.ylim(0.0, 1.0)
    plt.ylabel("success rate")
    plt.title("Step 5 success rate")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_switching_trajectory(output_path: Path, result: dict) -> None:
    trajectory = result["trajectory"]
    xs = [p[0] for p in trajectory]
    ys = [p[1] for p in trajectory]
    plt.figure(figsize=(7, 5))
    plt.plot(xs, ys, linewidth=2.0, label="trajectory")
    for call in result["llm_calls"]:
        idx = min(call["step"], len(trajectory) - 1)
        plt.scatter([trajectory[idx][0]], [trajectory[idx][1]], marker="x", c="purple", s=55)
    ox, oy, _ = args_cli.obstacle_pos
    sx, sy, _ = args_cli.obstacle_size
    plt.gca().add_patch(plt.Rectangle((ox - sx / 2, oy - sy / 2), sx, sy, color="gray", alpha=0.35, label="obstacle"))
    plt.scatter([0.0], [0.0], c="green", s=45, label="start")
    plt.scatter([args_cli.goal_x], [args_cli.goal_y], c="red", marker="*", s=85, label="goal")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("uncertainty_switching trajectory and LLM calls")
    plt.legend(loc="best")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_uncertainty(output_path: Path, result: dict) -> None:
    steps = list(range(len(result["uncertainty"])))
    plt.figure(figsize=(8, 4))
    plt.plot(steps, result["uncertainty"], linewidth=1.8, label="uncertainty")
    plt.axhline(args_cli.uncertainty_threshold, color="red", linestyle="--", linewidth=1.2, label="threshold")
    for call in result["llm_calls"]:
        plt.axvline(call["step"], color="purple", alpha=0.3, linewidth=1.0)
    plt.xlabel("step")
    plt.ylabel("progress uncertainty")
    plt.title("uncertainty_switching uncertainty and triggers")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    output_dir = Path(args_cli.output_dir)
    condition_dir = output_dir / args_cli.condition
    metrics_dir = condition_dir / "metrics"
    images_dir = condition_dir / "llm_images"
    plots_dir = condition_dir / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    load_api_key()
    client = None if args_cli.condition == "policy_only" else OpenAI()

    checkpoint = args_cli.checkpoint or get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
    if not checkpoint:
        raise RuntimeError(f"No pretrained checkpoint available for {args_cli.task}")
    print(f"[CHECKPOINT] {checkpoint}", flush=True)

    if args_cli.num_envs < args_cli.episodes:
        args_cli.num_envs = args_cli.episodes
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    configure_env(env_cfg)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args_cli.device

    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    robot = env.unwrapped.scene["robot"]
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    env_count = args_cli.episodes
    env_origins = env.unwrapped.scene.env_origins[:env_count, :2]
    goal_tuple = (args_cli.goal_x, args_cli.goal_y)
    goal_offsets = torch.tensor([goal_tuple] * env_count, device=env.unwrapped.device)
    goal_world = env_origins + goal_offsets
    trajectories = [[] for _ in range(env_count)]
    distances = [[] for _ in range(env_count)]
    uncertainties = [[] for _ in range(env_count)]
    command_traces = [[] for _ in range(env_count)]
    llm_calls = [[] for _ in range(env_count)]
    override_cmd = torch.zeros((env_count, 3), device=env.unwrapped.device)
    override_until = torch.full((env_count,), -1, dtype=torch.long, device=env.unwrapped.device)
    next_llm_allowed = torch.zeros(env_count, dtype=torch.long, device=env.unwrapped.device)
    success = torch.zeros(env_count, dtype=torch.bool, device=env.unwrapped.device)
    done_seen = torch.zeros_like(success)
    success_steps = torch.full((env_count,), -1, dtype=torch.long, device=env.unwrapped.device)
    total_llm_calls = 0
    llm_errors = []

    obs, _ = env.get_observations()
    for step in range(args_cli.max_steps):
        pos_world = robot.data.root_pos_w[:env_count, :2]
        pos_local = pos_world - env_origins
        dist_tensor = torch.linalg.norm(goal_world - pos_world, dim=1)
        newly_success = (~success) & (dist_tensor < args_cli.success_radius)
        success_steps[newly_success] = step
        success |= newly_success
        active_mask = ~success & ~done_seen

        for env_id in range(env_count):
            if bool(done_seen[env_id].detach().cpu()):
                continue
            point = [float(pos_local[env_id, 0].detach().cpu()), float(pos_local[env_id, 1].detach().cpu())]
            trajectories[env_id].append(point)
            distances[env_id].append(float(dist_tensor[env_id].detach().cpu()))
            uncertainties[env_id].append(current_uncertainty(distances[env_id]))

        if bool((success | done_seen).all()):
            print(f"[RUN] {args_cli.condition} finished at step={step}", flush=True)
            break

        cmd = compute_goal_command(robot, goal_world, active_mask)
        if args_cli.condition != "policy_only":
            for env_id in range(env_count):
                if not bool(active_mask[env_id].detach().cpu()):
                    continue
                should_call = False
                uncertainty = uncertainties[env_id][-1] if uncertainties[env_id] else 0.0
                if args_cli.condition == "always_llm":
                    should_call = step % args_cli.llm_cooldown == 0
                elif args_cli.condition == "uncertainty_switching":
                    should_call = uncertainty >= args_cli.uncertainty_threshold and step >= int(next_llm_allowed[env_id])
                if should_call and total_llm_calls < args_cli.max_llm_calls and trajectories[env_id]:
                    image_path = images_dir / f"episode_{env_id:03d}_step_{step:04d}.png"
                    render_overhead(image_path, trajectories[env_id], step, goal_tuple)
                    try:
                        llm_response, raw = call_llm(client, image_path, trajectories[env_id][-1], goal_tuple)
                    except Exception as exc:
                        llm_response = fallback_command_from_position(trajectories[env_id][-1])
                        raw = None
                        llm_errors.append({"episode": env_id, "step": step, "error": repr(exc)})
                        print(f"[LLM_ERROR] episode={env_id} step={step} error={exc!r}", flush=True)
                    override_cmd[env_id] = llm_cmd_to_tensor(llm_response, env.unwrapped.device)
                    override_until[env_id] = step + args_cli.llm_duration
                    next_llm_allowed[env_id] = step + args_cli.llm_cooldown
                    call_record = {
                        "episode": env_id,
                        "step": step,
                        "uncertainty": uncertainty,
                        "image_path": str(image_path),
                        "response": llm_response,
                        "raw_response": raw,
                    }
                    llm_calls[env_id].append(call_record)
                    total_llm_calls += 1
                    print(
                        f"[LLM_CALL] condition={args_cli.condition} episode={env_id} step={step} "
                        f"cmd=({llm_response['vx']:.2f},{llm_response['vy']:.2f},{llm_response['yaw']:.2f})",
                        flush=True,
                    )
        override_mask = (torch.arange(env_count, device=env.unwrapped.device) < env_count) & (step < override_until)
        override_mask &= active_mask
        cmd[:env_count] = torch.where(override_mask.unsqueeze(1), override_cmd, cmd[:env_count])
        set_command(env, cmd)
        obs, _ = env.get_observations()
        for env_id in range(env_count):
            command_traces[env_id].append(cmd[env_id].detach().cpu().tolist())
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
        done_seen |= dones[:env_count].bool()
        if (step + 1) % 100 == 0:
            print(
                f"[RUN] {args_cli.condition} step={step + 1} success={int(success.sum().detach().cpu())}/{env_count} "
                f"llm_calls={total_llm_calls}",
                flush=True,
            )

    env.close()
    results = []
    for episode in range(env_count):
        episode_success = bool(success[episode].detach().cpu())
        steps = int(success_steps[episode].detach().cpu()) if episode_success else len(trajectories[episode])
        if steps < 0:
            steps = len(trajectories[episode])
        result = {
            "episode": episode,
            "condition": args_cli.condition,
            "success": episode_success,
            "steps": steps,
            "final_distance": distances[episode][-1] if distances[episode] else None,
            "min_distance": min(distances[episode]) if distances[episode] else None,
            "final_pos": trajectories[episode][-1] if trajectories[episode] else None,
            "trajectory": trajectories[episode],
            "distances": distances[episode],
            "uncertainty": uncertainties[episode],
            "commands": command_traces[episode],
            "llm_calls": llm_calls[episode],
        }
        results.append(result)

    success_rate = sum(1 for r in results if r["success"]) / len(results)
    mean_episode_length = sum(r["steps"] for r in results) / len(results)
    mean_final_distance = sum(r["final_distance"] for r in results if r["final_distance"] is not None) / len(results)
    summary = {
        "condition": args_cli.condition,
        "task": args_cli.task,
        "episodes": env_count,
        "max_steps": args_cli.max_steps,
        "goal": [args_cli.goal_x, args_cli.goal_y],
        "obstacle_pos": list(args_cli.obstacle_pos),
        "obstacle_size": list(args_cli.obstacle_size),
        "uncertainty_threshold": args_cli.uncertainty_threshold,
        "llm_model": args_cli.llm_model if args_cli.condition != "policy_only" else None,
        "llm_cooldown": args_cli.llm_cooldown,
        "llm_duration": args_cli.llm_duration,
        "success_rate": success_rate,
        "mean_episode_length": mean_episode_length,
        "mean_final_distance": mean_final_distance,
        "total_llm_calls": total_llm_calls,
        "llm_call_rate": total_llm_calls / max(1, env_count * args_cli.max_steps),
        "llm_errors": llm_errors,
        "results": results,
    }
    summary_path = metrics_dir / f"{args_cli.condition}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[SUMMARY] {args_cli.condition} success_rate={success_rate:.3f} "
        f"mean_len={mean_episode_length:.1f} mean_final_distance={mean_final_distance:.3f} "
        f"llm_calls={total_llm_calls} errors={len(llm_errors)}",
        flush=True,
    )
    print(f"[WROTE] {summary_path}", flush=True)
    if args_cli.condition == "uncertainty_switching":
        plot_switching_trajectory(plots_dir / "uncertainty_switching_trajectory.png", results[0])
        plot_uncertainty(plots_dir / "uncertainty_switching_uncertainty.png", results[0])


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
