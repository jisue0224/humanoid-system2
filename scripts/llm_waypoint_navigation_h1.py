#!/usr/bin/env python3
"""LLM-assisted H1 obstacle navigation using intermediate waypoints."""

import argparse
import base64
import json
import os
import re
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run H1 LLM waypoint navigation condition.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-H1-v0")
parser.add_argument("--condition", choices=("policy_only", "always_llm", "uncertainty_switching"), required=True)
parser.add_argument("--num_envs", type=int, default=20)
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--max_steps", type=int, default=500)
parser.add_argument("--success_radius", type=float, default=0.5)
parser.add_argument("--waypoint_radius", type=float, default=0.8)
parser.add_argument("--waypoint_timeout", type=int, default=320)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--output_dir", type=str, default="experiments/llm_waypoint_navigation")
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
parser.add_argument("--llm_model", type=str, default="gpt-5.4")
parser.add_argument("--llm_timeout", type=float, default=60.0)
parser.add_argument("--max_llm_calls", type=int, default=100000)
parser.add_argument("--disable_waypoint_postprocess", action="store_true")
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
    "You are a navigation planner for a humanoid robot.\n"
    "You receive an overhead view of the environment and choose one safe intermediate waypoint.\n"
    "The low-level walking controller will walk toward your waypoint."
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


def compute_goal_command(robot, target_xy: torch.Tensor, active_mask: torch.Tensor, stop_radius: float) -> torch.Tensor:
    pos = robot.data.root_pos_w[:, :2]
    delta = target_xy - pos
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
    cmd = torch.where(distance.unsqueeze(1) < stop_radius, torch.zeros_like(cmd), cmd)
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
    active_waypoint: list[float] | None = None,
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
    if active_waypoint is not None:
        ax.scatter([active_waypoint[0]], [active_waypoint[1]], s=130, color="#9467bd", marker="D", label="active waypoint")
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


def fallback_waypoint_from_position(pos: list[float]) -> dict:
    ox, oy, _ = args_cli.obstacle_pos
    sx, sy, _ = args_cli.obstacle_size
    side = -1.0 if pos[1] <= oy else 1.0
    lateral_clearance = abs(pos[1] - oy)
    if lateral_clearance < sy / 2 + 0.35:
        waypoint_x = min(pos[0] - 0.25, ox - sx / 2 - 0.35)
        reasoning = "Fallback waypoint: first move laterally away from the obstacle front before resuming toward the goal."
    else:
        waypoint_x = ox + sx / 2 + 0.75
        reasoning = "Fallback waypoint: stay on the same cleared side and move past the obstacle before returning to the goal."
    return {
        "reasoning": reasoning,
        "waypoint_x": waypoint_x,
        "waypoint_y": oy + side * 1.35,
    }


def call_llm_waypoint(
    client: OpenAI | None,
    image_path: Path,
    pos: list[float],
    goal: tuple[float, float],
    active_waypoint: list[float] | None,
) -> tuple[dict, str | None]:
    if args_cli.dry_run_llm:
        return fallback_waypoint_from_position(pos), None
    ox, oy, _ = args_cli.obstacle_pos
    sx, sy, _ = args_cli.obstacle_size
    waypoint_text = "No active waypoint." if active_waypoint is None else f"Active waypoint: ({active_waypoint[0]:.2f}, {active_waypoint[1]:.2f})."
    user_text = (
        "The image shows an overhead view of the environment.\n"
        f"- Blue circle: robot current position ({pos[0]:.2f}, {pos[1]:.2f})\n"
        f"- Green star: final goal position ({goal[0]:.2f}, {goal[1]:.2f})\n"
        "- Gray box: obstacle\n"
        "- Blue line: recent trajectory\n"
        "- Grid shows coordinates in meters\n"
        f"- Obstacle center: ({ox:.2f}, {oy:.2f}), size: ({sx:.2f}, {sy:.2f})\n"
        f"- {waypoint_text}\n\n"
        "The robot is stuck because the direct path is blocked.\n"
        "Choose ONE intermediate waypoint that is safely outside the gray obstacle and gives the walking controller a clear detour.\n"
        "The walking controller turns and walks forward toward the waypoint; it cannot sidestep in place.\n"
        "If the robot is touching or close to the obstacle front, do not put the waypoint beyond the obstacle front face.\n"
        "In that case, first choose a waypoint beside the robot or slightly behind it, with at least 0.6 m lateral clearance from the obstacle.\n"
        "If the robot already has lateral clearance above or below the obstacle, keep it on that same side and choose a waypoint past the obstacle.\n"
        "Do not switch from the lower side to the upper side, or from the upper side to the lower side, after clearance has been created.\n"
        "Do not choose a waypoint inside the obstacle.\n"
        "After reaching your waypoint, the robot will resume walking to the final goal.\n\n"
        "Think step by step:\n"
        "STEP 1: Where is the robot relative to the obstacle?\n"
        "STEP 2: Which side of the obstacle gives the clearest detour?\n"
        "STEP 3: What waypoint should the robot walk toward first?\n\n"
        "Respond in JSON only:\n"
        "{\n"
        '  "reasoning": "one sentence",\n'
        '  "waypoint_x": float,\n'
        '  "waypoint_y": float\n'
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
    parsed["waypoint_x"] = clamp(float(parsed.get("waypoint_x", ox)), -0.25, 5.25)
    parsed["waypoint_y"] = clamp(float(parsed.get("waypoint_y", oy)), -1.8, 1.8)
    parsed["reasoning"] = str(parsed.get("reasoning", ""))
    return parsed, raw


def postprocess_waypoint(response: dict, pos: list[float]) -> tuple[list[float], str | None]:
    waypoint = [float(response["waypoint_x"]), float(response["waypoint_y"])]
    if args_cli.disable_waypoint_postprocess:
        return waypoint, None

    ox, oy, _ = args_cli.obstacle_pos
    sx, sy, _ = args_cli.obstacle_size
    front_x = ox - sx / 2
    back_x = ox + sx / 2
    lateral_clearance = abs(pos[1] - oy)
    needed_clearance = sy / 2 + 0.35

    waypoint_inside_x = front_x <= waypoint[0] <= back_x
    waypoint_inside_y = (oy - sy / 2) <= waypoint[1] <= (oy + sy / 2)
    if waypoint_inside_x and waypoint_inside_y:
        side = -1.0 if waypoint[1] <= oy else 1.0
        waypoint[1] = oy + side * 1.35
        return waypoint, "moved waypoint out of obstacle"

    if lateral_clearance < needed_clearance and pos[0] < front_x + 0.25:
        side = -1.0 if waypoint[1] <= oy else 1.0
        waypoint[0] = min(pos[0] - 0.25, front_x - 0.35)
        waypoint[1] = oy + side * 1.35
        return waypoint, "forced first-stage lateral escape before obstacle front"

    current_side = -1.0 if pos[1] <= oy else 1.0
    waypoint_side = -1.0 if waypoint[1] <= oy else 1.0
    if lateral_clearance >= needed_clearance and pos[0] < back_x + 0.5:
        if waypoint[0] < back_x + 0.5 or waypoint_side != current_side:
            waypoint[0] = back_x + 0.75
            waypoint[1] = oy + current_side * 1.35
            return waypoint, "forced second-stage waypoint past obstacle on same side"

    return waypoint, None


def plot_waypoint_trajectory(output_path: Path, result: dict) -> None:
    trajectory = result["trajectory"]
    xs = [p[0] for p in trajectory]
    ys = [p[1] for p in trajectory]
    plt.figure(figsize=(7, 5))
    plt.plot(xs, ys, linewidth=2.0, label="trajectory")
    for call in result["llm_calls"]:
        idx = min(call["step"], len(trajectory) - 1)
        plt.scatter([trajectory[idx][0]], [trajectory[idx][1]], marker="x", c="purple", s=55)
        wp = call["waypoint"]
        plt.scatter([wp[0]], [wp[1]], marker="D", c="orange", s=45)
    ox, oy, _ = args_cli.obstacle_pos
    sx, sy, _ = args_cli.obstacle_size
    plt.gca().add_patch(plt.Rectangle((ox - sx / 2, oy - sy / 2), sx, sy, color="gray", alpha=0.35, label="obstacle"))
    plt.scatter([0.0], [0.0], c="green", s=45, label="start")
    plt.scatter([args_cli.goal_x], [args_cli.goal_y], c="red", marker="*", s=85, label="goal")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("waypoint navigation trajectory")
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
    plt.title("waypoint navigation uncertainty and triggers")
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
    final_goal_tuple = (args_cli.goal_x, args_cli.goal_y)
    final_goal_offsets = torch.tensor([final_goal_tuple] * env_count, device=env.unwrapped.device)
    final_goal_world = env_origins + final_goal_offsets
    waypoint_local = torch.zeros((env_count, 2), device=env.unwrapped.device)
    waypoint_active = torch.zeros(env_count, dtype=torch.bool, device=env.unwrapped.device)
    waypoint_started = torch.full((env_count,), -1, dtype=torch.long, device=env.unwrapped.device)
    next_llm_allowed = torch.zeros(env_count, dtype=torch.long, device=env.unwrapped.device)

    trajectories = [[] for _ in range(env_count)]
    distances = [[] for _ in range(env_count)]
    uncertainties = [[] for _ in range(env_count)]
    command_traces = [[] for _ in range(env_count)]
    target_traces = [[] for _ in range(env_count)]
    llm_calls = [[] for _ in range(env_count)]
    success = torch.zeros(env_count, dtype=torch.bool, device=env.unwrapped.device)
    done_seen = torch.zeros_like(success)
    success_steps = torch.full((env_count,), -1, dtype=torch.long, device=env.unwrapped.device)
    total_llm_calls = 0
    llm_errors = []

    obs, _ = env.get_observations()
    for step in range(args_cli.max_steps):
        pos_world = robot.data.root_pos_w[:env_count, :2]
        pos_local = pos_world - env_origins
        final_dist_tensor = torch.linalg.norm(final_goal_world - pos_world, dim=1)
        newly_success = (~success) & (final_dist_tensor < args_cli.success_radius)
        success_steps[newly_success] = step
        success |= newly_success
        active_mask = ~success & ~done_seen

        waypoint_world = env_origins + waypoint_local
        waypoint_dist = torch.linalg.norm(waypoint_world - pos_world, dim=1)
        reached_waypoint = waypoint_active & (waypoint_dist < args_cli.waypoint_radius)
        timed_out_waypoint = waypoint_active & ((step - waypoint_started) >= args_cli.waypoint_timeout)
        waypoint_active &= ~(reached_waypoint | timed_out_waypoint | success | done_seen)

        for env_id in range(env_count):
            if bool(done_seen[env_id].detach().cpu()):
                continue
            point = [float(pos_local[env_id, 0].detach().cpu()), float(pos_local[env_id, 1].detach().cpu())]
            trajectories[env_id].append(point)
            distances[env_id].append(float(final_dist_tensor[env_id].detach().cpu()))
            uncertainties[env_id].append(current_uncertainty(distances[env_id]))

        if bool((success | done_seen).all()):
            print(f"[RUN] {args_cli.condition} finished at step={step}", flush=True)
            break

        if args_cli.condition != "policy_only":
            for env_id in range(env_count):
                if not bool(active_mask[env_id].detach().cpu()):
                    continue
                uncertainty = uncertainties[env_id][-1] if uncertainties[env_id] else 0.0
                should_call = False
                if args_cli.condition == "always_llm":
                    should_call = step % args_cli.llm_cooldown == 0
                elif args_cli.condition == "uncertainty_switching":
                    should_call = (
                        uncertainty >= args_cli.uncertainty_threshold
                        and step >= int(next_llm_allowed[env_id])
                        and not bool(waypoint_active[env_id].detach().cpu())
                    )
                if should_call and total_llm_calls < args_cli.max_llm_calls and trajectories[env_id]:
                    active_wp = None
                    if bool(waypoint_active[env_id].detach().cpu()):
                        active_wp = waypoint_local[env_id].detach().cpu().tolist()
                    image_path = images_dir / f"episode_{env_id:03d}_step_{step:04d}.png"
                    render_overhead(image_path, trajectories[env_id], step, final_goal_tuple, active_wp)
                    try:
                        llm_response, raw = call_llm_waypoint(
                            client,
                            image_path,
                            trajectories[env_id][-1],
                            final_goal_tuple,
                            active_wp,
                        )
                    except Exception as exc:
                        llm_response = fallback_waypoint_from_position(trajectories[env_id][-1])
                        raw = None
                        llm_errors.append({"episode": env_id, "step": step, "error": repr(exc)})
                        print(f"[LLM_ERROR] episode={env_id} step={step} error={exc!r}", flush=True)
                    raw_waypoint = [llm_response["waypoint_x"], llm_response["waypoint_y"]]
                    waypoint, postprocess_reason = postprocess_waypoint(llm_response, trajectories[env_id][-1])
                    waypoint_local[env_id] = torch.tensor(waypoint, device=env.unwrapped.device)
                    waypoint_active[env_id] = True
                    waypoint_started[env_id] = step
                    next_llm_allowed[env_id] = step + args_cli.llm_cooldown
                    call_record = {
                        "episode": env_id,
                        "step": step,
                        "uncertainty": uncertainty,
                        "image_path": str(image_path),
                        "raw_waypoint": raw_waypoint,
                        "waypoint": waypoint,
                        "postprocess_reason": postprocess_reason,
                        "response": llm_response,
                        "raw_response": raw,
                    }
                    llm_calls[env_id].append(call_record)
                    total_llm_calls += 1
                    print(
                        f"[LLM_CALL] condition={args_cli.condition} episode={env_id} step={step} "
                        f"raw=({raw_waypoint[0]:.2f},{raw_waypoint[1]:.2f}) "
                        f"waypoint=({waypoint[0]:.2f},{waypoint[1]:.2f})",
                        flush=True,
                    )

        waypoint_world = env_origins + waypoint_local
        target_world = torch.where(waypoint_active.unsqueeze(1), waypoint_world, final_goal_world)
        stop_radii = torch.where(
            waypoint_active,
            torch.full_like(final_dist_tensor, args_cli.waypoint_radius),
            torch.full_like(final_dist_tensor, args_cli.success_radius),
        )
        cmd = compute_goal_command(robot, target_world, active_mask, stop_radius=0.0)
        target_dist = torch.linalg.norm(target_world - pos_world, dim=1)
        cmd = torch.where((target_dist > stop_radii).unsqueeze(1), cmd, torch.zeros_like(cmd))
        set_command(env, cmd)
        obs, _ = env.get_observations()
        for env_id in range(env_count):
            command_traces[env_id].append(cmd[env_id].detach().cpu().tolist())
            target_traces[env_id].append((target_world[env_id] - env_origins[env_id]).detach().cpu().tolist())
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
        done_seen |= dones[:env_count].bool()
        if (step + 1) % 100 == 0:
            print(
                f"[RUN] {args_cli.condition} step={step + 1} success={int(success.sum().detach().cpu())}/{env_count} "
                f"llm_calls={total_llm_calls} active_waypoints={int(waypoint_active.sum().detach().cpu())}",
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
            "targets": target_traces[episode],
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
        "waypoint_radius": args_cli.waypoint_radius,
        "waypoint_timeout": args_cli.waypoint_timeout,
        "waypoint_postprocess": not args_cli.disable_waypoint_postprocess,
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
        plot_waypoint_trajectory(plots_dir / "uncertainty_switching_waypoint_trajectory.png", results[0])
        plot_uncertainty(plots_dir / "uncertainty_switching_waypoint_uncertainty.png", results[0])
    elif args_cli.condition == "always_llm":
        plot_waypoint_trajectory(plots_dir / "always_llm_waypoint_trajectory.png", results[0])


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
