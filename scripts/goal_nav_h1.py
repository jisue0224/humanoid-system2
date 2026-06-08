#!/usr/bin/env python3
"""Goal navigation with H1 pretrained velocity policy and a rule-based high-level controller."""

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run goal-navigation episodes with H1 velocity policy.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-H1-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=500)
parser.add_argument("--episodes", type=int, default=30)
parser.add_argument("--success_radius", type=float, default=0.5)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--output_dir", type=str, default="experiments/goal_nav")
parser.add_argument("--goal_set", type=str, default="default")
parser.add_argument("--max_vx", type=float, default=1.0)
parser.add_argument("--yaw_gain", type=float, default=1.5)
parser.add_argument("--yaw_limit", type=float, default=1.0)
parser.add_argument("--forward_angle_gate", type=float, default=1.2)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg


DEFAULT_GOALS = [(5.0, 0.0), (0.0, 5.0), (5.0, 5.0), (-3.0, 4.0)]


def wrap_pi_tensor(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def wrap_pi_float(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def configure_env(env_cfg) -> None:
    cmd = env_cfg.commands.base_velocity
    cmd.heading_command = False
    cmd.rel_heading_envs = 0.0
    cmd.rel_standing_envs = 0.0
    cmd.debug_vis = False
    cmd.resampling_time_range = (1000.0, 1000.0)
    cmd.ranges.lin_vel_x = (-1.0, 1.0)
    cmd.ranges.lin_vel_y = (-0.2, 0.2)
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


def compute_velocity_command(robot, goal_xy: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
    pos = robot.data.root_pos_w[:, :2]
    delta = goal_xy - pos
    distance = torch.linalg.norm(delta, dim=1)
    target_heading = torch.atan2(delta[:, 1], delta[:, 0])
    heading = robot.data.heading_w
    heading_error = wrap_pi_tensor(target_heading - heading)

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


def plot_trajectory(output_path: Path, trajectory: list[list[float]], goal: tuple[float, float], success: bool) -> None:
    xs = [p[0] for p in trajectory]
    ys = [p[1] for p in trajectory]
    plt.figure(figsize=(5, 5))
    plt.plot(xs, ys, linewidth=2, label="H1 trajectory")
    plt.scatter([xs[0]], [ys[0]], c="green", s=50, label="start")
    plt.scatter([goal[0]], [goal[1]], c="red", s=70, marker="*", label="goal")
    circle = plt.Circle(goal, args_cli.success_radius, color="red", fill=False, linestyle="--", alpha=0.6)
    plt.gca().add_patch(circle)
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(f"Goal {goal} - {'success' if success else 'failure'}")
    plt.legend(loc="best")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    output_dir = Path(args_cli.output_dir)
    metrics_dir = output_dir / "metrics"
    plots_dir = output_dir / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = args_cli.checkpoint or get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
    if not checkpoint:
        raise RuntimeError(f"No pretrained checkpoint available for {args_cli.task}")
    print(f"[GOAL_NAV] checkpoint={checkpoint}", flush=True)

    num_envs = max(args_cli.num_envs, args_cli.episodes)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=num_envs)
    configure_env(env_cfg)
    env_cfg.scene.num_envs = num_envs
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args_cli.device

    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    robot = env.unwrapped.scene["robot"]
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    goals = [DEFAULT_GOALS[i % len(DEFAULT_GOALS)] for i in range(args_cli.episodes)]
    env_origins = env.unwrapped.scene.env_origins[: args_cli.episodes, :2]
    goal_offsets = torch.tensor(goals, device=env.unwrapped.device, dtype=torch.float32)
    goal_world = env_origins + goal_offsets
    trajectories: list[list[list[float]]] = [[] for _ in range(args_cli.episodes)]
    command_traces: list[list[list[float]]] = [[] for _ in range(args_cli.episodes)]
    success = torch.zeros(args_cli.episodes, dtype=torch.bool, device=env.unwrapped.device)
    done_seen = torch.zeros_like(success)
    success_steps = torch.full((args_cli.episodes,), -1, dtype=torch.long, device=env.unwrapped.device)

    obs, _ = env.get_observations()
    for step in range(args_cli.max_steps):
        pos_world = robot.data.root_pos_w[: args_cli.episodes, :2]
        pos_local = pos_world - env_origins
        distances = torch.linalg.norm(goal_world - pos_world, dim=1)
        newly_success = (~success) & (distances < args_cli.success_radius)
        success_steps[newly_success] = step
        success |= newly_success
        active_mask = ~success & ~done_seen

        for env_id in range(args_cli.episodes):
            trajectories[env_id].append([float(pos_local[env_id, 0].detach().cpu()), float(pos_local[env_id, 1].detach().cpu())])

        if bool(success.all()):
            print(f"[GOAL_NAV] all environments succeeded at step={step}", flush=True)
            break

        cmd = compute_velocity_command(robot, goal_world, active_mask)
        set_command(env, cmd)
        obs, _ = env.get_observations()
        for env_id in range(args_cli.episodes):
            command_traces[env_id].append(cmd[env_id].detach().cpu().tolist())
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
        done_seen |= dones[: args_cli.episodes].bool()
        if (step + 1) % 100 == 0:
            print(
                f"[GOAL_NAV] step={step + 1} successes={int(success.sum().detach().cpu())}/{args_cli.episodes}",
                flush=True,
            )

    results = []
    final_pos_world = robot.data.root_pos_w[: args_cli.episodes, :2]
    final_pos_local = final_pos_world - env_origins
    final_distances = torch.linalg.norm(goal_world - final_pos_world, dim=1)
    final_headings = robot.data.heading_w[: args_cli.episodes]
    for episode in range(args_cli.episodes):
        goal = goals[episode]
        episode_success = bool(success[episode].detach().cpu())
        plot_path = plots_dir / f"episode_{episode:03d}_goal_{goal[0]:.1f}_{goal[1]:.1f}.png"
        plot_trajectory(plot_path, trajectories[episode], goal, episode_success)
        steps = int(success_steps[episode].detach().cpu()) if episode_success else len(trajectories[episode])
        if steps < 0:
            steps = len(trajectories[episode])
        result = {
            "episode": episode,
            "goal": list(goal),
            "success": episode_success,
            "steps": steps,
            "duration_s": steps * raw_env.unwrapped.step_dt,
            "final_distance": float(final_distances[episode].detach().cpu()),
            "final_heading": float(final_headings[episode].detach().cpu()),
            "final_pos": [
                float(final_pos_local[episode, 0].detach().cpu()),
                float(final_pos_local[episode, 1].detach().cpu()),
            ],
            "plot": str(plot_path),
            "trajectory": trajectories[episode],
            "commands": command_traces[episode],
        }
        results.append(result)
        print(
            f"[GOAL_RESULT] episode={episode} goal={goal} success={episode_success} "
            f"steps={result['steps']} final_distance={result['final_distance']:.3f} final_pos={result['final_pos']}",
            flush=True,
        )

    success_rate = sum(1 for r in results if r["success"]) / len(results)
    summary = {
        "task": args_cli.task,
        "episodes": args_cli.episodes,
        "max_steps": args_cli.max_steps,
        "success_radius": args_cli.success_radius,
        "success_rate": success_rate,
        "goals": [list(g) for g in goals],
        "results": results,
    }
    (metrics_dir / "goal_nav_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[GOAL_SUMMARY] success_rate={success_rate:.3f} episodes={args_cli.episodes}", flush=True)
    print(f"[GOAL_SUMMARY] wrote={metrics_dir / 'goal_nav_summary.json'}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
