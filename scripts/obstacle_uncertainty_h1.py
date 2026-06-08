#!/usr/bin/env python3
"""Compare H1 goal navigation with and without a static obstacle and progress uncertainty."""

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run H1 obstacle and uncertainty rollouts.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-H1-v0")
parser.add_argument("--num_envs", type=int, default=10)
parser.add_argument("--episodes_per_scenario", type=int, default=10)
parser.add_argument("--max_steps", type=int, default=700)
parser.add_argument("--success_radius", type=float, default=0.5)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--output_dir", type=str, default="experiments/obstacle_uncertainty")
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
parser.add_argument("--trigger_cooldown", type=int, default=20)
parser.add_argument("--scenario", choices=("no_obstacle", "obstacle"), default="no_obstacle")
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

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg


def configure_env(env_cfg, with_obstacle: bool) -> None:
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
    if getattr(env_cfg.terminations, "base_contact", None) is not None:
        env_cfg.terminations.base_contact = None

    if with_obstacle:
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


def wrap_pi_tensor(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def compute_velocity_command(robot, goal_xy: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
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


def progress_uncertainty(distances: list[float], window: int) -> list[float]:
    values = []
    for idx, distance in enumerate(distances):
        if idx < window:
            values.append(0.0)
        else:
            progress = distances[idx - window] - distance
            values.append(float(max(0.0, args_cli.min_progress - progress)))
    return values


def cooldown_triggers(uncertainty: list[float], threshold: float, cooldown: int) -> list[int]:
    triggers = []
    next_allowed = 0
    for idx, value in enumerate(uncertainty):
        if idx >= next_allowed and value >= threshold and value > 0.0:
            triggers.append(idx)
            next_allowed = idx + cooldown
    return triggers


def obstacle_overlap_xy(point: list[float], obstacle_pos: tuple[float, float, float], obstacle_size: tuple[float, float, float]) -> bool:
    half_x = obstacle_size[0] * 0.5
    half_y = obstacle_size[1] * 0.5
    return abs(point[0] - obstacle_pos[0]) <= half_x and abs(point[1] - obstacle_pos[1]) <= half_y


def plot_scenario_trajectories(
    output_path: Path,
    results: list[dict],
    goal: tuple[float, float],
    with_obstacle: bool,
    title: str,
) -> None:
    plt.figure(figsize=(6, 5))
    for result in results:
        xs = [p[0] for p in result["trajectory"]]
        ys = [p[1] for p in result["trajectory"]]
        plt.plot(xs, ys, linewidth=1.5, alpha=0.75)
    plt.scatter([0.0], [0.0], c="green", s=50, label="start")
    plt.scatter([goal[0]], [goal[1]], c="red", s=70, marker="*", label="goal")
    if with_obstacle:
        ox, oy, _ = args_cli.obstacle_pos
        sx, sy, _ = args_cli.obstacle_size
        rect = plt.Rectangle((ox - sx / 2, oy - sy / 2), sx, sy, color="black", alpha=0.25, label="obstacle")
        plt.gca().add_patch(rect)
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_uncertainty(
    output_path: Path,
    result: dict,
    threshold: float,
    title: str,
) -> None:
    steps = list(range(len(result["uncertainty"])))
    plt.figure(figsize=(8, 4))
    plt.plot(steps, result["uncertainty"], linewidth=1.8, label="uncertainty")
    plt.axhline(threshold, color="red", linestyle="--", linewidth=1.2, label="90th percentile threshold")
    for trigger_step in result["trigger_steps"]:
        plt.axvline(trigger_step, color="purple", alpha=0.35, linewidth=1.0)
    plt.xlabel("step")
    plt.ylabel("progress uncertainty")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_comparison(output_path: Path, scenario_results: dict[str, dict]) -> None:
    plt.figure(figsize=(7, 5))
    for scenario_name, summary in scenario_results.items():
        result = summary["results"][0]
        xs = [p[0] for p in result["trajectory"]]
        ys = [p[1] for p in result["trajectory"]]
        plt.plot(xs, ys, linewidth=2.0, label=scenario_name)
    goal = (args_cli.goal_x, args_cli.goal_y)
    plt.scatter([0.0], [0.0], c="green", s=50, label="start")
    plt.scatter([goal[0]], [goal[1]], c="red", s=70, marker="*", label="goal")
    ox, oy, _ = args_cli.obstacle_pos
    sx, sy, _ = args_cli.obstacle_size
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


def run_scenario(name: str, with_obstacle: bool, checkpoint: str, output_dir: Path) -> dict:
    print(f"[SCENARIO] {name} obstacle={with_obstacle}", flush=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    configure_env(env_cfg, with_obstacle)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args_cli.device

    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    robot = env.unwrapped.scene["robot"]
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    env_count = args_cli.episodes_per_scenario
    env_origins = env.unwrapped.scene.env_origins[:env_count, :2]
    goal_offsets = torch.tensor([[args_cli.goal_x, args_cli.goal_y]] * env_count, device=env.unwrapped.device)
    goal_world = env_origins + goal_offsets
    trajectories = [[] for _ in range(env_count)]
    distance_traces = [[] for _ in range(env_count)]
    command_traces = [[] for _ in range(env_count)]
    inside_obstacle_steps = [0 for _ in range(env_count)]
    success = torch.zeros(env_count, dtype=torch.bool, device=env.unwrapped.device)
    done_seen = torch.zeros_like(success)
    success_steps = torch.full((env_count,), -1, dtype=torch.long, device=env.unwrapped.device)
    done_steps = torch.full((env_count,), -1, dtype=torch.long, device=env.unwrapped.device)

    obs, _ = env.get_observations()
    for step in range(args_cli.max_steps):
        pos_world = robot.data.root_pos_w[:env_count, :2]
        pos_local = pos_world - env_origins
        distances = torch.linalg.norm(goal_world - pos_world, dim=1)
        newly_success = (~success) & (distances < args_cli.success_radius)
        success_steps[newly_success] = step
        success |= newly_success
        active_mask = ~success & ~done_seen

        for env_id in range(env_count):
            if bool(done_seen[env_id].detach().cpu()):
                continue
            point = [float(pos_local[env_id, 0].detach().cpu()), float(pos_local[env_id, 1].detach().cpu())]
            trajectories[env_id].append(point)
            distance_traces[env_id].append(float(distances[env_id].detach().cpu()))
            if with_obstacle and obstacle_overlap_xy(point, tuple(args_cli.obstacle_pos), tuple(args_cli.obstacle_size)):
                inside_obstacle_steps[env_id] += 1

        if bool((success | done_seen).all()):
            print(f"[SCENARIO] {name} finished at step={step}", flush=True)
            break

        cmd = compute_velocity_command(robot, goal_world, active_mask)
        set_command(env, cmd)
        obs, _ = env.get_observations()
        for env_id in range(env_count):
            command_traces[env_id].append(cmd[env_id].detach().cpu().tolist())
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
        dones = dones[:env_count].bool()
        newly_done = (~done_seen) & dones
        done_steps[newly_done] = step + 1
        done_seen |= dones
        if (step + 1) % 100 == 0:
            print(f"[SCENARIO] {name} step={step + 1} success={int(success.sum().cpu())}/{env_count}", flush=True)

    env.close()

    all_uncertainty = []
    results = []
    for episode in range(env_count):
        uncertainty = progress_uncertainty(distance_traces[episode], args_cli.uncertainty_window)
        all_uncertainty.extend(uncertainty)
        final_distance = distance_traces[episode][-1]
        final_pos = trajectories[episode][-1]
        episode_success = bool(success[episode].detach().cpu())
        episode_done = bool(done_seen[episode].detach().cpu())
        min_distance = min(distance_traces[episode])
        delta_last_100 = distance_traces[episode][max(0, len(distance_traces[episode]) - 101)] - final_distance
        result = {
            "episode": episode,
            "scenario": name,
            "with_obstacle": with_obstacle,
            "goal": [args_cli.goal_x, args_cli.goal_y],
            "success": episode_success,
            "done": episode_done,
            "steps": int(success_steps[episode].detach().cpu()) if episode_success else len(trajectories[episode]),
            "done_step": int(done_steps[episode].detach().cpu()),
            "final_distance": final_distance,
            "min_distance": min_distance,
            "final_pos": final_pos,
            "delta_distance_last_100_steps": delta_last_100,
            "inside_obstacle_steps": inside_obstacle_steps[episode],
            "trajectory": trajectories[episode],
            "distances": distance_traces[episode],
            "uncertainty": uncertainty,
            "commands": command_traces[episode],
        }
        results.append(result)

    threshold = float(torch.quantile(torch.tensor(all_uncertainty), 0.9).item()) if all_uncertainty else 0.0
    for result in results:
        result["trigger_steps"] = cooldown_triggers(result["uncertainty"], threshold, args_cli.trigger_cooldown)
        result["max_uncertainty"] = max(result["uncertainty"]) if result["uncertainty"] else 0.0
        result["mean_uncertainty"] = sum(result["uncertainty"]) / len(result["uncertainty"]) if result["uncertainty"] else 0.0

    plots_dir = output_dir / "plots"
    plot_scenario_trajectories(
        plots_dir / f"{name}_trajectories.png",
        results,
        (args_cli.goal_x, args_cli.goal_y),
        with_obstacle,
        f"{name}: trajectories",
    )
    plot_uncertainty(
        plots_dir / f"{name}_uncertainty_episode_000.png",
        results[0],
        threshold,
        f"{name}: uncertainty over time",
    )
    summary = {
        "scenario": name,
        "with_obstacle": with_obstacle,
        "episodes": env_count,
        "success_rate": sum(1 for r in results if r["success"]) / len(results),
        "done_rate": sum(1 for r in results if r["done"]) / len(results),
        "uncertainty_threshold_p90": threshold,
        "mean_max_uncertainty": sum(r["max_uncertainty"] for r in results) / len(results),
        "mean_uncertainty": sum(r["mean_uncertainty"] for r in results) / len(results),
        "mean_final_distance": sum(r["final_distance"] for r in results) / len(results),
        "mean_min_distance": sum(r["min_distance"] for r in results) / len(results),
        "mean_inside_obstacle_steps": sum(r["inside_obstacle_steps"] for r in results) / len(results),
        "obstacle_pos": list(args_cli.obstacle_pos) if with_obstacle else None,
        "obstacle_size": list(args_cli.obstacle_size) if with_obstacle else None,
        "results": results,
    }
    print(
        f"[SUMMARY] {name} success_rate={summary['success_rate']:.3f} done_rate={summary['done_rate']:.3f} "
        f"threshold={threshold:.4f} mean_final_distance={summary['mean_final_distance']:.3f} "
        f"mean_inside_obstacle_steps={summary['mean_inside_obstacle_steps']:.1f}",
        flush=True,
    )
    return summary


def main() -> None:
    output_dir = Path(args_cli.output_dir)
    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    checkpoint = args_cli.checkpoint or get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
    if not checkpoint:
        raise RuntimeError(f"No pretrained checkpoint available for {args_cli.task}")
    print(f"[CHECKPOINT] {checkpoint}", flush=True)

    if args_cli.num_envs < args_cli.episodes_per_scenario:
        args_cli.num_envs = args_cli.episodes_per_scenario

    scenario_name = "scenario_a_no_obstacle" if args_cli.scenario == "no_obstacle" else "scenario_b_static_obstacle"
    summaries = {scenario_name: run_scenario(scenario_name, args_cli.scenario == "obstacle", checkpoint, output_dir)}
    combined = {
        "task": args_cli.task,
        "episodes_per_scenario": args_cli.episodes_per_scenario,
        "max_steps": args_cli.max_steps,
        "success_radius": args_cli.success_radius,
        "goal": [args_cli.goal_x, args_cli.goal_y],
        "obstacle_default": {"pos": list(args_cli.obstacle_pos), "size": list(args_cli.obstacle_size)},
        "uncertainty": {
            "window": args_cli.uncertainty_window,
            "min_progress": args_cli.min_progress,
            "definition": "max(0, min_progress - (distance[t-window] - distance[t]))",
            "threshold": "90th percentile per scenario rollout",
            "trigger_cooldown": args_cli.trigger_cooldown,
        },
        "scenarios": summaries,
    }
    summary_path = output_dir / "metrics" / f"{scenario_name}_summary.json"
    summary_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(f"[WROTE] {summary_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
