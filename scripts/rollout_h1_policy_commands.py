#!/usr/bin/env python3
"""Evaluate H1 pretrained velocity policy under fixed commands."""

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Roll out pretrained H1 velocity policy with fixed commands.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-H1-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--output", type=str, default="experiments/logs/h1_policy_command_metrics.json")
parser.add_argument("--case", type=str, default="forward_vx_1")
parser.add_argument("--vx", type=float, default=1.0)
parser.add_argument("--vy", type=float, default=0.0)
parser.add_argument("--yaw", type=float, default=0.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg


def wrap_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def configure_fixed_command(env_cfg, vx: float, vy: float, yaw: float) -> None:
    cmd = env_cfg.commands.base_velocity
    cmd.heading_command = False
    cmd.rel_heading_envs = 0.0
    cmd.rel_standing_envs = 0.0
    cmd.debug_vis = False
    cmd.resampling_time_range = (1000.0, 1000.0)
    cmd.ranges.lin_vel_x = (vx, vx)
    cmd.ranges.lin_vel_y = (vy, vy)
    cmd.ranges.ang_vel_z = (yaw, yaw)
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


def run_case(case_name: str, command: tuple[float, float, float], checkpoint: str) -> dict:
    vx, vy, yaw = command
    print(f"[ROLL] case={case_name} command=(vx={vx}, vy={vy}, yaw={yaw})", flush=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    configure_fixed_command(env_cfg, vx, vy, yaw)

    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args_cli.device

    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    robot = env.unwrapped.scene["robot"]

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs, _ = env.get_observations()
    pos0 = robot.data.root_pos_w[0].detach().cpu().tolist()
    heading0 = float(robot.data.heading_w[0].detach().cpu())
    xs = []
    ys = []
    headings = []
    lin_vel_x = []
    yaw_vel = []
    done_count = 0
    for step in range(args_cli.steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
        xs.append(float(robot.data.root_pos_w[0, 0].detach().cpu()))
        ys.append(float(robot.data.root_pos_w[0, 1].detach().cpu()))
        headings.append(float(robot.data.heading_w[0].detach().cpu()))
        lin_vel_x.append(float(robot.data.root_lin_vel_b[0, 0].detach().cpu()))
        yaw_vel.append(float(robot.data.root_ang_vel_b[0, 2].detach().cpu()))
        done_count += int(dones[0].detach().cpu())
        if (step + 1) % 25 == 0:
            print(f"[ROLL] {case_name} completed_step={step + 1}", flush=True)

    pos1 = robot.data.root_pos_w[0].detach().cpu().tolist()
    heading1 = float(robot.data.heading_w[0].detach().cpu())
    duration = args_cli.steps * raw_env.unwrapped.step_dt
    result = {
        "case": case_name,
        "command": {"vx": vx, "vy": vy, "yaw": yaw},
        "steps": args_cli.steps,
        "duration_s": duration,
        "start_pos": pos0,
        "end_pos": pos1,
        "delta_x": pos1[0] - pos0[0],
        "delta_y": pos1[1] - pos0[1],
        "start_heading": heading0,
        "end_heading": heading1,
        "delta_heading_wrapped": wrap_pi(heading1 - heading0),
        "mean_base_lin_vel_x_b": sum(lin_vel_x) / len(lin_vel_x),
        "mean_base_yaw_vel_b": sum(yaw_vel) / len(yaw_vel),
        "done_count": done_count,
        "sampled_final_command": env.unwrapped.command_manager.get_command("base_velocity")[0].detach().cpu().tolist(),
    }
    print("[ROLL_RESULT] " + json.dumps(result, sort_keys=True), flush=True)
    env.close()
    return result


def main() -> None:
    checkpoint = args_cli.checkpoint or get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
    if not checkpoint:
        raise RuntimeError(f"No pretrained checkpoint available for {args_cli.task}")
    print(f"[ROLL] checkpoint={checkpoint}", flush=True)

    results = [run_case(args_cli.case, (args_cli.vx, args_cli.vy, args_cli.yaw), checkpoint)]
    output_path = Path(args_cli.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[ROLL] wrote_metrics={output_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
