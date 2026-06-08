#!/usr/bin/env python3
"""Run Isaac-Velocity-Flat-H1-v0 for a fixed number of random-action steps."""

import argparse
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Fixed-length H1 Isaac Lab smoke run.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-H1-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    print("[SMOKE] parsing env cfg", flush=True)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    print("[SMOKE] creating gym env", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f"[INFO] observation_space={env.observation_space}")
    print(f"[INFO] action_space={env.action_space}")
    print("[SMOKE] resetting env", flush=True)
    env.reset()

    print(f"[SMOKE] stepping {args_cli.steps} steps", flush=True)
    for step in range(args_cli.steps):
        with torch.inference_mode():
            actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            env.step(actions)
        if (step + 1) % 10 == 0:
            print(f"[INFO] completed_step={step + 1}")

    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        print("[SMOKE] exception raised", flush=True)
        traceback.print_exc()
        raise
    finally:
        print("[SMOKE] closing simulation app", flush=True)
        simulation_app.close()
