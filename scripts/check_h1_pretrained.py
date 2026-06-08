#!/usr/bin/env python3
"""Check whether Isaac Lab publishes an RSL-RL checkpoint for H1 flat velocity."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Check H1 pretrained checkpoint availability.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-H1-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint


def main() -> int:
    checkpoint = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
    if checkpoint:
        print(f"[PRETRAINED] available path={checkpoint}", flush=True)
        return 0
    print(f"[PRETRAINED] unavailable task={args_cli.task}", flush=True)
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()

