#!/usr/bin/env python3
"""Minimal H1 camera sensor test for Step 4."""

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Attach a camera to H1 and save one RGB/depth frame.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-H1-v0")
parser.add_argument("--output_dir", type=str, default="experiments/step4_camera_test")
parser.add_argument("--width", type=int, default=224)
parser.add_argument("--height", type=int, default=224)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def save_rgb(path: Path, rgb: torch.Tensor) -> None:
    array = rgb.detach().cpu().numpy()
    if array.ndim == 4:
        array = array[0]
    array = np.asarray(array, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, array)


def save_depth(path: Path, depth: torch.Tensor) -> None:
    array = depth.detach().cpu().numpy()
    if array.ndim == 4:
        array = array[0]
    array = np.squeeze(array)
    finite = np.isfinite(array)
    if finite.any():
        lo = float(array[finite].min())
        hi = float(array[finite].max())
        norm = (array - lo) / max(hi - lo, 1e-6)
    else:
        norm = np.zeros_like(array, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, norm, cmap="viridis")


def configure_env(env_cfg) -> None:
    env_cfg.scene.num_envs = 1
    env_cfg.seed = 0
    env_cfg.sim.device = args_cli.device
    env_cfg.sim.render.antialiasing_mode = "OFF"
    env_cfg.rerender_on_reset = True
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

    # H1_MINIMAL in Isaac-Velocity-Flat-H1-v0 exposes torso_link but no separate head link.
    # This places the optical sensor at head height, attached to the torso frame.
    env_cfg.scene.head_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/head_cam",
        update_period=0.0,
        height=args_cli.height,
        width=args_cli.width,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.25, 0.0, 0.45), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"),
    )


def main() -> None:
    output_dir = Path(args_cli.output_dir)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    configure_env(env_cfg)
    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()
    del obs
    zero_action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    env.step(zero_action)
    env.unwrapped.sim.render()
    camera = env.unwrapped.scene["head_cam"]
    output = camera.data.output
    rgb = output["rgb"]
    depth = output["distance_to_image_plane"]
    rgb_path = output_dir / "h1_head_camera_rgb.png"
    depth_path = output_dir / "h1_head_camera_depth.png"
    meta_path = output_dir / "h1_head_camera_metadata.json"
    save_rgb(rgb_path, rgb)
    save_depth(depth_path, depth)
    metadata = {
        "status": "success",
        "task": args_cli.task,
        "camera_prim_path": "{ENV_REGEX_NS}/Robot/torso_link/head_cam",
        "rgb_shape": list(rgb.shape),
        "rgb_dtype": str(rgb.dtype),
        "depth_shape": list(depth.shape),
        "depth_dtype": str(depth.dtype),
        "rgb_path": str(rgb_path),
        "depth_path": str(depth_path),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
