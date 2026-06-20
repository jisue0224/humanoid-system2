"""Frozen Ant-v4 locomotion policy with target-aligned observation transform."""

from __future__ import annotations

import math
from pathlib import Path

import gymnasium as gym
import numpy as np
from huggingface_sb3 import load_from_hub
from stable_baselines3 import SAC


DEFAULT_REPO_ID = "jren123/sac-ant-v4"
DEFAULT_FILENAME = "SAC-Ant-v4.zip"


def yaw_quat(theta: float) -> np.ndarray:
    return np.array([math.cos(theta * 0.5), 0.0, 0.0, math.sin(theta * 0.5)], dtype=np.float64)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def rotate_xy(value: np.ndarray, theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([c * value[0] - s * value[1], s * value[0] + c * value[1]], dtype=np.float64)


def target_aligned_observation(obs: np.ndarray, target_heading: float) -> np.ndarray:
    """Express root orientation and xy velocity in the target-aligned frame.

    Ant-v4 default observations are qpos[2:] followed by qvel[:]. qpos[0:2]
    world x/y are excluded. The first qpos entries in obs are:
      obs[0]    root z
      obs[1:5] root quaternion in MuJoCo wxyz order
      obs[5:13] joint positions
      obs[13:27] qvel, with root xy velocity at obs[13:15]
    """

    transformed = obs.copy()
    transformed[1:5] = quat_mul(yaw_quat(-target_heading), transformed[1:5])
    transformed[13:15] = rotate_xy(transformed[13:15], -target_heading)
    return transformed


class DirectionalAntController:
    """High-level target heading wrapper around the frozen pretrained SAC policy."""

    def __init__(self, repo_id: str = DEFAULT_REPO_ID, filename: str = DEFAULT_FILENAME):
        checkpoint = load_from_hub(repo_id=repo_id, filename=filename)
        self.model = SAC.load(checkpoint)
        self.checkpoint = checkpoint

    def predict(self, obs: np.ndarray, current_xy: np.ndarray, target_xy: np.ndarray) -> tuple[np.ndarray, float]:
        delta = target_xy - current_xy
        target_heading = math.atan2(float(delta[1]), float(delta[0]))
        aligned_obs = target_aligned_observation(obs, target_heading)
        action, _ = self.model.predict(aligned_obs, deterministic=True)
        return action, target_heading


def make_env(*, render_mode: str | None = None):
    return gym.make("Ant-v4", render_mode=render_mode)
