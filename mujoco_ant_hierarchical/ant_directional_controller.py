"""Frozen Ant-v4 locomotion policy with target-aligned observation transform."""

from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from huggingface_sb3 import load_from_hub
from stable_baselines3 import SAC


DEFAULT_REPO_ID = "jren123/sac-ant-v4"
DEFAULT_FILENAME = "SAC-Ant-v4.zip"


@dataclass(frozen=True)
class DirectionalControllerConfig:
    """No-learning high-level parameters for the frozen Ant policy."""

    success_radius: float = 0.75
    slow_radius: float = 0.0
    min_action_scale: float = 1.0


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


def distance_action_scale(distance: float, config: DirectionalControllerConfig) -> float:
    """Scale frozen policy actions down near the target to reduce overshoot."""

    if config.slow_radius <= config.success_radius:
        return 1.0
    alpha = (distance - config.success_radius) / (config.slow_radius - config.success_radius)
    alpha = max(0.0, min(1.0, alpha))
    return config.min_action_scale + (1.0 - config.min_action_scale) * alpha


class DirectionalAntController:
    """High-level target heading wrapper around the frozen pretrained SAC policy."""

    def __init__(
        self,
        repo_id: str = DEFAULT_REPO_ID,
        filename: str = DEFAULT_FILENAME,
        config: DirectionalControllerConfig | None = None,
    ):
        checkpoint = load_from_hub(repo_id=repo_id, filename=filename)
        self.model = SAC.load(checkpoint)
        self.checkpoint = checkpoint
        self.config = config or DirectionalControllerConfig()

    def predict(self, obs: np.ndarray, current_xy: np.ndarray, target_xy: np.ndarray) -> tuple[np.ndarray, float]:
        action, target_heading, _ = self.predict_with_info(obs, current_xy, target_xy)
        return action, target_heading

    def predict_with_info(
        self, obs: np.ndarray, current_xy: np.ndarray, target_xy: np.ndarray
    ) -> tuple[np.ndarray, float, dict]:
        delta = target_xy - current_xy
        distance = float(np.linalg.norm(delta))
        target_heading = math.atan2(float(delta[1]), float(delta[0]))
        aligned_obs = target_aligned_observation(obs, target_heading)
        action, _ = self.model.predict(aligned_obs, deterministic=True)
        action_scale = distance_action_scale(distance, self.config)
        return action * action_scale, target_heading, {
            "distance": distance,
            "action_scale": action_scale,
        }


def make_env(*, render_mode: str | None = None):
    return gym.make("Ant-v4", render_mode=render_mode)
