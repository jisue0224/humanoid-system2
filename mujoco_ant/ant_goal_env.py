"""Goal-conditioned Ant-v4 wrapper for native MuJoCo System 1 experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class GoalSpec:
    min_radius: float = 3.0
    max_radius: float = 6.0
    max_abs_y: float = 3.0
    success_radius: float = 0.6


class AntGoalEnv(gym.Env):
    """Dense goal-reaching wrapper around Gymnasium `Ant-v4`.

    Observation = Ant-v4 observation plus:
      - dx, dy to goal, clipped to [-10, 10] and scaled by 1/10
      - distance to goal, clipped to [0, 10] and scaled by 1/10
      - bearing to goal as sin/cos of angle from global +x

    The wrapper is intentionally simple. It tests whether a native Ant policy can
    be trained to accept waypoints before adding obstacles and LLM calls.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        *,
        render_mode: str | None = None,
        goal: tuple[float, float] | None = None,
        goal_spec: GoalSpec = GoalSpec(),
        max_episode_steps: int = 300,
        progress_reward_scale: float = 10.0,
        success_reward: float = 50.0,
        healthy_reward: float = 0.5,
        action_penalty_scale: float = 0.02,
    ) -> None:
        self._env = gym.make("Ant-v4", render_mode=render_mode, exclude_current_positions_from_observation=True)
        self._fixed_goal = np.array(goal, dtype=np.float32) if goal is not None else None
        self._goal_spec = goal_spec
        self._max_episode_steps = max_episode_steps
        self._progress_reward_scale = progress_reward_scale
        self._success_reward = success_reward
        self._healthy_reward = healthy_reward
        self._action_penalty_scale = action_penalty_scale
        self._goal = np.zeros(2, dtype=np.float32)
        self._prev_distance = 0.0
        self._steps = 0

        base_space = self._env.observation_space
        assert isinstance(base_space, spaces.Box)
        low = np.concatenate([base_space.low.astype(np.float32), np.array([-1.0, -1.0, 0.0, -1.0, -1.0], dtype=np.float32)])
        high = np.concatenate([base_space.high.astype(np.float32), np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = self._env.action_space

    @property
    def goal_xy(self) -> np.ndarray:
        return self._goal.copy()

    @property
    def data(self):
        return self._env.unwrapped.data

    @property
    def model(self):
        return self._env.unwrapped.model

    def _sample_goal(self) -> np.ndarray:
        if self._fixed_goal is not None:
            return self._fixed_goal.copy()
        radius = self.np_random.uniform(self._goal_spec.min_radius, self._goal_spec.max_radius)
        y = self.np_random.uniform(-self._goal_spec.max_abs_y, self._goal_spec.max_abs_y)
        x = math.sqrt(max(radius * radius - y * y, self._goal_spec.min_radius * self._goal_spec.min_radius))
        return np.array([x, y], dtype=np.float32)

    def _xy(self) -> np.ndarray:
        return np.array(self._env.unwrapped.data.qpos[:2], dtype=np.float32)

    def _distance(self) -> float:
        return float(np.linalg.norm(self._goal - self._xy()))

    def _augment_obs(self, obs: np.ndarray) -> np.ndarray:
        delta = self._goal - self._xy()
        distance = float(np.linalg.norm(delta))
        bearing = math.atan2(float(delta[1]), float(delta[0]))
        goal_features = np.array(
            [
                np.clip(delta[0] / 10.0, -1.0, 1.0),
                np.clip(delta[1] / 10.0, -1.0, 1.0),
                np.clip(distance / 10.0, 0.0, 1.0),
                math.sin(bearing),
                math.cos(bearing),
            ],
            dtype=np.float32,
        )
        return np.concatenate([obs.astype(np.float32), goal_features])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self._env.reset(seed=seed, options=options)
        self._goal = self._sample_goal()
        self._prev_distance = self._distance()
        self._steps = 0
        info = dict(info)
        info["goal_xy"] = self.goal_xy
        info["distance_to_goal"] = self._prev_distance
        return self._augment_obs(obs), info

    def step(self, action: np.ndarray):
        obs, _, terminated, truncated, info = self._env.step(action)
        self._steps += 1

        distance = self._distance()
        progress = self._prev_distance - distance
        success = distance < self._goal_spec.success_radius
        reward = (
            self._progress_reward_scale * progress
            + self._healthy_reward
            - self._action_penalty_scale * float(np.square(action).sum())
        )
        if success:
            reward += self._success_reward

        self._prev_distance = distance
        truncated = bool(truncated or self._steps >= self._max_episode_steps)
        info = dict(info)
        info["goal_xy"] = self.goal_xy
        info["distance_to_goal"] = distance
        info["is_success"] = success
        info["xy"] = self._xy()
        return self._augment_obs(obs), float(reward), bool(terminated or success), truncated, info

    def render(self):
        return self._env.render()

    def close(self) -> None:
        self._env.close()
