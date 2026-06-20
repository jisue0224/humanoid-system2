#!/usr/bin/env python3
"""Step 1 smoke test for a HuggingFace pretrained SAC Ant-v4 policy."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from huggingface_sb3 import load_from_hub
from stable_baselines3 import SAC


DEFAULT_REPO_ID = "jren123/sac-ant-v4"
DEFAULT_FILENAME = "SAC-Ant-v4.zip"


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def rollout(
    *,
    env_id: str,
    repo_id: str,
    filename: str,
    steps: int,
    seed: int,
    artifact_dir: Path,
) -> dict:
    artifact_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MUJOCO_GL", "egl")

    checkpoint_path = load_from_hub(repo_id=repo_id, filename=filename)
    model = SAC.load(checkpoint_path)

    env = gym.make(env_id)
    obs, _ = env.reset(seed=seed)

    unwrapped = env.unwrapped
    dt = float(getattr(unwrapped, "dt", np.nan))
    start_xy = np.array(unwrapped.data.qpos[:2], dtype=np.float64)

    positions = [start_xy.copy()]
    torso_z = [float(unwrapped.data.qpos[2])]
    rewards: list[float] = []
    actions: list[np.ndarray] = []
    terminated = False
    truncated = False
    actual_steps = 0

    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        actual_steps += 1

        positions.append(np.array(unwrapped.data.qpos[:2], dtype=np.float64))
        torso_z.append(float(unwrapped.data.qpos[2]))
        rewards.append(float(reward))
        actions.append(np.array(action, dtype=np.float64))

        if terminated or truncated:
            break

    end_xy = positions[-1]
    delta_xy = end_xy - start_xy
    elapsed_s = actual_steps * dt if np.isfinite(dt) else np.nan
    mean_xy_velocity = delta_xy / elapsed_s if elapsed_s > 0 else np.array([np.nan, np.nan])
    path_length = float(np.sum(np.linalg.norm(np.diff(np.vstack(positions), axis=0), axis=1)))

    summary = {
        "env_id": env_id,
        "policy_repo_id": repo_id,
        "policy_filename": filename,
        "checkpoint_path": checkpoint_path,
        "seed": seed,
        "requested_steps": steps,
        "actual_steps": actual_steps,
        "terminated": terminated,
        "truncated": truncated,
        "dt": dt,
        "elapsed_s": elapsed_s,
        "observation_shape": list(env.observation_space.shape),
        "action_shape": list(env.action_space.shape),
        "start_xy": start_xy,
        "end_xy": end_xy,
        "delta_xy": delta_xy,
        "xy_displacement": float(np.linalg.norm(delta_xy)),
        "path_length": path_length,
        "mean_xy_velocity": mean_xy_velocity,
        "torso_z_min": float(np.min(torso_z)),
        "torso_z_max": float(np.max(torso_z)),
        "torso_z_final": float(torso_z[-1]),
        "reward_sum": float(np.sum(rewards)),
        "reward_mean": float(np.mean(rewards)) if rewards else np.nan,
        "action_abs_mean": float(np.mean(np.abs(actions))) if actions else np.nan,
        "stable_100_step_rollout": actual_steps == steps and not terminated and not truncated,
    }

    summary_path = artifact_dir / "step1_pretrained_ant_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default) + "\n")

    positions_array = np.vstack(positions)
    plt.figure(figsize=(6, 5))
    plt.plot(positions_array[:, 0], positions_array[:, 1], marker="o", markersize=2, linewidth=1.5)
    plt.scatter([start_xy[0]], [start_xy[1]], c="green", label="start")
    plt.scatter([end_xy[0]], [end_xy[1]], c="red", label="end")
    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(f"{env_id} pretrained SAC rollout ({actual_steps} steps)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(artifact_dir / "step1_pretrained_ant_trajectory.png", dpi=160)
    plt.close()

    env.close()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="Ant-v4")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--artifact-dir", type=Path, default=Path("mujoco_ant/artifacts/step1"))
    args = parser.parse_args()

    summary = rollout(
        env_id=args.env_id,
        repo_id=args.repo_id,
        filename=args.filename,
        steps=args.steps,
        seed=args.seed,
        artifact_dir=args.artifact_dir,
    )
    print(json.dumps(summary, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
