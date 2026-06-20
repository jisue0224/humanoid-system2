#!/usr/bin/env python3
"""Train a small goal-conditioned Ant System 1 policy and measure viability."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from ant_goal_env import AntGoalEnv, GoalSpec


def evaluate(model: SAC, *, episodes: int, max_steps: int, seed: int) -> tuple[list[dict], float]:
    results = []
    for episode in range(episodes):
        env = AntGoalEnv(max_episode_steps=max_steps, goal_spec=GoalSpec())
        obs, info = env.reset(seed=seed + episode)
        trajectory = [env.data.qpos[:2].copy().tolist()]
        goal = np.array(info["goal_xy"], dtype=np.float32)
        success = False
        total_reward = 0.0
        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            trajectory.append(env.data.qpos[:2].copy().tolist())
            success = bool(info.get("is_success", False))
            if terminated or truncated:
                break
        final_xy = np.array(trajectory[-1], dtype=np.float32)
        final_distance = float(np.linalg.norm(goal - final_xy))
        results.append(
            {
                "episode": episode,
                "success": success,
                "steps": step + 1,
                "goal_xy": goal.tolist(),
                "final_xy": final_xy.tolist(),
                "final_distance": final_distance,
                "total_reward": total_reward,
                "trajectory": trajectory,
            }
        )
        env.close()
    success_rate = sum(result["success"] for result in results) / len(results)
    return results, success_rate


def plot_eval(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    for result in results:
        trajectory = np.array(result["trajectory"], dtype=np.float32)
        plt.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.75, linewidth=1.3)
        plt.scatter([result["goal_xy"][0]], [result["goal_xy"][1]], marker="*", s=60, c="red")
    plt.scatter([0.0], [0.0], c="green", s=60, label="start")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.title("Goal-conditioned Ant evaluation")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_timesteps", type=int, default=25_000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--artifact_dir", type=Path, default=Path("mujoco_ant/artifacts/step2_goal_ant"))
    args = parser.parse_args()

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    env = Monitor(AntGoalEnv(max_episode_steps=args.max_steps, goal_spec=GoalSpec()))
    model = SAC(
        "MlpPolicy",
        env,
        seed=args.seed,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=256,
        learning_starts=2_000,
        train_freq=1,
        gradient_steps=1,
        gamma=0.98,
        policy_kwargs={"net_arch": [256, 256]},
    )

    started = time.perf_counter()
    model.learn(total_timesteps=args.total_timesteps, progress_bar=False)
    train_seconds = time.perf_counter() - started
    env.close()

    model_path = args.artifact_dir / "sac_ant_goal_smoke.zip"
    model.save(model_path)

    eval_results, success_rate = evaluate(
        model,
        episodes=args.eval_episodes,
        max_steps=args.max_steps,
        seed=args.seed + 10_000,
    )
    plot_eval(eval_results, args.artifact_dir / "goal_ant_eval_trajectories.png")

    summary = {
        "total_timesteps": args.total_timesteps,
        "train_seconds": train_seconds,
        "steps_per_second": args.total_timesteps / train_seconds,
        "eval_episodes": args.eval_episodes,
        "max_steps": args.max_steps,
        "success_rate": success_rate,
        "mean_final_distance": float(np.mean([r["final_distance"] for r in eval_results])),
        "median_final_distance": float(np.median([r["final_distance"] for r in eval_results])),
        "model_path": str(model_path),
        "results": eval_results,
    }
    summary_path = args.artifact_dir / "goal_ant_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
