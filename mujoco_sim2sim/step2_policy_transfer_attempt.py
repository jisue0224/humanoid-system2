#!/usr/bin/env python3
"""Step 2: attempt to wire the Isaac Lab H1 RSL-RL policy interface to MuJoCo.

This script implements the MuJoCo side of the Isaac Lab flat H1 interface:

- 69D observation:
  base linear velocity, base angular velocity, projected gravity, velocity command,
  relative joint position, joint velocity, previous action.
- 19D action:
  joint-position target residual with Isaac Lab scale=0.5 and default-joint offset.
- MuJoCo torque application:
  PD torque tracking around target joint positions, using Isaac Lab H1 actuator gains.

If a checkpoint is supplied, the script tries to load an RSL-RL actor MLP. If no
checkpoint is available, it still runs a zero-action PD hold baseline to verify
the action wrapper.
"""

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - reported in JSON
    torch = None
    nn = None


MODEL_PATH = Path("mujoco_sim2sim/vendor/mujoco_menagerie/unitree_h1/scene.xml")
DEFAULT_CHECKPOINT = Path("mujoco_sim2sim/checkpoints/rsl_rl_Isaac-Velocity-Flat-H1-v0_checkpoint.pt")

JOINT_NAMES = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
    "torso",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
]

DEFAULT_JOINT_POS = np.array(
    [
        0.0,
        0.0,
        -0.28,
        0.79,
        -0.52,
        0.0,
        0.0,
        -0.28,
        0.79,
        -0.52,
        0.0,
        0.28,
        0.0,
        0.0,
        0.52,
        0.28,
        0.0,
        0.0,
        0.52,
    ],
    dtype=np.float64,
)

KP = np.array(
    [
        150.0,
        150.0,
        200.0,
        200.0,
        20.0,
        150.0,
        150.0,
        200.0,
        200.0,
        20.0,
        200.0,
        40.0,
        40.0,
        40.0,
        40.0,
        40.0,
        40.0,
        40.0,
        40.0,
    ],
    dtype=np.float64,
)

KD = np.array(
    [
        5.0,
        5.0,
        5.0,
        5.0,
        4.0,
        5.0,
        5.0,
        5.0,
        5.0,
        4.0,
        5.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
        10.0,
    ],
    dtype=np.float64,
)

EFFORT_LIMIT = np.array(
    [300, 300, 300, 300, 100, 300, 300, 300, 300, 100, 300, 300, 300, 300, 300, 300, 300, 300, 300],
    dtype=np.float64,
)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_rotate_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = quat_conjugate(q_wxyz)
    w, x, y, z = q
    qvec = np.array([x, y, z])
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2.0 * (w * uv + uuv)


def yaw_from_quat(q: np.ndarray) -> float:
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def set_isaac_default_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = np.array([0.0, 0.0, 1.05])
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qpos[7 : 7 + len(JOINT_NAMES)] = DEFAULT_JOINT_POS
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def make_observation(data: mujoco.MjData, command: np.ndarray, previous_action: np.ndarray) -> np.ndarray:
    q = data.qpos[3:7].copy()
    base_lin_vel_b = quat_rotate_inverse(q, data.qvel[0:3].copy())
    base_ang_vel_b = quat_rotate_inverse(q, data.qvel[3:6].copy())
    projected_gravity_b = quat_rotate_inverse(q, np.array([0.0, 0.0, -1.0]))
    joint_pos_rel = data.qpos[7 : 7 + len(JOINT_NAMES)] - DEFAULT_JOINT_POS
    joint_vel = data.qvel[6 : 6 + len(JOINT_NAMES)]
    return np.concatenate(
        [base_lin_vel_b, base_ang_vel_b, projected_gravity_b, command, joint_pos_rel, joint_vel, previous_action]
    ).astype(np.float32)


def pd_torque(data: mujoco.MjData, action: np.ndarray) -> np.ndarray:
    target = DEFAULT_JOINT_POS + 0.5 * action
    q = data.qpos[7 : 7 + len(JOINT_NAMES)]
    qd = data.qvel[6 : 6 + len(JOINT_NAMES)]
    tau = KP * (target - q) - KD * qd
    return np.clip(tau, -EFFORT_LIMIT, EFFORT_LIMIT)


class Actor(nn.Module):
    def __init__(self, obs_dim: int = 69, action_dim: int = 19):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, obs):
        return self.net(obs)


def extract_actor_state_dict(checkpoint: Path) -> tuple[dict | None, dict]:
    if torch is None:
        return None, {"error": "torch is not importable"}
    if not checkpoint.exists():
        return None, {"error": f"checkpoint not found: {checkpoint}"}
    ckpt = torch.load(checkpoint, map_location="cpu")
    metadata = {"top_level_keys": list(ckpt.keys()) if isinstance(ckpt, dict) else str(type(ckpt))}
    candidates = []
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "actor_critic_state_dict", "state_dict", "model"):
            value = ckpt.get(key)
            if isinstance(value, dict):
                candidates.append((key, value))
        candidates.append(("root", ckpt))
    for key, state in candidates:
        actor_items = {}
        for name, tensor in state.items():
            if not hasattr(tensor, "shape"):
                continue
            if name.startswith("actor."):
                actor_items[name.removeprefix("actor.")] = tensor
            elif name.startswith("actor_net."):
                actor_items[name.removeprefix("actor_net.")] = tensor
        if actor_items:
            metadata["actor_source_key"] = key
            metadata["actor_tensor_keys"] = list(actor_items.keys())[:20]
            return actor_items, metadata
    metadata["error"] = "could not identify actor weights in checkpoint"
    return None, metadata


def load_actor(checkpoint: Path) -> tuple[Actor | None, dict]:
    actor_state, metadata = extract_actor_state_dict(checkpoint)
    if actor_state is None:
        return None, metadata
    actor = Actor()
    try:
        actor.load_state_dict(actor_state, strict=True)
    except Exception as exc:
        metadata["error"] = f"actor load_state_dict failed: {exc!r}"
        return None, metadata
    actor.eval()
    return actor, metadata


def render(model: mujoco.MjModel, data: mujoco.MjData, output_path: Path) -> None:
    renderer = mujoco.Renderer(model, height=480, width=480)
    renderer.update_scene(data)
    pixels = renderer.render()
    renderer.close()
    Image.fromarray(pixels).save(output_path)


def run_rollout(args) -> dict:
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    set_isaac_default_pose(model, data)

    actor, checkpoint_metadata = load_actor(args.checkpoint)
    command = np.array([args.vx, args.vy, args.yaw], dtype=np.float32)
    previous_action = np.zeros(19, dtype=np.float32)

    steps = int(args.seconds / model.opt.timestep)
    policy_decimation = max(1, int(round(args.policy_dt / model.opt.timestep)))
    action = np.zeros(19, dtype=np.float32)
    trace = []
    used_policy = actor is not None

    for step in range(steps):
        if step % policy_decimation == 0:
            obs = make_observation(data, command, previous_action)
            if actor is not None:
                with torch.inference_mode():
                    action = actor(torch.tensor(obs).unsqueeze(0)).squeeze(0).numpy()
                action = np.clip(action, -args.clip_actions, args.clip_actions)
            else:
                action = np.zeros(19, dtype=np.float32)
            previous_action = action.copy()
        data.ctrl[:] = pd_torque(data, action)
        mujoco.mj_step(model, data)
        if step % policy_decimation == 0:
            trace.append(
                {
                    "time": float(data.time),
                    "base_pos": data.qpos[:3].copy().tolist(),
                    "yaw": yaw_from_quat(data.qpos[3:7].copy()),
                    "action_norm": float(np.linalg.norm(action)),
                    "torque_norm": float(np.linalg.norm(data.ctrl)),
                }
            )

    final_yaw = yaw_from_quat(data.qpos[3:7])
    final_pos = data.qpos[:3].copy()
    collapsed = bool(final_pos[2] < 0.6)
    render(model, data, args.output_dir / "step2_final.png")
    return {
        "model_path": str(args.model),
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_metadata": checkpoint_metadata,
        "used_policy": used_policy,
        "fallback_mode": None if used_policy else "zero_action_pd_hold",
        "obs_dim": 69,
        "action_dim": 19,
        "action_semantics": "target_joint_position = default_joint_position + 0.5 * action",
        "policy_dt": args.policy_dt,
        "mujoco_timestep": model.opt.timestep,
        "policy_decimation": policy_decimation,
        "command": {"vx": args.vx, "vy": args.vy, "yaw": args.yaw},
        "initial_base_pos": [0.0, 0.0, 1.05],
        "final_base_pos": final_pos.tolist(),
        "delta_x": float(final_pos[0]),
        "delta_y": float(final_pos[1]),
        "final_yaw": final_yaw,
        "collapsed": collapsed,
        "trace": trace,
        "image": str(args.output_dir / "step2_final.png"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output_dir", type=Path, default=Path("mujoco_sim2sim/artifacts/step2"))
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--policy_dt", type=float, default=0.02)
    parser.add_argument("--clip_actions", type=float, default=100.0)
    parser.add_argument("--vx", type=float, default=1.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = run_rollout(args)
    summary_path = args.output_dir / "step2_policy_transfer_attempt.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["used_policy", "fallback_mode", "delta_x", "delta_y", "final_yaw", "collapsed"]}, indent=2))
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
