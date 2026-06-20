#!/usr/bin/env python3
"""Step 1: load the Unitree H1 MJCF and run a basic passive simulation."""

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_MODEL = "mujoco_sim2sim/vendor/mujoco_menagerie/unitree_h1/scene.xml"


def quat_to_roll_pitch_yaw_wxyz(quat: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = quat
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def object_names(model: mujoco.MjModel, obj_type: mujoco.mjtObj, count: int) -> list[str | None]:
    return [mujoco.mj_id2name(model, obj_type, idx) for idx in range(count)]


def render(model: mujoco.MjModel, data: mujoco.MjData, output_path: Path) -> None:
    renderer = mujoco.Renderer(model, height=480, width=480)
    renderer.update_scene(data)
    pixels = renderer.render()
    renderer.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(output_path)


def plot_trace(output_path: Path, times: list[float], base_z: list[float], roll: list[float], pitch: list[float]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    axes[0].plot(times, base_z, linewidth=1.8)
    axes[0].axhline(0.6, color="red", linestyle="--", linewidth=1.0, label="collapse threshold")
    axes[0].set_ylabel("base z (m)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(times, roll, linewidth=1.5, label="roll")
    axes[1].plot(times, pitch, linewidth=1.5, label="pitch")
    axes[1].axhline(math.radians(60.0), color="red", linestyle="--", linewidth=1.0)
    axes[1].axhline(-math.radians(60.0), color="red", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("angle (rad)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    fig.suptitle("H1 passive zero-torque simulation")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path(DEFAULT_MODEL))
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--output_dir", type=Path, default=Path("mujoco_sim2sim/artifacts/step1"))
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    joint_names = object_names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    actuator_names = object_names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)
    hinge_joint_names = [name for name in joint_names if name is not None]
    has_freejoint = model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE

    qpos_initial = data.qpos.copy()
    xpos_initial = data.xpos.copy()
    initial_base_z = float(data.qpos[2])
    initial_roll, initial_pitch, initial_yaw = quat_to_roll_pitch_yaw_wxyz(data.qpos[3:7])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    render(model, data, args.output_dir / "h1_initial.png")

    steps = int(args.seconds / model.opt.timestep)
    times: list[float] = []
    base_z: list[float] = []
    roll: list[float] = []
    pitch: list[float] = []
    yaw: list[float] = []
    ctrl_norm: list[float] = []
    for _ in range(steps):
        data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)
        r, p, y = quat_to_roll_pitch_yaw_wxyz(data.qpos[3:7])
        times.append(float(data.time))
        base_z.append(float(data.qpos[2]))
        roll.append(r)
        pitch.append(p)
        yaw.append(y)
        ctrl_norm.append(float(np.linalg.norm(data.ctrl)))

    render(model, data, args.output_dir / "h1_passive_final.png")
    plot_trace(args.output_dir / "h1_passive_trace.png", times, base_z, roll, pitch)

    final_roll, final_pitch, final_yaw = quat_to_roll_pitch_yaw_wxyz(data.qpos[3:7])
    max_abs_tilt = max(abs(final_roll), abs(final_pitch))
    collapsed = bool(data.qpos[2] < 0.6 or max_abs_tilt > math.radians(60.0))
    qpos_delta_norm = float(np.linalg.norm(data.qpos - qpos_initial))
    root_delta = (data.qpos[:3] - qpos_initial[:3]).tolist()

    summary = {
        "model_path": str(args.model),
        "mujoco_version": mujoco.__version__,
        "timestep": model.opt.timestep,
        "simulated_seconds": args.seconds,
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "njnt": model.njnt,
        "nbody": model.nbody,
        "ngeom": model.ngeom,
        "has_freejoint": bool(has_freejoint),
        "actuated_dof": model.nu,
        "hinge_joint_count": len(hinge_joint_names),
        "joint_names": joint_names,
        "actuator_names": actuator_names,
        "initial": {
            "base_z": initial_base_z,
            "roll": initial_roll,
            "pitch": initial_pitch,
            "yaw": initial_yaw,
            "qpos": qpos_initial.tolist(),
            "pelvis_xpos": xpos_initial[1].tolist(),
        },
        "final": {
            "time": float(data.time),
            "base_z": float(data.qpos[2]),
            "roll": final_roll,
            "pitch": final_pitch,
            "yaw": final_yaw,
            "root_delta": root_delta,
            "qpos_delta_norm": qpos_delta_norm,
        },
        "passive_zero_torque_collapsed": collapsed,
        "collapse_rule": "base_z < 0.6 or max(abs(roll), abs(pitch)) > 60deg",
        "trace": {
            "time": times,
            "base_z": base_z,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "ctrl_norm": ctrl_norm,
        },
        "images": {
            "initial": str(args.output_dir / "h1_initial.png"),
            "passive_final": str(args.output_dir / "h1_passive_final.png"),
            "passive_trace": str(args.output_dir / "h1_passive_trace.png"),
        },
    }
    summary_path = args.output_dir / "h1_mjcf_step1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({k: summary[k] for k in ["nq", "nv", "nu", "njnt", "has_freejoint", "actuated_dof", "hinge_joint_count"]}, indent=2))
    print(
        "initial_base_z={:.3f} final_base_z={:.3f} final_roll={:.3f} final_pitch={:.3f} collapsed={}".format(
            initial_base_z,
            summary["final"]["base_z"],
            final_roll,
            final_pitch,
            collapsed,
        )
    )
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
