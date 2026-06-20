#!/usr/bin/env python3
"""Step 3-1 EGL rendering smoke test for hierarchical Ant scenes."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from PIL import Image

from ant_scene import DEFAULT_GOAL_XY, make_scene_env


def capture(camera_name: str, output_path: Path) -> dict:
    env = make_scene_env(
        with_obstacle=True,
        render_mode="rgb_array",
        camera_name=camera_name,
        goal_xy=DEFAULT_GOAL_XY,
    )
    env.reset(seed=0)
    frame = env.render()
    Image.fromarray(frame).save(output_path)
    env.close()
    return {
        "camera": camera_name,
        "path": str(output_path),
        "shape": list(frame.shape),
        "dtype": str(frame.dtype),
        "min": int(np.min(frame)),
        "max": int(np.max(frame)),
        "mean": float(np.mean(frame)),
    }


def stability_check(camera_name: str, frames: int) -> dict:
    env = make_scene_env(with_obstacle=True, render_mode="rgb_array", camera_name=camera_name)
    env.reset(seed=42)
    stats = []
    for _ in range(frames):
        frame = env.render()
        stats.append(
            {
                "shape": list(frame.shape),
                "min": int(np.min(frame)),
                "max": int(np.max(frame)),
                "mean": float(np.mean(frame)),
            }
        )
    env.close()
    return {
        "camera": camera_name,
        "frames": frames,
        "all_nonblank": all(item["max"] > item["min"] for item in stats),
        "stats": stats,
    }


def main() -> None:
    out_dir = Path("mujoco_ant_hierarchical/artifacts/step3_1_vision")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "gl_backend": os.environ.get("MUJOCO_GL"),
        "goal_xy": list(DEFAULT_GOAL_XY),
        "captures": [
            capture("overhead", out_dir / "overhead_obstacle_goal.png"),
            capture("ego", out_dir / "egocentric_obstacle_goal.png"),
        ],
        "stability": [
            stability_check("overhead", 10),
            stability_check("ego", 10),
        ],
    }
    path = out_dir / "egl_vision_smoke_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
