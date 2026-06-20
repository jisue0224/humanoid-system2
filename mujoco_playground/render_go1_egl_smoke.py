#!/usr/bin/env python3
"""Render Go1 camera frames through MuJoCo EGL."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jp
import numpy as np
from PIL import Image
from mujoco_playground import registry


def main() -> None:
    artifact_dir = Path("mujoco_playground/artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    env = registry.load("Go1JoystickFlatTerrain", config_overrides={"impl": "jax"})
    state = env.reset(jax.random.PRNGKey(1))
    state.info["command"] = jp.array([1.0, 0.0, 0.0])

    outputs = {}
    for camera, filename in [
        ("top", "go1_top_egl.png"),
        ("back", "go1_back_egl.png"),
    ]:
        frame = env.render(state, height=480, width=640, camera=camera)
        Image.fromarray(frame).save(artifact_dir / filename)
        outputs[camera] = {
            "filename": filename,
            "shape": list(frame.shape),
            "dtype": str(frame.dtype),
            "min": int(np.min(frame)),
            "max": int(np.max(frame)),
        }

    summary = {
        "renderer": "mujoco.Renderer",
        "gl_backend": os.environ.get("MUJOCO_GL"),
        "env": "Go1JoystickFlatTerrain",
        "impl": "jax",
        "outputs": outputs,
    }
    (artifact_dir / "step3_go1_egl_render_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
