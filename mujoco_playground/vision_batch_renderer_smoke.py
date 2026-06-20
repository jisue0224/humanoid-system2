#!/usr/bin/env python3
"""Exercise the MuJoCo Playground MJWarp vision path."""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import jax
from mujoco_playground import registry


def main() -> None:
    artifact_dir = Path("mujoco_playground/artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "target": "CartpoleBalance vision=True",
        "gl_backend": os.environ.get("MUJOCO_GL"),
        "jax_backend": jax.default_backend(),
    }

    try:
        env_cfg = registry.get_default_config("CartpoleBalance")
        env_cfg.vision = True
        env_cfg.vision_config.nworld = 1
        env = registry.load("CartpoleBalance", config=env_cfg)
        state = env.reset(jax.random.PRNGKey(0))
        summary["success"] = True
        summary["obs"] = {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in state.obs.items()
        }
    except Exception as exc:  # pylint: disable=broad-exception-caught
        summary["success"] = False
        summary["error_type"] = type(exc).__name__
        summary["error"] = str(exc)
        summary["traceback"] = traceback.format_exc()

    path = artifact_dir / "step3_mjwarp_vision_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
