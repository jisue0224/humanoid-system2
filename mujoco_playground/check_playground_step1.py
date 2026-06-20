#!/usr/bin/env python3
"""Step 1 capability checks for MuJoCo Playground in this container."""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path


def main() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    artifact_dir = Path("mujoco_playground/artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {}

    try:
        import jax
        import mujoco
        import mujoco_playground
        from mujoco_playground import registry

        summary["jax_version"] = jax.__version__
        summary["jax_backend"] = jax.default_backend()
        summary["jax_devices"] = [str(device) for device in jax.devices()]
        summary["mujoco_version"] = mujoco.__version__
        summary["mujoco_playground_import"] = True
        summary["locomotion_envs"] = list(registry.locomotion.ALL_ENVS)

        env = registry.load("Go1JoystickFlatTerrain", config_overrides={"impl": "jax"})
        state = env.reset(jax.random.PRNGKey(0))
        state = env.step(state, jax.numpy.zeros(env.action_size))
        summary["go1_jax_load"] = True
        summary["go1_action_size"] = int(env.action_size)
        summary["go1_dt"] = float(env.dt)
        summary["go1_observation_size"] = {
            key: list(value) for key, value in env.observation_size.items()
        }
        summary["go1_initial_command"] = [
            float(x) for x in state.info["command"].tolist()
        ]
        summary["go1_one_step_reward"] = float(state.reward)
        summary["go1_one_step_done"] = float(state.done)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        summary["step1_error_type"] = type(exc).__name__
        summary["step1_error"] = str(exc)
        summary["step1_traceback"] = traceback.format_exc()

    try:
        from mujoco_playground import registry

        registry.load("Go1JoystickFlatTerrain", config_overrides={"impl": "warp"})
        summary["go1_warp_load"] = True
    except Exception as exc:  # pylint: disable=broad-exception-caught
        summary["go1_warp_load"] = False
        summary["go1_warp_error_type"] = type(exc).__name__
        summary["go1_warp_error"] = str(exc)

    path = artifact_dir / "step1_playground_capability_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
