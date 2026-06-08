#!/usr/bin/env python3
"""Minimal smoke tests before running expensive Isaac Sim workflows."""

import importlib
import os
import sys


MODULES = [
    "torch",
    "isaaclab",
]


def main() -> int:
    for name in MODULES:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        print(f"{name}: ok ({version})")

    if os.environ.get("OMNI_KIT_ACCEPT_EULA", "").upper() in {"YES", "Y", "1", "TRUE"}:
        module = importlib.import_module("isaacsim")
        version = getattr(module, "__version__", "unknown")
        print(f"isaacsim: ok ({version})")

        simulation_app = None
        if hasattr(module, "SimulationApp"):
            simulation_app = module.SimulationApp({"headless": True})

        for name in ["isaaclab_tasks", "isaaclab_rl"]:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "unknown")
            print(f"{name}: ok ({version})")

        if simulation_app is not None:
            simulation_app.close()
    else:
        print("isaacsim and task imports skipped; set OMNI_KIT_ACCEPT_EULA=YES after accepting NVIDIA Omniverse EULA")

    print("basic smoke test completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
