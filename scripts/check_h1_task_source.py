#!/usr/bin/env python3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK_INIT = (
    ROOT
    / "external"
    / "IsaacLab"
    / "source"
    / "isaaclab_tasks"
    / "isaaclab_tasks"
    / "manager_based"
    / "locomotion"
    / "velocity"
    / "config"
    / "h1"
    / "__init__.py"
)


def main() -> int:
    if not TASK_INIT.exists():
        print(f"missing: {TASK_INIT}")
        return 1
    text = TASK_INIT.read_text(encoding="utf-8")
    task_id = "Isaac-Velocity-Flat-H1-v0"
    if task_id not in text:
        print(f"missing task id: {task_id}")
        return 1
    print(f"found task id: {task_id}")
    print(f"source file: {TASK_INIT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
