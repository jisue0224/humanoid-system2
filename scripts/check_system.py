#!/usr/bin/env python3
import importlib.util
import os
import platform
import shutil
import subprocess
import sys


def run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> None:
    print("== Python ==")
    print(sys.version.replace("\n", " "))
    print("executable:", sys.executable)
    print("platform:", platform.platform())
    print()

    print("== OS ==")
    if os.path.exists("/etc/os-release"):
        with open("/etc/os-release", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(("PRETTY_NAME", "VERSION_ID")):
                    print(line.strip())
    print("glibc:", run(["ldd", "--version"]).splitlines()[0])
    print()

    print("== GPU ==")
    print(run(["nvidia-smi"]))
    print()

    print("== Storage ==")
    print(run(["df", "-h", "/workspace", os.path.expanduser("~")]))
    print(run(["df", "-i", "/workspace", os.path.expanduser("~")]))
    print()

    print("== Tools ==")
    for tool in ["git", "docker", "python3.10", "python3.11", "conda", "uv"]:
        print(f"{tool}: {shutil.which(tool) or 'missing'}")
    print()

    print("== Python Packages ==")
    for mod in ["torch", "isaacsim", "isaaclab", "isaaclab_tasks", "gymnasium", "mujoco"]:
        print(f"{mod}: {'installed' if has_module(mod) else 'missing'}")


if __name__ == "__main__":
    main()

