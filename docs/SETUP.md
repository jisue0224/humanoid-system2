# Setup

Current cluster constraint: `/workspace` is ephemeral across VESSL restarts. Keep code in GitHub and treat this directory as disposable.

## Environment Snapshot

Run:

```bash
python scripts/check_system.py
```

## Isaac Lab Plan

The first attempt uses Isaac Lab pip packages on Ubuntu 22.04 / Python 3.10:

```bash
bash scripts/install_isaaclab.sh
```

Then verify:

```bash
source env_isaaclab/bin/activate
python scripts/smoke_isaac.py
python scripts/check_h1_task_source.py
```

If Isaac Sim prompts for the NVIDIA Omniverse EULA on first run, accept only if you are allowed to use it in this environment.

For non-interactive smoke tests after you have accepted the EULA:

```bash
OMNI_KIT_ACCEPT_EULA=YES python scripts/smoke_isaac.py
```

## Fallback Plan

If Isaac Lab cannot run headlessly on VESSL, use MuJoCo:

- `gymnasium[mujoco]`
- `mujoco`
- `stable-baselines3` or a small PPO implementation

The fallback keeps the same high-level experiment contract:

- low-level policy receives `(vx, vy, yaw)`
- goal controller produces velocity commands
- uncertainty is progress based
- LLM intervenes only above threshold
