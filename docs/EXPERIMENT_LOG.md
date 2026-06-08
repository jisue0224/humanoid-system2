# Experiment Log

## 2026-06-08

Initial restart after lost workspace.

Known server state:

- Workspace: `/workspace`, ephemeral.
- GPU: NVIDIA RTX 3090 x1.
- Driver/CUDA from `nvidia-smi`: 550.107.02 / CUDA 12.4.
- Python: 3.10.12.
- Docker: unavailable in current container.

Decision:

- Try Isaac Lab pip/source path first.
- Docker is not available inside this VESSL session.
- Fall back to MuJoCo Humanoid if Isaac Sim or Isaac Lab cannot run headlessly.

Install note:

- First `isaaclab[isaacsim,all]==2.1.0` attempt reached Isaac Lab dependency resolution but failed while building `flatdict==4.0.1`.
- Error cause: `ModuleNotFoundError: No module named 'pkg_resources'` inside isolated build environment.
- Mitigation added: install `setuptools<80`, `wheel`, and preinstall `flatdict==4.0.1 --no-build-isolation` before Isaac Lab.
- Second attempt installed Isaac Sim 4.5.0 and Isaac Lab 2.1.0 successfully.
- `isaaclab_tasks` was still unavailable from the pip-only install, so source tag `v2.1.0` is used under ignored `external/IsaacLab` to install bundled tasks/RL/assets.
- `isaacsim` import prompts for NVIDIA Omniverse EULA. Do not set `OMNI_KIT_ACCEPT_EULA=YES` unless the user has accepted the license.
- Source grep confirms `Isaac-Velocity-Flat-H1-v0` exists under `manager_based/locomotion/velocity/config/h1`.
- Cleaned pip cache after installation to reduce root inode pressure.
- After EULA acceptance, raw `isaacsim` import works, but task imports require an Isaac Sim `SimulationApp` bootstrap first because `isaacsim.core.*` APIs are extension-backed.

Current Step 1 status:

- Isaac Lab install path: viable.
- Isaac Sim package install: successful.
- H1 task source: present.
- Docker path: unavailable because `docker` is not installed in this container.
- EULA: accepted by user on 2026-06-08.
- Remaining blocker: current VESSL container exposes CUDA compute but not a working NVIDIA Vulkan graphics device.

Next command after EULA acceptance:

```bash
source env_isaaclab/bin/activate
OMNI_KIT_ACCEPT_EULA=YES python scripts/smoke_isaac.py
```

Post-EULA result:

- `isaacsim` import starts.
- `SimulationApp({"headless": True})` reaches Kit startup.
- Initial missing system libraries were installed: `libGL`, `libGLX`, `libSM`, `libXt`, Vulkan loader/tools.
- `vulkaninfo --summary` still lists only `llvmpipe`, not RTX 3090.
- Installing `libnvidia-gl-550=550.90.12` adds NVIDIA ICD but `vulkaninfo` still fails with `ERROR_INCOMPATIBLE_DRIVER`.
- Installing newer `550.163.01` NVIDIA user-space causes `nvidia-smi` driver/library mismatch against host kernel `550.107.02`; reverted to `550.90.12` to restore compute.

Interpretation:

- Isaac Lab is installable on this server.
- The current container was likely launched with `NVIDIA_DRIVER_CAPABILITIES=compute,utility`, not graphics/display.
- H1 Isaac Sim execution probably requires relaunching VESSL with graphics-capable NVIDIA runtime or an Isaac Sim-ready image.

## H1 100 Step Smoke Run

Command:

```bash
source env_isaaclab/bin/activate
OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 TERM=xterm \
  external/IsaacLab/isaaclab.sh -p scripts/run_h1_100_steps.py \
  --headless --device cuda:0 --num_envs 1 --steps 100 \
  --task Isaac-Velocity-Flat-H1-v0
```

Result:

- `Isaac-Velocity-Flat-H1-v0` config parsed.
- Gym environment created.
- Observation space: `Dict('policy': Box(-inf, inf, (1, 69), float32))`.
- Action space: `Box(-inf, inf, (1, 19), float32)`.
- Environment reset succeeded.
- 100 random-action steps completed.
- Exit code: 0.

Persistent warning/error:

- Isaac Sim reports `Driver Version: 0 | Graphics API: Vulkan`.
- `gpu.foundation.plugin` reports `No device could be created`.
- `PhysXFoundation` reports it cannot get GPU foundation / graphics.

Interpretation:

- Physics stepping for H1 works in headless mode despite graphics initialization errors.
- Rendering, cameras, and RGB/depth sensors are still high risk until NVIDIA Vulkan/graphics device enumeration is fixed.

## H1 Pretrained Policy Command Rollouts

Pretrained checkpoint:

- Task: `Isaac-Velocity-Flat-H1-v0`
- Workflow: `rsl_rl`
- Status: available
- Cached path: `.pretrained_checkpoints/rsl_rl/Isaac-Velocity-Flat-H1-v0/checkpoint.pt`

Forward command rollout:

```bash
OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 TERM=xterm \
  external/IsaacLab/isaaclab.sh -p scripts/rollout_h1_policy_commands.py \
  --headless --device cuda:0 --num_envs 1 --steps 100 \
  --task Isaac-Velocity-Flat-H1-v0 \
  --case forward_vx_1 --vx 1.0 --vy 0.0 --yaw 0.0 \
  --output experiments/logs/h1_forward_metrics.json
```

Result:

- Duration: 2.0 s.
- Start position: `[0.0, 0.0, 1.05]`.
- End position: `[1.624, 0.057, 0.900]`.
- Delta x: `+1.624 m`.
- Delta y: `+0.057 m`.
- Delta heading: `-0.0015 rad`.
- Mean base x velocity: `0.819 m/s`.
- Done count: `0`.

Yaw command rollout:

```bash
OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 TERM=xterm \
  external/IsaacLab/isaaclab.sh -p scripts/rollout_h1_policy_commands.py \
  --headless --device cuda:0 --num_envs 1 --steps 100 \
  --task Isaac-Velocity-Flat-H1-v0 \
  --case yaw_0_5 --vx 0.0 --vy 0.0 --yaw 0.5 \
  --output experiments/logs/h1_yaw_metrics.json
```

Result:

- Duration: 2.0 s.
- Start heading: `0.0 rad`.
- End heading: `1.041 rad`.
- Delta heading: `+1.041 rad`.
- Mean base yaw velocity: `0.532 rad/s`.
- Delta x: `+0.011 m`.
- Delta y: `+0.004 m`.
- Done count: `0`.

Interpretation:

- The pretrained H1 flat velocity policy is usable.
- `vx=1.0` produces forward locomotion.
- `yaw=0.5` produces in-place turning with minimal translation.
- This is sufficient to proceed with physics-only high-level navigation.
