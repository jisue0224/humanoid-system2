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

## Step 2 Goal Navigation

Implementation:

- Added `scripts/goal_nav_h1.py`.
- Loads the published RSL-RL checkpoint for `Isaac-Velocity-Flat-H1-v0`.
- Runs multiple goal-navigation episodes in parallel envs to avoid repeated Isaac Sim reset hangs observed in this headless container.
- High-level controller computes a world-frame target direction from current H1 base position to goal.
- Controller outputs `(vx, vy, yaw)` by turning toward the goal and moving forward when heading error is small.
- `vy` is kept in the command interface, but defaults to `0.0` because the pretrained H1 flat policy was validated mainly for forward velocity and yaw command following.

Command:

```bash
source env_isaaclab/bin/activate
OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 TERM=xterm \
  external/IsaacLab/isaaclab.sh -p scripts/goal_nav_h1.py \
  --headless --device cuda:0 --num_envs 30 --episodes 30 --max_steps 700 \
  --task Isaac-Velocity-Flat-H1-v0 \
  --output_dir experiments/goal_nav
```

Goal set:

- `(5.0, 0.0)`
- `(0.0, 5.0)`
- `(5.0, 5.0)`
- `(-3.0, 4.0)`

Result:

- Episodes: `30`.
- Success criterion: final goal distance `< 0.5 m`.
- Success rate: `1.000`.
- All environments succeeded by step `363`.
- Per-goal result:
  - `(5.0, 0.0)`: `8/8`, mean final distance `0.342 m`, max final distance `0.347 m`.
  - `(0.0, 5.0)`: `8/8`, mean final distance `0.329 m`, max final distance `0.335 m`.
  - `(5.0, 5.0)`: `7/7`, mean final distance `0.487 m`, max final distance `0.492 m`.
  - `(-3.0, 4.0)`: `7/7`, mean final distance `0.313 m`, max final distance `0.322 m`.

Artifacts:

- Summary JSON: `experiments/goal_nav/metrics/goal_nav_summary.json`.
- Per-episode overhead trajectory plots: `experiments/goal_nav/plots/episode_*.png`.
- Log: `experiments/logs/goal_nav_30ep.log` locally, ignored by git.

Interpretation:

- Physics-based H1 goal navigation works with the pretrained velocity policy and a rule-based high-level controller.
- The controller can reach forward, side, diagonal, and behind/side goals by combining yaw alignment with forward walking.
- This is a usable baseline for Step 3 obstacle insertion and progress-based uncertainty measurement.

## Step 3 Static Obstacle and Progress Uncertainty

Implementation:

- Added `scripts/obstacle_uncertainty_h1.py`.
- Added `scripts/merge_obstacle_uncertainty_results.py`.
- Static obstacle uses Isaac Lab `RigidObjectCfg` with `CuboidCfg`.
- Default obstacle size: `(1.0, 1.0, 1.0) m`.
- Default obstacle position: `(2.5, 0.0, 0.5)`, between start `(0.0, 0.0)` and goal `(5.0, 0.0)`.
- Obstacle position and size are configurable with `--obstacle_pos` and `--obstacle_size`.
- `base_contact` termination is disabled in this runner so obstacle contact can be observed as stalled physics instead of immediate episode reset.
- Isaac Sim was run once per scenario because creating a second env inside the same process can hang in this headless VESSL container.

Commands:

```bash
source env_isaaclab/bin/activate
OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 TERM=xterm \
  external/IsaacLab/isaaclab.sh -p scripts/obstacle_uncertainty_h1.py \
  --headless --device cuda:0 --num_envs 10 --episodes_per_scenario 10 --max_steps 700 \
  --scenario no_obstacle \
  --task Isaac-Velocity-Flat-H1-v0 \
  --output_dir experiments/obstacle_uncertainty

OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 TERM=xterm \
  external/IsaacLab/isaaclab.sh -p scripts/obstacle_uncertainty_h1.py \
  --headless --device cuda:0 --num_envs 10 --episodes_per_scenario 10 --max_steps 700 \
  --scenario obstacle \
  --task Isaac-Velocity-Flat-H1-v0 \
  --output_dir experiments/obstacle_uncertainty

python scripts/merge_obstacle_uncertainty_results.py \
  --output_dir experiments/obstacle_uncertainty
```

Uncertainty:

- Window: recent `5` steps.
- Definition: `max(0, min_progress - (distance[t-window] - distance[t]))`.
- `min_progress`: `0.02 m` over the 5-step window.
- Common A/B trigger threshold: p90 over the combined scenario A+B uncertainty distribution.
- Combined p90 threshold: `0.0247`.

Scenario A: no obstacle:

- Episodes: `10`.
- Success rate: `1.000`.
- Mean final distance: `0.484 m`.
- Mean uncertainty: `0.00041`.
- Mean max uncertainty: `0.0211`.
- Mean trigger count with combined threshold: `0.0`.

Scenario B: static obstacle:

- Episodes: `10`.
- Success rate: `0.000`.
- Done rate: `0.000` because base-contact termination is disabled.
- Mean final distance: `3.094 m`.
- Mean min distance: `3.074 m`.
- Mean uncertainty: `0.0168`.
- Mean max uncertainty: `0.0280`.
- Mean trigger count with combined threshold: `18.3`.
- Mean inside-obstacle steps: `0.0`.

Obstacle behavior:

- H1 does not pass through the obstacle.
- With default termination enabled, contact ended the episode around base `x ~= 1.91 m`, just before the obstacle front face at `x = 2.0 m`.
- With base-contact termination disabled for analysis, H1 remains blocked/stalled near the obstacle front and keeps trying to move forward.
- This confirms physics collision is active and the obstacle is not just visual.

Artifacts:

- Combined summary JSON: `experiments/obstacle_uncertainty/metrics/obstacle_uncertainty_summary.json`.
- Scenario summaries:
  - `experiments/obstacle_uncertainty/metrics/scenario_a_no_obstacle_summary.json`
  - `experiments/obstacle_uncertainty/metrics/scenario_b_static_obstacle_summary.json`
- Plots:
  - `experiments/obstacle_uncertainty/plots/scenario_a_vs_b_trajectory.png`
  - `experiments/obstacle_uncertainty/plots/scenario_a_vs_b_uncertainty.png`
  - scenario-specific trajectory and uncertainty plots.

Interpretation:

- Scenario A keeps uncertainty effectively low while reaching the goal.
- Scenario B creates a clear progress stall near the obstacle.
- The progress-based uncertainty measure separates the two cases under the shared A/B p90 threshold.
- This is a usable trigger signal for Step 4/5 camera capture and LLM intervention.
