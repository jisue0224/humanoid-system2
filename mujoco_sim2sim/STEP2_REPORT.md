# Step 2 Report

Status: blocked before pretrained rollout.

Date: 2026-06-20

## Goal

Attempt to run the Isaac Lab RSL-RL pretrained H1 velocity policy in MuJoCo.

Target Isaac Lab baseline:

```text
forward command, 100 Isaac steps = 2.0 s:
  delta_x = +1.624 m
  mean base x velocity = 0.819 m/s

yaw command, 100 Isaac steps = 2.0 s:
  heading change = +1.041 rad
  mean yaw velocity = 0.532 rad/s
```

## Current Environment

Available:

- `mujoco==3.9.0`
- `torch==2.5.1+cpu`
- MuJoCo Menagerie H1 MJCF
- EGL rendering

Unavailable in the current fresh workspace:

- `isaaclab`
- `isaaclab_tasks`
- `rsl_rl`
- Isaac Lab pretrained H1 checkpoint cache

Filesystem search found no local `checkpoint.pt` or H1 pretrained `.pt` file under `/workspace` or `/root`.

## Isaac Lab Interface Reconstructed

Source inspected from a sparse checkout of `isaac-sim/IsaacLab`.

Flat H1 uses the rough H1 config with height scan removed:

- `H1FlatEnvCfg`
- `H1FlatPPORunnerCfg`

Policy architecture:

```text
actor_hidden_dims = [128, 128, 128]
activation = elu
actor_obs_normalization = false
```

Observation dimension for flat H1:

```text
base_lin_vel        3
base_ang_vel        3
projected_gravity   3
velocity_commands   3
joint_pos_rel      19
joint_vel          19
last_action        19
---------------------
total              69
```

Action dimension:

```text
19
```

Action semantics from Isaac Lab config:

```python
joint_pos = JointPositionActionCfg(
    asset_name="robot",
    joint_names=[".*"],
    scale=0.5,
    use_default_offset=True,
)
```

So the MuJoCo wrapper uses:

```text
target_joint_pos = default_joint_pos + 0.5 * policy_action
```

Then applies MuJoCo torques with a PD wrapper using Isaac Lab H1 gains.

## Checkpoint Status

Expected local path:

```text
mujoco_sim2sim/checkpoints/rsl_rl_Isaac-Velocity-Flat-H1-v0_checkpoint.pt
```

Result:

```text
checkpoint not found
```

I also tried the SourceForge mirror path suggested by search results:

```text
https://sourceforge.net/projects/nvidia-isaac-lab.mirror/files/training-checkpoints-v2.3/rsl_rl_Isaac-Velocity-Flat-H1-v0_checkpoint.pt/download
```

but it returned `404` from this environment. The release RSS listing exposes source-code archives but not that checkpoint file path.

Therefore the actual pretrained policy inference could not be run yet.

## MuJoCo Wrapper Smoke Test

Script:

```bash
MUJOCO_GL=egl python mujoco_sim2sim/step2_policy_transfer_attempt.py \
  --seconds 2.0 --vx 1.0 --vy 0.0 --yaw 0.0
```

Because the checkpoint is missing, the script ran:

```text
fallback_mode = zero_action_pd_hold
```

Forward-command fallback result:

```text
used_policy = false
delta_x = 1.109 m
delta_y = 0.001 m
final_yaw = -2.107 rad
collapsed = true
```

Yaw-command fallback result:

```bash
MUJOCO_GL=egl python mujoco_sim2sim/step2_policy_transfer_attempt.py \
  --output_dir mujoco_sim2sim/artifacts/step2_yaw \
  --seconds 2.0 --vx 0.0 --vy 0.0 --yaw 0.5
```

```text
used_policy = false
delta_x = 1.109 m
delta_y = 0.001 m
final_yaw = -2.107 rad
collapsed = true
```

The identical forward/yaw result is expected because no policy was loaded. The command is only an observation input; zero-action fallback ignores it.

## Interpretation

Step 2 is blocked at the checkpoint/inference stage, not at MuJoCo model loading.

What was validated:

- The MuJoCo side can construct a 69D observation vector.
- The MuJoCo side can expose a 19D action interface matching Isaac Lab's joint-position target semantics.
- The default H1 joint order and action dimension can be mapped.
- A PD wrapper with Isaac Lab gains can be applied to MuJoCo torque motors.

What failed or remains unvalidated:

- The pretrained RSL-RL checkpoint is not available in this fresh workspace.
- Without checkpoint weights, `vx=1.0` and `yaw=0.5` policy rollouts cannot be compared to Isaac Lab.
- Zero-action PD hold does not stabilize the Menagerie H1. The robot collapses, so the learned policy or a stronger balancing controller is required.

## Next Required Input

To complete Step 2 properly, provide or recover one of:

1. The cached checkpoint from the previous Isaac Lab session:

```text
.pretrained_checkpoints/rsl_rl/Isaac-Velocity-Flat-H1-v0/checkpoint.pt
```

2. A direct downloadable URL for:

```text
rsl_rl_Isaac-Velocity-Flat-H1-v0_checkpoint.pt
```

3. A working Isaac Lab environment so `get_published_pretrained_checkpoint("rsl_rl", "Isaac-Velocity-Flat-H1-v0")` can download/cache it again.

Once checkpoint weights are available, rerun:

```bash
MUJOCO_GL=egl python mujoco_sim2sim/step2_policy_transfer_attempt.py \
  --checkpoint mujoco_sim2sim/checkpoints/rsl_rl_Isaac-Velocity-Flat-H1-v0_checkpoint.pt \
  --seconds 2.0 --vx 1.0 --vy 0.0 --yaw 0.0

MUJOCO_GL=egl python mujoco_sim2sim/step2_policy_transfer_attempt.py \
  --checkpoint mujoco_sim2sim/checkpoints/rsl_rl_Isaac-Velocity-Flat-H1-v0_checkpoint.pt \
  --output_dir mujoco_sim2sim/artifacts/step2_yaw \
  --seconds 2.0 --vx 0.0 --vy 0.0 --yaw 0.5
```

## Artifacts

- Script: `mujoco_sim2sim/step2_policy_transfer_attempt.py`
- Forward fallback summary: `mujoco_sim2sim/artifacts/step2/step2_policy_transfer_attempt.json`
- Forward fallback final render: `mujoco_sim2sim/artifacts/step2/step2_final.png`
- Yaw fallback summary: `mujoco_sim2sim/artifacts/step2_yaw/step2_policy_transfer_attempt.json`
- Yaw fallback final render: `mujoco_sim2sim/artifacts/step2_yaw/step2_final.png`
