# Step 2 Report

Status: pretrained rollout attempted; sim2sim does not work yet.

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

Initially unavailable in the fresh workspace:

- `isaaclab`
- `isaaclab_tasks`
- `rsl_rl`
- Isaac Lab pretrained H1 checkpoint cache

Filesystem search initially found no local `checkpoint.pt` or H1 pretrained `.pt` file under `/workspace` or `/root`.

I then reinstalled Isaac Lab v2.1.0 with the project script and fetched the published checkpoint through Isaac Lab:

```bash
bash scripts/install_isaaclab.sh

source env_isaaclab/bin/activate
OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 TERM=xterm \
  external/IsaacLab/isaaclab.sh -p scripts/check_h1_pretrained.py \
  --headless --device cuda:0
```

The first `check_h1_pretrained.py` run failed because the fresh container was missing graphics support libraries such as `libSM.so.6` and `libXt.so.6`. Installing the same minimal OS libraries used in the previous Isaac session fixed that import-level issue:

```bash
apt-get install -y \
  libsm6 libxt6 libxrender1 libxext6 libx11-6 libxcb1 libxau6 libxdmcp6 \
  libvulkan1 vulkan-tools mesa-vulkan-drivers
```

Isaac Sim still reports no Vulkan GPU, as expected for this VESSL container, but checkpoint retrieval succeeded:

```text
Fetching pre-trained checkpoint : http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Velocity-Flat-H1-v0/checkpoint.pt
[PRETRAINED] available path=/workspace/humanoid-system2/.pretrained_checkpoints/rsl_rl/Isaac-Velocity-Flat-H1-v0/checkpoint.pt
```

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

Recovered Isaac Lab cache path:

```text
.pretrained_checkpoints/rsl_rl/Isaac-Velocity-Flat-H1-v0/checkpoint.pt
```

Copied local MuJoCo-side path:

```text
mujoco_sim2sim/checkpoints/rsl_rl_Isaac-Velocity-Flat-H1-v0_checkpoint.pt
```

The checkpoint is intentionally ignored by git through:

```text
mujoco_sim2sim/checkpoints/
```

File size:

```text
1.1M
```

Checkpoint structure:

```text
top-level keys: model_state_dict, optimizer_state_dict, iter, infos
actor.0.weight: [128, 69]
actor.0.bias:   [128]
actor.2.weight: [128, 128]
actor.2.bias:   [128]
actor.4.weight: [128, 128]
actor.4.bias:   [128]
actor.6.weight: [19, 128]
actor.6.bias:   [19]
std:            [19]
```

The MuJoCo wrapper maps `actor.*` checkpoint keys to the local actor MLP's `net.*` keys.

## MuJoCo Wrapper Smoke Test Without Checkpoint

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

## Pretrained Policy Rollout Attempt

After recovering the checkpoint, I ran the same wrapper with the actual actor weights.

Forward command:

```bash
source env_isaaclab/bin/activate
MUJOCO_GL=egl python mujoco_sim2sim/step2_policy_transfer_attempt.py \
  --output_dir mujoco_sim2sim/artifacts/step2_policy_forward \
  --seconds 2.0 --vx 1.0 --vy 0.0 --yaw 0.0
```

Result:

```text
used_policy = true
delta_x = +0.950 m
delta_y = +1.270 m
final_yaw = +0.058 rad
collapsed = true
```

Collapse timing:

```text
base_z < 0.9 m at 1.102 s
base_z < 0.6 m at 1.382 s
base_z < 0.3 m at 1.522 s
max_action_norm = 5.544
max_torque_norm = 364.418
```

Yaw command:

```bash
source env_isaaclab/bin/activate
MUJOCO_GL=egl python mujoco_sim2sim/step2_policy_transfer_attempt.py \
  --output_dir mujoco_sim2sim/artifacts/step2_policy_yaw \
  --seconds 2.0 --vx 0.0 --vy 0.0 --yaw 0.5
```

Result:

```text
used_policy = true
delta_x = -0.563 m
delta_y = +0.367 m
final_yaw = +1.425 rad
collapsed = true
```

Collapse timing:

```text
base_z < 0.9 m at 1.222 s
base_z < 0.6 m at 1.522 s
base_z < 0.3 m at 1.702 s
max_action_norm = 6.176
max_torque_norm = 335.476
```

Comparison with Isaac Lab baseline:

| Case | Isaac Lab expected | MuJoCo attempt |
|---|---:|---:|
| forward `vx=1.0` | `delta_x=+1.624m`, stable | `delta_x=+0.950m`, `delta_y=+1.270m`, collapsed |
| yaw `yaw=0.5` | heading `+1.041rad`, stable | heading `+1.425rad`, collapsed |

The policy is not transferring correctly yet.

## Interpretation

Step 2 is no longer blocked by checkpoint availability. The pretrained actor loads and runs, but the sim2sim transfer fails dynamically.

What was validated:

- The MuJoCo side can construct a 69D observation vector.
- The MuJoCo side can expose a 19D action interface matching Isaac Lab's joint-position target semantics.
- The default H1 joint order and action dimension can be mapped.
- A PD wrapper with Isaac Lab gains can be applied to MuJoCo torque motors.
- The Isaac Lab published RSL-RL H1 checkpoint can be fetched in this environment after reinstalling Isaac Lab.
- The actor MLP weights load into the MuJoCo wrapper.

What failed or remains unvalidated:

- `vx=1.0` and `yaw=0.5` do not reproduce the Isaac Lab baseline.
- The H1 collapses in both policy rollouts within about 1.4-1.5 seconds.
- Zero-action PD hold also does not stabilize the Menagerie H1.
- The remaining mismatch is likely in one or more of:
  - MuJoCo Menagerie H1 inertial/contact/actuator parameters differ from Isaac Lab `H1_MINIMAL_CFG`.
  - MuJoCo actuator motors are torque motors while Isaac Lab uses implicit actuators.
  - Observation frame/sign conventions may still differ.
  - MuJoCo joint order/name semantics may look aligned but not be exactly the same as Isaac Lab's resolved joint order.
  - Contact/friction/foot geometry differs enough to destabilize the learned gait.

## Next Work

Do not proceed as if sim2sim is solved. The next step should be a focused mismatch audit:

1. Extract Isaac Lab's resolved joint order, default joint positions, action manager scale/offset, and actuator gains from a live Isaac env.
2. Compare them numerically against the MuJoCo wrapper.
3. Try a stronger explicit PD hold or actuator model closer to Isaac implicit actuators.
4. Compare base frame velocity and projected gravity signs using a known pose/velocity.
5. If still unstable, use MuJoCo for rendering but plan to train or fine-tune a MuJoCo H1 locomotion policy instead of expecting direct checkpoint transfer.

## Artifacts

- Script: `mujoco_sim2sim/step2_policy_transfer_attempt.py`
- Forward fallback summary: `mujoco_sim2sim/artifacts/step2/step2_policy_transfer_attempt.json`
- Forward fallback final render: `mujoco_sim2sim/artifacts/step2/step2_final.png`
- Yaw fallback summary: `mujoco_sim2sim/artifacts/step2_yaw/step2_policy_transfer_attempt.json`
- Yaw fallback final render: `mujoco_sim2sim/artifacts/step2_yaw/step2_final.png`
- Forward policy summary: `mujoco_sim2sim/artifacts/step2_policy_forward/step2_policy_transfer_attempt.json`
- Forward policy final render: `mujoco_sim2sim/artifacts/step2_policy_forward/step2_final.png`
- Yaw policy summary: `mujoco_sim2sim/artifacts/step2_policy_yaw/step2_policy_transfer_attempt.json`
- Yaw policy final render: `mujoco_sim2sim/artifacts/step2_policy_yaw/step2_final.png`
