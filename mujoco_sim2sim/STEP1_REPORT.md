# Step 1 Report

Status: passed with expected passive collapse.

Date: 2026-06-20

## Model

Source:

- `mujoco_sim2sim/vendor/mujoco_menagerie/unitree_h1/scene.xml`
- Sparse checkout from `google-deepmind/mujoco_menagerie`
- Menagerie commit checked locally during Step 0: `accb6df`

## Load Check

Command:

```bash
MUJOCO_GL=egl python mujoco_sim2sim/step1_h1_mjcf_smoke.py
```

Result:

```text
nq=26
nv=25
nu=19
njnt=20
has_freejoint=true
actuated_dof=19
hinge_joint_count=19
```

Interpretation:

- `nq=26`: 7 free-base coordinates + 19 actuated joint coordinates.
- `nv=25`: 6 free-base velocities + 19 actuated joint velocities.
- `njnt=20`: one floating base joint plus 19 named hinge joints.
- `nu=19`: 19 motor actuators.
- The actuated DOF count matches the Isaac Lab H1 policy action dimension: `19`.

Actuated joints:

```text
left_hip_yaw
left_hip_roll
left_hip_pitch
left_knee
left_ankle
right_hip_yaw
right_hip_roll
right_hip_pitch
right_knee
right_ankle
torso
left_shoulder_pitch
left_shoulder_roll
left_shoulder_yaw
left_elbow
right_shoulder_pitch
right_shoulder_roll
right_shoulder_yaw
right_elbow
```

## Static Initial Pose

The model loads from the Menagerie `home` keyframe without immediate collapse at `mj_forward`.

Initial state:

```text
base_z=0.980 m
roll=0.000 rad
pitch=0.000 rad
yaw=0.000 rad
```

Rendered artifact:

- `mujoco_sim2sim/artifacts/step1/h1_initial.png`

## Passive Gravity Simulation

The H1 was simulated for `5.0 s` with zero motor control:

```python
data.ctrl[:] = 0.0
mujoco.mj_step(model, data)
```

Final state:

```text
base_z=0.102 m
roll=2.502 rad
pitch=-1.486 rad
yaw=-2.556 rad
root_delta=[-0.496, -0.006, -0.878] m
qpos_delta_norm=3.191
```

Collapse rule used:

```text
base_z < 0.6 or max(abs(roll), abs(pitch)) > 60 deg
```

Result:

```text
passive_zero_torque_collapsed=true
```

Interpretation:

- The MJCF is physically loadable and renders correctly.
- The initial keyframe is a plausible standing/crouched H1 pose.
- With gravity and zero torque, the robot falls quickly. This is expected for a humanoid without a balance controller or a PD hold.
- Step 2 should not assume passive stability. Policy transfer needs the correct actuator mode and observation/action normalization, or at least a PD wrapper if the Isaac policy action represents target joint positions.

## Artifacts

- Summary JSON: `mujoco_sim2sim/artifacts/step1/h1_mjcf_step1_summary.json`
- Initial render: `mujoco_sim2sim/artifacts/step1/h1_initial.png`
- Passive final render: `mujoco_sim2sim/artifacts/step1/h1_passive_final.png`
- Passive trace plot: `mujoco_sim2sim/artifacts/step1/h1_passive_trace.png`

## Step 1 Conclusion

MuJoCo can load and simulate the H1 MJCF. The model has the expected 19 actuated DOF. Passive zero-torque standing is not stable, so Step 2 policy transfer must focus on matching Isaac Lab's action semantics instead of expecting the raw model to stand without control.

Proceed to Step 2 only after user confirmation.
