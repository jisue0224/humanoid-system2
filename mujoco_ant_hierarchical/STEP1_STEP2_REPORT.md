# Hierarchical MuJoCo Ant Step 1-2 Report

Date: 2026-06-20

## Decision

Use **Way A: target-aligned observation transform**.

Reason: `Ant-v4` default observation excludes world `x/y` position but includes root orientation and root velocity:

```python
position = self.data.qpos.flat.copy()
velocity = self.data.qvel.flat.copy()

if self._exclude_current_positions_from_observation:
    position = position[2:]

obs = np.concatenate((position, velocity))
```

For the default 27D `Ant-v4` observation:

| Obs slice | Meaning |
| --- | --- |
| `obs[0]` | root z |
| `obs[1:5]` | root quaternion, MuJoCo `wxyz` |
| `obs[5:13]` | joint positions |
| `obs[13:27]` | qvel |
| `obs[13:15]` | root xy velocity |

There is no explicit goal/command slot. However, the pretrained forward locomotion policy can be redirected by expressing root orientation and root xy velocity in a frame where the current target direction is treated as +x.

## Step 1 Probe

Frozen low-level policy:

- `jren123/sac-ant-v4`
- `SAC-Ant-v4.zip`
- no locomotion retraining

Transform:

```python
transformed = obs.copy()
transformed[1:5] = quat_mul(yaw_quat(-target_heading), transformed[1:5])
transformed[13:15] = rotate_xy(transformed[13:15], -target_heading)
```

100-step probe results:

| Goal | Transform | Delta xy | Final distance |
| --- | --- | ---: | ---: |
| `(0, 5)` | no | `(22.327, 1.924)` | `22.571` |
| `(0, 5)` | yes | `(2.425, 5.924)` | `2.604` |
| `(5, 0)` | no | `(22.327, 1.924)` | `17.456` |
| `(5, 0)` | yes | `(6.329, -0.347)` | `1.412` |

This is a strong signal that Way A is valid. The policy is not simply world-frame forward-only; target-frame observation rotation changes the physical trajectory.

Artifact:

- `mujoco_ant_hierarchical/artifacts/step1/observation_frame_check.json`

## Step 2 Smoke Test

Because Way A is geometry-based, no high-level RL training is needed for the smoke test.

Setup:

- Environment: open-field `Ant-v4`, equivalent to goal-reaching without maze walls.
- Episodes: `10`
- Max steps: `300`
- Success radius: `0.75 m`
- Low-level policy: frozen `jren123/sac-ant-v4`
- High-level controller: target heading from current xy to goal xy
- Training timesteps: `0`

Result:

| Metric | Value |
| --- | ---: |
| success rate | `0.6` |
| successes | `6 / 10` |
| mean final distance | `1.610 m` |

Final distances:

```text
0.748, 2.581, 1.557, 4.177, 3.688, 0.705, 0.599, 0.702, 0.658, 0.689
```

Artifacts:

- `mujoco_ant_hierarchical/artifacts/step2/directional_smoke_summary.json`
- `mujoco_ant_hierarchical/artifacts/step2/directional_smoke_trajectories.png`

## Interpretation

This passes the Step 2 gate.

The previous goal-conditioned Ant attempt had `success_rate = 0.0` and trajectories mostly stayed near the origin. This hierarchical version reaches 6/10 goals without retraining locomotion. The failed episodes are not directionless circling; they generally move toward the target and then overshoot or enter a loose orbit/limit cycle.

That means the next engineering target is not “learn to walk and navigate from scratch.” It is controller refinement around the frozen forward policy:

- slow down or stop near goal,
- reduce overshoot,
- possibly add waypoint hysteresis,
- then test obstacle/uncertainty/LLM.

## Stop Point

Per the requested workflow, stop after Step 2 and report before continuing to vision + uncertainty + LLM.
