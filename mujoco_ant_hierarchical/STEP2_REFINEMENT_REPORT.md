# Hierarchical MuJoCo Ant Step 2 Refinement

Date: 2026-06-20

## Goal

Fix the overshoot / loose-orbit behavior from the target-aligned frozen Ant controller without retraining locomotion.

## Option 1 Implemented

No learning was used. The low-level policy remains frozen:

- `jren123/sac-ant-v4`
- `SAC-Ant-v4.zip`
- `training_timesteps = 0`

The refinement adds distance-based action scaling after the frozen policy predicts an action:

```python
alpha = (distance - success_radius) / (slow_radius - success_radius)
alpha = max(0.0, min(1.0, alpha))
action_scale = min_action_scale + (1.0 - min_action_scale) * alpha
action = frozen_policy_action * action_scale
```

Selected parameters after a small sweep:

| Parameter | Value |
| --- | ---: |
| success radius | `0.75` |
| slow radius | `7.0` |
| min action scale | `0.3` |

Interpretation: the Ant uses full frozen-policy action far from the goal, then gradually reduces action magnitude as it approaches the target. This provides the missing "slow down near goal" signal while keeping the locomotion policy frozen.

## Result

Same 10 episodes and seeds as the previous smoke test.

| Metric | Baseline target-aligned | Refined action scaling |
| --- | ---: | ---: |
| success rate | `0.6` | `0.9` |
| successes | `6 / 10` | `9 / 10` |
| mean final distance | `1.610 m` | `0.736 m` |
| median final distance | not recorded | `0.740 m` |

Final distances:

```text
baseline: 0.748, 2.581, 1.557, 4.177, 3.688, 0.705, 0.599, 0.702, 0.658, 0.689
refined:  0.740, 0.793, 0.660, 0.742, 0.737, 0.726, 0.749, 0.726, 0.744, 0.740
```

The one refined failure is only slightly outside the strict `0.75 m` radius:

```text
goal=(0, 5), final_distance=0.793 m
```

## Artifacts

- `mujoco_ant_hierarchical/artifacts/step2_refinement/refinement_summary.json`
- `mujoco_ant_hierarchical/artifacts/step2_refinement/refinement_trajectories.png`

## Decision

Option 1 is sufficient for now.

The success rate is above the requested `0.8` threshold and the obvious overshoot / loose-orbit pattern is mostly gone. Option 2 small RL fine-tuning is skipped for this checkpoint.

The next step can move to vision + uncertainty + LLM experiments on top of this refined frozen-policy controller.
