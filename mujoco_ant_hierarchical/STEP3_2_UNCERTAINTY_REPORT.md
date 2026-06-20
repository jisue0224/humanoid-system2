# Hierarchical MuJoCo Ant Step 3-2 Uncertainty Report

Date: 2026-06-20

## Goal

Add a static obstacle to the hierarchical Ant scene and verify that progress-based uncertainty separates obstacle and no-obstacle rollouts.

## Setup

- Controller: frozen `jren123/sac-ant-v4` with target-aligned observation and distance action scaling.
- Goal: `(5.0, 0.0)`
- Static obstacle: red box centered at `(2.5, 0.0, 0.45)`, half-size `(0.45, 1.15, 0.45)`
- Episodes per scenario: `10`
- Max steps: `220`
- Success radius: `0.75`

Uncertainty formula:

```python
progress = distance[t - 5] - distance[t]
uncertainty = max(0, min_progress - progress)
```

Parameters:

```text
window = 5
min_progress = 0.02
threshold = 90th percentile of no_obstacle uncertainty = 0.02
```

## Result

| Scenario | Success rate | Mean final dist | Mean uncertainty | p90 uncertainty | Max uncertainty | Fraction >= threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no_obstacle | `0.9` | `0.721 m` | `0.012` | `0.020` | `0.328` | `0.262` |
| obstacle | `0.0` | `3.536 m` | `0.058` | `0.199` | `0.752` | `0.437` |

## Interpretation

The obstacle condition is clearly harder:

- success drops from `0.9` to `0.0`
- mean final distance increases from `0.721 m` to `3.536 m`
- mean uncertainty increases by about `4.8x`
- p90 uncertainty increases from `0.020` to `0.199`

The separation is useful but not perfectly clean. The no-obstacle condition still produces some threshold crossings because the refined controller intentionally slows down near the goal, which reduces short-window progress and can raise progress uncertainty. For LLM triggering, this means a cooldown and/or a slightly stricter threshold may be needed to avoid unnecessary calls near successful goal arrival.

## Artifacts

- `mujoco_ant_hierarchical/artifacts/step3_2_uncertainty/obstacle_uncertainty_summary.json`
- `mujoco_ant_hierarchical/artifacts/step3_2_uncertainty/obstacle_uncertainty_trajectories.png`
- `mujoco_ant_hierarchical/artifacts/step3_2_uncertainty/obstacle_uncertainty_traces.png`

## Decision

Step 3-2 passes as a blocking/stuckness signal. Continue to Step 3-3 only with the caveat that trigger cooldown or goal-near suppression should be used to avoid LLM calls caused purely by normal near-goal slowdown.
