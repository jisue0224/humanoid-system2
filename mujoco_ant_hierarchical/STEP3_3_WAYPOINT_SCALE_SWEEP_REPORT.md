# Step 3-3 Waypoint Scale Sweep

Date: 2026-06-20

## Setup

This sweep keeps the refined Step 3-3 logic:

- waypoint timeout: `50` steps
- uncertainty warm-up: `15` steps
- recent waypoint list included in the prompt
- final goal scaling unchanged:
  - `slow_radius=7.0`
  - `min_action_scale=0.3`

For waypoint targets, the sweep uses:

- `waypoint_slow_radius=7.0`
- `waypoint_min_action_scale in {0.7, 0.85}`

Note: the previous `1.0` run used `waypoint_slow_radius=0.0`, which disables
scaling entirely. For intermediate values, the slow radius must be nonzero.

## Quick Sweep

Only `uncertainty_switching` was swept for 10 episodes.

| waypoint_min_action_scale | Episodes | Success rate | Mean final distance | Mean LLM calls |
| ---: | ---: | ---: | ---: | ---: |
| `0.3` previous/original | `20` | `0.00` | `3.413 m` | `2.50` |
| `0.7` | `10` | `0.20` | `2.689 m` | `3.60` |
| `0.85` | `10` | `0.40` | `2.431 m` | `3.30` |
| `1.0` refined | `20` | `0.20` | `3.004 m` | `3.25` |

Best quick-sweep value: `waypoint_min_action_scale=0.85`.

## Final 20-Episode Run at 0.85

| Condition | Episodes | Success rate | Mean final distance | Mean LLM calls | Total LLM calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| always_llm | `20` | `0.10` | `2.815 m` | `5.25` | `105` |
| uncertainty_switching | `20` | `0.15` | `2.892 m` | `3.40` | `68` |

For comparison, the previous `1.0` refined `uncertainty_switching` result was:

| Condition | waypoint scaling | Episodes | Success rate | Mean final distance | Mean LLM calls | Total LLM calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| uncertainty_switching | disabled / `1.0` | `20` | `0.20` | `3.004 m` | `3.25` | `65` |

## Interpretation

The quick sweep suggested `0.85` was best, but the 20-episode run regressed from
the 10-episode estimate:

- `0.85` produced better mean final distance than the `1.0` refined run
  (`2.892 m` vs `3.004 m` for uncertainty switching).
- `0.85` produced lower success rate than the `1.0` refined run
  (`0.15` vs `0.20` for uncertainty switching).
- `always_llm` at `0.85` improved over the prior `1.0` always run
  (`0.10` vs `0.00` success), but it used many more calls (`105`).
- `uncertainty_switching` still uses far fewer calls than `always_llm`
  at the same scale (`68` vs `105`).

Overall, `0.85` is a reasonable waypoint scaling value, but it does not fully
solve the subgoal transition problem. The result now supports the project claim
better than the previous always-on setup because switching gets comparable or
better task performance with substantially fewer LLM calls.

## Deferred Gate-Passing Change

The optional gate-passing waypoint transition was not implemented in this round.
The next targeted change should replace circular waypoint completion with a
pass-through condition around the obstacle corridor, for example:

- upper route considered complete once `x > obstacle_x_max` and `y > obstacle_y_max`
- lower route considered complete once `x > obstacle_x_max` and `y < obstacle_y_min`

That should match the semantics of a waypoint as a gate rather than a stop point.

## Artifacts

- `mujoco_ant_hierarchical/artifacts/step3_3_llm_scale_sweep/scale_070/uncertainty_switching/summary.json`
- `mujoco_ant_hierarchical/artifacts/step3_3_llm_scale_sweep/scale_070/uncertainty_switching/trajectories.png`
- `mujoco_ant_hierarchical/artifacts/step3_3_llm_scale_sweep/scale_085/uncertainty_switching/summary.json`
- `mujoco_ant_hierarchical/artifacts/step3_3_llm_scale_sweep/scale_085/uncertainty_switching/trajectories.png`
- `mujoco_ant_hierarchical/artifacts/step3_3_llm_scale085_final/always_llm/summary.json`
- `mujoco_ant_hierarchical/artifacts/step3_3_llm_scale085_final/always_llm/trajectories.png`
- `mujoco_ant_hierarchical/artifacts/step3_3_llm_scale085_final/uncertainty_switching/summary.json`
- `mujoco_ant_hierarchical/artifacts/step3_3_llm_scale085_final/uncertainty_switching/trajectories.png`
