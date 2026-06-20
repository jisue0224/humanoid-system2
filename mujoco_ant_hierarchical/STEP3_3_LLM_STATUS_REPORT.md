# Hierarchical MuJoCo Ant Step 3-3/3-4 LLM Status

Date: 2026-06-20

## Implemented

Added the LLM waypoint experiment script:

- `mujoco_ant_hierarchical/step3_3_llm_waypoint_experiment.py`

Supported conditions:

```text
policy_only
always_llm
uncertainty_switching
```

The LLM conditions:

- capture real MuJoCo EGL-rendered `overhead` frames at the call step
- optionally capture `ego` frames with `--include_ego`
- send image(s) and numeric state to the configured model
- parse JSON waypoint:

```json
{
  "reasoning": "short",
  "waypoint_x": 0.0,
  "waypoint_y": 0.0
}
```

No waypoint postprocess is applied. If the model returns a waypoint, the hierarchical controller temporarily uses that waypoint as the target. Once the Ant reaches the waypoint radius, it resumes the final goal.

## Comparison Run Completed

Commands:

```bash
source env_isaaclab/bin/activate
python mujoco_ant_hierarchical/step3_3_llm_waypoint_experiment.py --condition policy_only --episodes 20
python mujoco_ant_hierarchical/step3_3_llm_waypoint_experiment.py --condition always_llm --episodes 20 --include_ego
python mujoco_ant_hierarchical/step3_3_llm_waypoint_experiment.py --condition uncertainty_switching --episodes 20 --include_ego
```

Result:

| Condition | Episodes | Success rate | Mean final distance | LLM calls |
| --- | ---: | ---: | ---: | ---: |
| policy_only | `20` | `0.0` | `3.153 m` | `0` |
| always_llm | `20` | `0.05` | `3.335 m` | `54` |
| uncertainty_switching | `20` | `0.0` | `3.413 m` | `50` |

Artifacts:

- `mujoco_ant_hierarchical/artifacts/step3_3_llm/policy_only/summary.json`
- `mujoco_ant_hierarchical/artifacts/step3_3_llm/policy_only/trajectories.png`
- `mujoco_ant_hierarchical/artifacts/step3_3_llm/always_llm/summary.json`
- `mujoco_ant_hierarchical/artifacts/step3_3_llm/always_llm/trajectories.png`
- `mujoco_ant_hierarchical/artifacts/step3_3_llm/always_llm/llm_images/`
- `mujoco_ant_hierarchical/artifacts/step3_3_llm/uncertainty_switching/summary.json`
- `mujoco_ant_hierarchical/artifacts/step3_3_llm/uncertainty_switching/trajectories.png`
- `mujoco_ant_hierarchical/artifacts/step3_3_llm/uncertainty_switching/llm_images/`

## Interpretation

The real-image LLM path works end-to-end:

- MuJoCo EGL rendered overhead and egocentric images at LLM call steps.
- The model returned parseable JSON waypoints.
- No waypoint postprocess was applied before handing waypoints to the hierarchical controller.

The comparison result is negative for the current prompt/controller interface. `always_llm`
had one success out of 20 episodes, but neither LLM condition beat `policy_only` on mean
final distance, and `uncertainty_switching` did not improve success rate.

Observed failure pattern:

- The LLM usually suggested plausible obstacle-avoidance waypoints near
  `x=2.5..3.2, y=+/-1.45..1.6`.
- The Ant often reached the side of the obstacle but then orbited or stalled near the
  intermediate waypoint.
- In the single successful `always_llm` episode, the model first routed above the
  obstacle and later returned the final goal once the direct path was clear.
- The current uncertainty trigger fires early and repeatedly in obstacle/stall regions,
  so it reduced calls only slightly versus `always_llm` (`50` vs `54` calls).

## Stop Point

Per the requested workflow, stop after Step 3-4 results. The current postprocess-free
LLM setup does not yet produce the desired improvement over `policy_only`.
