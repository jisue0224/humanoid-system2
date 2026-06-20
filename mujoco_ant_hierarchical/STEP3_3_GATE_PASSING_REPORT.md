# Step 3-3 Gate-Passing Waypoint Completion

Date: 2026-06-20

## Change

Added an optional waypoint completion mode:

```bash
--waypoint_completion gate
--gate_x_margin 0.05
--gate_y_abs 0.5
```

For gate mode, an LLM waypoint is treated as an obstacle-side gate rather than a
point to stop at:

- If the waypoint is above the obstacle, the route side is `+y`.
- If the waypoint is below the obstacle, the route side is `-y`.
- The waypoint completes when the Ant reaches `x >= obstacle_x_max + 0.05`
  while staying on the selected side with `abs(y) >= 0.5`.
- If the Ant enters the obstacle x-corridor and violates the selected side, the
  gate is marked invalid and can only clear by timeout.
- Timeout remains `50` steps.

The action scaling setting is unchanged from the best scale sweep:

```text
waypoint_slow_radius = 7.0
waypoint_min_action_scale = 0.85
```

## Commands

```bash
source env_isaaclab/bin/activate

python mujoco_ant_hierarchical/step3_3_llm_waypoint_experiment.py \
  --condition uncertainty_switching \
  --episodes 20 \
  --include_ego \
  --waypoint_completion gate \
  --gate_x_margin 0.05 \
  --gate_y_abs 0.5 \
  --waypoint_slow_radius 7.0 \
  --waypoint_min_action_scale 0.85 \
  --output_dir mujoco_ant_hierarchical/artifacts/step3_3_gate_passing

python mujoco_ant_hierarchical/step3_3_llm_waypoint_experiment.py \
  --condition always_llm \
  --episodes 20 \
  --include_ego \
  --waypoint_completion gate \
  --gate_x_margin 0.05 \
  --gate_y_abs 0.5 \
  --waypoint_slow_radius 7.0 \
  --waypoint_min_action_scale 0.85 \
  --output_dir mujoco_ant_hierarchical/artifacts/step3_3_gate_passing
```

## Results

| Condition | Completion | Episodes | Success rate | Mean final distance | Mean LLM calls | Total LLM calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| always_llm | radius, scale 0.85 | `20` | `0.10` | `2.815 m` | `5.25` | `105` |
| uncertainty_switching | radius, scale 0.85 | `20` | `0.15` | `2.892 m` | `3.40` | `68` |
| always_llm | gate, scale 0.85 | `20` | `0.15` | `2.794 m` | `4.70` | `94` |
| uncertainty_switching | gate, scale 0.85 | `20` | `0.15` | `2.626 m` | `3.10` | `62` |

## Gate Event Counts

| Condition | Gate completions | Timeouts |
| --- | ---: | ---: |
| always_llm | `23` | `55` |
| uncertainty_switching | `17` | `33` |

## Interpretation

Gate-passing did not improve `uncertainty_switching` success rate above the
previous `0.15`, so it is not a clear success-rate breakthrough.

It did improve secondary metrics:

- `uncertainty_switching` mean final distance improved from `2.892 m` to
  `2.626 m`.
- `uncertainty_switching` total LLM calls decreased from `68` to `62`.
- `always_llm` success improved from `0.10` to `0.15`, with fewer calls
  (`105` to `94`).

The large timeout count means many waypoints still do not cleanly become
pass-through gates. The next issue is likely planner/controller mismatch rather
than just the completion criterion.

## Artifacts

- `mujoco_ant_hierarchical/artifacts/step3_3_gate_passing/always_llm/summary.json`
- `mujoco_ant_hierarchical/artifacts/step3_3_gate_passing/always_llm/trajectories.png`
- `mujoco_ant_hierarchical/artifacts/step3_3_gate_passing/uncertainty_switching/summary.json`
- `mujoco_ant_hierarchical/artifacts/step3_3_gate_passing/uncertainty_switching/trajectories.png`
