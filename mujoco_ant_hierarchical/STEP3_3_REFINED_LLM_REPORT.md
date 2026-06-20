# Step 3-3 Refined LLM Waypoint Results

Date: 2026-06-20

## Changes

The refined run separates intermediate waypoint behavior from final-goal behavior:

- Final goal keeps distance action scaling:
  - `success_radius=0.75`
  - `slow_radius=7.0`
  - `min_action_scale=0.3`
- LLM waypoints disable distance action scaling:
  - `waypoint_slow_radius=0.0`
  - `waypoint_min_action_scale=1.0`
- Active waypoint timeout:
  - `waypoint_timeout_steps=50`
- Uncertainty warm-up:
  - `uncertainty_warmup_steps=15`
- Prompt includes recently attempted waypoints to discourage near-duplicate replans.

## Commands

```bash
source env_isaaclab/bin/activate

python mujoco_ant_hierarchical/step3_3_llm_waypoint_experiment.py \
  --condition always_llm \
  --episodes 20 \
  --include_ego \
  --output_dir mujoco_ant_hierarchical/artifacts/step3_3_llm_refined

python mujoco_ant_hierarchical/step3_3_llm_waypoint_experiment.py \
  --condition uncertainty_switching \
  --episodes 20 \
  --include_ego \
  --output_dir mujoco_ant_hierarchical/artifacts/step3_3_llm_refined
```

## Results

| Condition | Variant | Episodes | Success rate | Mean final distance | LLM calls |
| --- | --- | ---: | ---: | ---: | ---: |
| policy_only | previous baseline | `20` | `0.00` | `3.153 m` | `0` |
| always_llm | previous | `20` | `0.05` | `3.335 m` | `54` |
| uncertainty_switching | previous | `20` | `0.00` | `3.413 m` | `50` |
| always_llm | refined | `20` | `0.00` | `3.781 m` | `87` |
| uncertainty_switching | refined | `20` | `0.20` | `3.004 m` | `65` |

## Trigger Pattern

| Condition | Variant | Calls before step 15 | Min call step | Median call step | Max call step |
| --- | --- | ---: | ---: | ---: | ---: |
| always_llm | previous | `20` | `0` | `94` | `214` |
| uncertainty_switching | previous | `16` | `5` | `87` | `215` |
| always_llm | refined | `20` | `0` | `100` | `210` |
| uncertainty_switching | refined | `0` | `20` | `82` | `194` |

The warm-up removed the early uncertainty false triggers. Previous
`uncertainty_switching` had 16 calls before step 15; the refined run had none.

## Interpretation

The refinement is directionally useful for uncertainty switching but not for
always-on replanning.

Positive:

- `uncertainty_switching` improved from `0.00` to `0.20` success rate.
- Mean final distance improved from `3.413 m` to `3.004 m`.
- The warm-up removed initial balance/wobble false triggers.

Negative:

- `always_llm` got worse: success rate dropped from `0.05` to `0.00`, and calls
  increased from `54` to `87`.
- `uncertainty_switching` calls also increased from `50` to `65`, because
  waypoint timeout creates more opportunities to replan.
- Fully disabling waypoint scaling (`action_scale=1.0`) appears too aggressive:
  it reduces waypoint stall, but it can overshoot or destabilize the subgoal
  sequence.

## Artifacts

- `mujoco_ant_hierarchical/artifacts/step3_3_llm_refined/always_llm/summary.json`
- `mujoco_ant_hierarchical/artifacts/step3_3_llm_refined/always_llm/trajectories.png`
- `mujoco_ant_hierarchical/artifacts/step3_3_llm_refined/always_llm/llm_images/`
- `mujoco_ant_hierarchical/artifacts/step3_3_llm_refined/uncertainty_switching/summary.json`
- `mujoco_ant_hierarchical/artifacts/step3_3_llm_refined/uncertainty_switching/trajectories.png`
- `mujoco_ant_hierarchical/artifacts/step3_3_llm_refined/uncertainty_switching/llm_images/`

## Next Hypothesis

The next likely fix is not `action_scale=1.0` for waypoints. A better setting is
probably a separate waypoint scaling floor around `0.7..0.9`, plus a pass-through
gate rule that switches to the next waypoint/final goal when the Ant crosses the
obstacle-side corridor, not only when it enters a circular waypoint radius.
