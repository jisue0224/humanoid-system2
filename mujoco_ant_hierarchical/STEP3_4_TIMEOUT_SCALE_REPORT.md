# Step 3-4 Waypoint Timeout / Scaling Sweep

Date: 2026-06-20

## Goal

Test whether the remaining bottleneck is low-level waypoint follow-through by
changing the same waypoint controller parameters for all LLM conditions:

```text
waypoint_timeout_steps
waypoint_min_action_scale
```

The uncertainty signal stays fixed:

```text
depth switching: center_20x30_mean, threshold 0.25
progress switching: progress threshold 0.0247
```

## Step 1: Quick Sweep on Depth Switching

10 episodes each, depth switching only:

| timeout | min_action_scale | success_rate | mean_final_distance | LLM calls | timeout count | gate count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `80` | `0.85` | `0.30` | `1.932 m` | `27` | `13` | `9` |
| `100` | `0.85` | `0.00` | `2.833 m` | `21` | `7` | `6` |
| `50` | `0.90` | `0.20` | `2.110 m` | `34` | `16` | `13` |
| `50` | `0.95` | `0.40` | `1.965 m` | `36` | `18` | `18` |

Chosen setting for the full run:

```text
waypoint_timeout_steps = 50
waypoint_min_action_scale = 0.95
```

Reason: highest quick-sweep success rate, with the most gate completions.

## Step 2: Full Comparison With the Chosen Setting

20 episodes each.

| condition | success_rate | mean_final_distance | LLM calls | timeout count |
| --- | ---: | ---: | ---: | ---: |
| policy_only | `0.05` | `3.432 m` | `0` | `0` |
| always_llm | `0.10` | `3.139 m` | `78` | `46` |
| progress_switching | `0.20` | `2.590 m` | `32` | `22` |
| depth_switching | `0.10` | `3.230 m` | `57` | `39` |

## Interpretation

- The quick sweep looked promising for `timeout=50, scale=0.95`, but the full
  20-episode run did not preserve that advantage.
- `progress_switching` was the best condition under this controller setting.
- `depth_switching` lost its earlier edge and remained timeout-heavy.
- Increasing waypoint timeout did reduce some premature exits, but it did not
  solve the main issue: the controller still does not reliably carry waypoints
  through to completion.

## Artifacts

- `mujoco_ant_hierarchical/artifacts/step3_4_depth_llm_timeout_sweep/quick_sweep_summary.json`
- `mujoco_ant_hierarchical/artifacts/step3_4_timeout_scale95_final/policy_only/policy_only/summary.json`
- `mujoco_ant_hierarchical/artifacts/step3_4_timeout_scale95_final/always_llm/always_llm/summary.json`
- `mujoco_ant_hierarchical/artifacts/step3_4_timeout_scale95_final/progress_switching/uncertainty_switching/summary.json`
- `mujoco_ant_hierarchical/artifacts/step3_4_timeout_scale95_final/depth_switching/uncertainty_switching/summary.json`

