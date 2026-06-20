# Step 3-4 Depth Occupancy LLM Experiment

Date: 2026-06-20

## Setup

Final depth occupancy signal:

```text
ROI: center_20x30_mean
threshold: 0.25
smoothing: none
depth_threshold: 3.0
```

LLM experiment configuration:

```text
condition: uncertainty_switching
waypoint_min_action_scale: 0.85
waypoint_slow_radius: 7.0
waypoint_completion: gate
postprocess: disabled
LLM model: gpt-5.4
images: overhead + egocentric
```

Depth trigger:

```text
trigger when depth occupancy >= 0.25
```

Compared against the previous gate-passing progress-based run:

```text
trigger when progress-based uncertainty >= 0.0247
```

## Results

| condition | success_rate | mean_final_distance | LLM calls |
| --- | ---: | ---: | ---: |
| policy_only | `0.05` | `3.432 m` | `0` |
| always_llm | `0.15` | `2.794 m` | `94` |
| progress_switching | `0.15` | `2.626 m` | `62` |
| depth_switching | `0.30` | `2.378 m` | `66` |

## Timing Comparison

The depth trigger fires earlier than the progress trigger:

| condition | mean first LLM call step | median first LLM call step | calls before step 15 |
| --- | ---: | ---: | ---: |
| progress_switching | `26.65` | `27.0` | `0` |
| depth_switching | `18.35` | `16.0` | `0` |

This means depth-based uncertainty is reacting closer to the obstacle, but it still
does not produce a large call reduction versus the progress baseline.

## Waypoint Follow-through

Depth switching still hits the waypoint timeout frequently:

| condition | waypoint timeouts | gate completions |
| --- | ---: | ---: |
| progress_switching | `33` | `17` |
| depth_switching | `35` | `22` |

Interpretation:

- Depth occupancy improves the trigger quality and raises success rate from `0.15`
  to `0.30`.
- The remaining bottleneck is not just uncertainty detection. The controller still
  spends many episodes timing out on intermediate waypoints.
- The call count stays in the same range as progress switching because the new
  signal is earlier, not substantially rarer.

## Artifacts

- `mujoco_ant_hierarchical/artifacts/step3_4_depth_llm/uncertainty_switching/summary.json`
- `mujoco_ant_hierarchical/artifacts/step3_4_depth_llm/uncertainty_switching/trajectories.png`
- `mujoco_ant_hierarchical/artifacts/step3_4_depth_llm/policy_only/summary.json`
- `mujoco_ant_hierarchical/artifacts/step3_4_depth_llm/plots/progress_vs_depth_calls.png`

