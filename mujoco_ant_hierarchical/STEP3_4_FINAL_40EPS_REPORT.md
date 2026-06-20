# Step 3-4 Final 40-Episode Comparison

Date: 2026-06-20

## Setup

Controller settings fixed across the comparison:

```text
waypoint_timeout_steps = 50
waypoint_min_action_scale = 0.85
waypoint_completion = gate
depth signal = center_20x30_mean, threshold 0.25, no smoothing
progress signal = threshold 0.0247
```

All runs used 40 episodes.

## Results

| condition | success_rate | 95% CI | mean_final_distance | LLM calls | timeout count |
| --- | ---: | ---: | ---: | ---: | ---: |
| policy_only | `1 / 40 = 0.025` | `[0.000, 0.073]` | `2.891 m` | `0` | `0` |
| progress_switching | `2 / 40 = 0.050` | `[0.000, 0.118]` | `2.747 m` | `62` | `41` |
| depth_switching | `9 / 40 = 0.225` | `[0.096, 0.354]` | `2.958 m` | `119` | `69` |

## Statistical Check

Using the standard binomial standard error approximation:

```text
SE(p) = sqrt(p * (1 - p) / n)
```

For success rate:

- progress_switching: `p = 0.05`, `SE = 0.0345`
- depth_switching: `p = 0.225`, `SE = 0.0660`

Difference:

```text
depth - progress = 0.175
SE(diff) ≈ 0.0745
z ≈ 2.35
```

Interpretation:

- Depth is better than progress on success rate in this 40-episode sample.
- The result is not overwhelming, but it is strong enough to treat as a real
  effect rather than noise.
- The 95% CIs barely overlap, which is consistent with a meaningful gap.
- The rough two-sample z score is `2.35`, which is enough to call the depth
  gain on success rate likely real in this sample.

## Practical Readout

- `depth_switching` is the best on success rate.
- `progress_switching` is cheaper and gives slightly lower final distance, but it
  succeeds much less often.
- `depth_switching` also incurs more LLM calls and more waypoint timeouts, so it
  is not uniformly better across all metrics.

## Final Takeaway

If the objective is navigation success, keep the depth-based trigger.
If the objective is cost or controller efficiency, progress remains lighter but
weaker.

For visualization, the representative success case is `seed 6001`. In that
case, `depth_switching` fired on step `17` and reached the goal, while
`progress_switching` on the same seed did not succeed and did not call until
step `109`.

## Artifacts

- `mujoco_ant_hierarchical/artifacts/step3_4_final_40eps_comparison.png`
- `mujoco_ant_hierarchical/artifacts/step3_4_final_40eps_repcompare/progress_vs_depth_seed6001.png`
- `mujoco_ant_hierarchical/artifacts/step3_4_final_40eps_representative_success/representative_success_depth_episode.png`
- `mujoco_ant_hierarchical/artifacts/step3_4_final_40eps_representative_success/ego_rgb_step17.png`
- `mujoco_ant_hierarchical/artifacts/step3_4_final_40eps_representative_success/ego_depth_step17.png`
- `mujoco_ant_hierarchical/artifacts/step3_4_final_40eps_representative/representative_depth_episode.png`
- `mujoco_ant_hierarchical/artifacts/step3_4_final_40eps_representative/ego_rgb_step16.png`
- `mujoco_ant_hierarchical/artifacts/step3_4_final_40eps_representative/ego_depth_step16.png`
- `mujoco_ant_hierarchical/artifacts/step3_4_final_40eps_progress/uncertainty_switching/summary.json`
- `mujoco_ant_hierarchical/artifacts/step3_4_final_40eps_progress/uncertainty_switching/trajectories.png`
- `mujoco_ant_hierarchical/artifacts/step3_4_final_40eps_depth/uncertainty_switching/summary.json`
- `mujoco_ant_hierarchical/artifacts/step3_4_final_40eps_depth/uncertainty_switching/trajectories.png`
- `mujoco_ant_hierarchical/artifacts/step3_4_final_40eps_policy/policy_only/summary.json`
