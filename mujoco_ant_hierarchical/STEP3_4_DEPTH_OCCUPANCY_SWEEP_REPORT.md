# Step 3-4 Depth Occupancy Sweep

Date: 2026-06-20

## Goal

This experiment focuses only on egocentric depth occupancy. Red RGB occlusion
and progress uncertainty are excluded from the main signal.

The goal is to find a depth-only signal that triggers before obstacle contact
without excessive false positives.

## Setup

Same policy-only obstacle rollouts as the previous vision uncertainty check:

```bash
source env_isaaclab/bin/activate
python mujoco_ant_hierarchical/step3_4_depth_occupancy_sweep.py \
  --episodes 20 \
  --max_steps 220 \
  --width 160 \
  --height 120 \
  --output_dir mujoco_ant_hierarchical/artifacts/step3_4_depth_occupancy_sweep
```

Depth occupancy formula:

```python
uncertainty = max(0, (depth_threshold - roi_depth_value) / depth_threshold)
```

with `depth_threshold=3.0`.

Reference timing event:

```text
distance_to_obstacle_box <= 1.0 m
```

Offset:

```text
offset = trigger_step - obstacle_near_step
```

Negative offset means earlier detection.

## Experiment 1: Threshold Sweep

Thresholds:

```text
0.05, 0.08, 0.10, 0.15, 0.20, 0.25
```

For the original ROI, `center_40x50_mean`, the tradeoff is:

| Threshold | Median offset | Mean offset | Early/on-time episodes | False positive rate |
| ---: | ---: | ---: | ---: | ---: |
| `0.05` | `+4.0` | `+6.75` | `3 / 20` | `0.201` |
| `0.08` | `+4.0` | `+6.75` | `3 / 20` | `0.199` |
| `0.10` | `+4.5` | `+7.55` | `3 / 20` | `0.197` |
| `0.15` | `+4.5` | `+8.45` | `2 / 20` | `0.188` |
| `0.20` | `+4.5` | `+8.55` | `2 / 20` | `0.157` |
| `0.25` | `+4.5` | `+8.75` | `2 / 20` | `0.149` |

Conclusion: threshold tuning alone does not fix the late-trigger issue for the
original ROI. Increasing threshold reduces false positives but remains late.

## Experiment 2: ROI / Statistic Sweep

Best threshold per ROI is selected by practical tradeoff, not just one metric.

| ROI / statistic | Corr(distance, signal) | Best threshold | Median offset | Mean offset | Early/on-time episodes | False positive rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| center 40%x50% mean | `-0.289` | `0.25` | `+4.5` | `+8.75` | `2 / 20` | `0.149` |
| center 20%x30% mean | `-0.326` | `0.25` | `-1.0` | `-3.60` | `13 / 20` | `0.293` |
| lower half mean | `-0.356` | `0.25` | `-14.0` | `-13.65` | `19 / 20` | `0.423` |
| center 40%x50% min | `-0.452` | `0.25` | `-17.5` | `-19.75` | `20 / 20` | `0.923` |

Interpretation:

- `center_40x50_mean` is stable but late.
- `center_20x30_mean` is the best balanced option: slightly early, moderate
  false positives, and better distance correlation than the original ROI.
- `lower_half_mean` is early but too noisy/overactive.
- `min` is too sensitive; it fires almost all the time, with unusable false
  positive rate.

## Best Current Candidate

```text
ROI: center_20x30_mean
threshold: 0.25
depth_threshold: 3.0
median_offset: -1.0 step
mean_offset: -3.60 steps
early/on-time: 13 / 20 episodes
false_positive_rate: 0.293
corr(distance, signal): -0.326
```

This is not perfect, but it is better aligned with the objective than the
previous setting:

- Previous original ROI at threshold `0.15`:
  - median offset `+4.5`
  - false positive `0.188`
- New best candidate:
  - median offset `-1.0`
  - false positive `0.293`

So the new setting trades a moderate false-positive increase for earlier
obstacle detection.

## Experiment 3: Temporal Smoothing

We ran 3-frame smoothing on the current best ROI/threshold candidate:

- ROI: `center_20x30_mean`
- threshold: `0.25`
- smoothing window: `3` frames

| Smoothing | Median offset | Mean offset | False positive rate | Corr(distance, signal) |
| --- | ---: | ---: | ---: | ---: |
| none | `-1.0` | `-3.60` | `0.293` | `-0.326` |
| 3-frame mean | `+1.5` | `+1.20` | `0.276` | `-0.282` |
| 3-frame median | `+2.5` | `+3.35` | `0.290` | `-0.324` |

Interpretation:

- 3-frame mean lowers the false-positive rate a little, but it flips the
  trigger from slightly early to late.
- 3-frame median preserves correlation better than mean, but it is still late
  and does not improve false positives enough.
- Neither smoothed variant beats the unsmoothed baseline on the combined
  objective of early detection plus acceptable false-positive rate.

If we relax the timing requirement and only optimize false positives, the best
smoothed settings are:

- `3-frame mean` at threshold `0.40`: median offset `+4.0`, false positive
  rate `0.241`
- `3-frame median` at threshold `0.45`: median offset `+4.0`, false positive
  rate `0.249`

These satisfy the false-positive target, but the trigger is too late for the
current use case.

## Final Selected Signal

The final signal choice remains the unsmoothed setting:

```text
ROI: center_20x30_mean
threshold: 0.25
smoothing: none
depth_threshold: 3.0
median_offset: -1.0 step
mean_offset: -3.60 steps
false_positive_rate: 0.293
corr(distance, signal): -0.326
```

This is the best practical tradeoff we have so far: it is early enough to be
useful, and smoothing does not improve the overall behavior.

## Artifacts

- `mujoco_ant_hierarchical/artifacts/step3_4_depth_occupancy_sweep/summary.json`
- `mujoco_ant_hierarchical/artifacts/step3_4_depth_occupancy_sweep/plots/threshold_roi_tradeoff.png`
- `mujoco_ant_hierarchical/artifacts/step3_4_depth_occupancy_sweep/plots/best_depth_signal_example.png`
