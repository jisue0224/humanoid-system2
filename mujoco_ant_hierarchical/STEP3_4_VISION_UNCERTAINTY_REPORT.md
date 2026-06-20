# Step 3-4 Egocentric Vision Uncertainty Measurement

Date: 2026-06-20

## Depth Rendering Check

MuJoCo EGL egocentric depth rendering works in this environment.

Smoke result:

```text
rgb:   (240, 320, 3) uint8
depth: (240, 320) float32, finite=True
depth min/max/mean: 0.869 / 459.016 / 216.745
```

Depth values have a large far-plane range, so the occupancy signal clips the
center crop to a bounded range before averaging.

## Signals

All signals were computed on the same 20 policy-only obstacle episodes.

### A. Progress-Based Baseline

```python
progress = distance[t - 5] - distance[t]
uncertainty_progress = max(0, min_progress - progress)
```

### B. Egocentric Depth Occupancy

Center crop:

- width: center 40%
- height: center 50%

```python
uncertainty_occupancy = max(0, (depth_threshold - mean_center_depth) / depth_threshold)
```

Configured `depth_threshold=3.0`.

### C. Egocentric RGB Red Occlusion

Center-ish crop:

- width: center 60%
- height: center 70%

The signal is the fraction of red-obstacle pixels in the crop.

## Experiment

Command:

```bash
source env_isaaclab/bin/activate
python mujoco_ant_hierarchical/step3_4_vision_uncertainty_measure.py \
  --episodes 20 \
  --max_steps 220 \
  --width 160 \
  --height 120 \
  --depth_threshold 3.0 \
  --obstacle_near_threshold 1.0 \
  --output_dir mujoco_ant_hierarchical/artifacts/step3_4_vision_uncertainty
```

Result:

```text
episodes: 20
success_rate: 0.0
mean_min_obstacle_distance: 0.393 m
```

## Timing Comparison

Reference event: first step where ground-truth distance to obstacle box is
`<= 1.0 m`.

Offset definition:

```text
offset = signal_trigger_step - obstacle_near_step
```

Negative offset means the signal triggered before obstacle-near ground truth.

| Signal | Threshold | Mean offset | Median offset | Early/on-time episodes | Far-step false positive rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| progress | `0.02` | `-9.40` steps | `-12.00` steps | `16 / 20` | `0.430` |
| depth occupancy | `0.15` | `+8.45` steps | `+4.50` steps | `2 / 20` | `0.188` |
| red occlusion | `0.01` | `-19.75` steps | `-17.50` steps | `20 / 20` | `0.449` |

## Obstacle Distance Correlation

Correlation is between ground-truth obstacle distance and signal value.
More negative is better for an obstacle-proximity signal.

| Signal | Corr(distance, signal) | Near mean | Far mean |
| --- | ---: | ---: | ---: |
| progress | `+0.047` | `0.049` | `0.063` |
| depth occupancy | `-0.289` | `0.283` | `0.105` |
| red occlusion | `+0.116` | `0.095` | `0.091` |

The depth occupancy signal has the most meaningful monotonic relationship with
actual obstacle distance.

## Interpretation

No single signal is immediately ideal.

Progress:

- Often triggers before the obstacle-near event.
- But it is not visually causal and has high false positives from locomotion
  wobble/stall.
- It has almost no useful correlation with true obstacle distance.

Depth occupancy:

- Best physical relationship with obstacle distance.
- Lowest false-positive rate among the three at the chosen threshold.
- But the default threshold `0.15` is too conservative for early triggering:
  it usually fires after the robot is already within 1m of the obstacle.

Red occlusion:

- Earliest trigger by timing.
- But false positives are high because red pixels can appear in the view before
  the obstacle is actually a blocking risk, and the signal is color/pose
  sensitive.

## Recommendation

Use depth occupancy as the main vision-derived uncertainty component, but tune
its threshold or combine it with red occlusion.

A reasonable next trigger candidate is:

```text
vision_uncertainty = max(occupancy, k * occlusion)
```

with:

- occupancy as the primary stable signal
- occlusion as an early warning term
- progress retained as a fallback/stall detector rather than the main trigger

Do not replace progress with raw red occlusion alone; its false-positive rate is
too high in this run.

## Artifacts

- `mujoco_ant_hierarchical/artifacts/step3_4_vision_uncertainty/summary.json`
- `mujoco_ant_hierarchical/artifacts/step3_4_vision_uncertainty/plots/episode_000_signals.png`
- `mujoco_ant_hierarchical/artifacts/step3_4_vision_uncertainty/plots/...`
- `mujoco_ant_hierarchical/artifacts/step3_4_vision_uncertainty/plots/obstacle_distance_vs_uncertainty.png`
- `mujoco_ant_hierarchical/artifacts/step3_4_vision_uncertainty/sample_images/...`
