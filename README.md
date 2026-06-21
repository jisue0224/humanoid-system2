# Uncertainty-Gated Hierarchical Control
### When Should a Robot Ask an LLM for Help?

**CAS4133 Final Project** — Proposing an alternative to fixed-interval
and termination-based switching in Hierarchical RL: an
uncertainty-gated trigger for invoking an LLM as a high-level policy.

📄 Full report: see [`uncertainty_gated_hrl_report.tex`](./uncertainty_gated_hrl_report.tex) (Overleaf-ready)

---

## TL;DR

A frozen, pretrained low-level locomotion policy handles navigation
autonomously. An LLM (high-level policy) is invoked **only** when an
uncertainty signal exceeds a threshold — not at every timestep, and
not on a fixed schedule. We compare two uncertainty signals
(progress-based vs. egocentric depth-based) and show that the
vision-based signal detects obstacles earlier and more accurately,
leading to a significantly higher task success rate while keeping LLM
calls selective.

```
SAC only (no LLM)          →  74% success                (PointMaze)
Oracle waypoint switching  →  88% success   (upper bound)
Uncertainty switching      →  88% success,  LLM called in only 14% of episodes

Ant + progress-based uncertainty  →  5.0% success
Ant + depth-based uncertainty     → 22.5% success   (z ≈ 2.35, statistically significant)
```

---

## Motivation

LLMs/VLMs can now reason about images and spatial layout, but calling
one at every control step is too slow and too expensive for
real-time robot control. This project asks: **can a cheap, fast
low-level policy be paired with an LLM that only gets called when it's
actually needed?**

This is framed as a Hierarchical RL (HRL) problem. Classic HRL
switches between high-level and low-level control using either:

1. a **fixed $H$-step interval** (e.g., HIRO), or
2. a **learned termination signal** (e.g., Option-Critic).

We propose a third option: an **uncertainty signal**, computed
directly from the low-level policy's progress or visual input, as the
switching trigger — inspired by the neuroscience of human dual-process
decision-making (Daw et al., 2005) and deployed systems like Waymo's
Fleet Response, which escalates to a human operator only in ambiguous
situations.

---

## Project Structure

This repo contains two phases of experiments.

```
.
├── mujoco_ant/                    # Phase 2 initial attempts
│   ├── STEP1_REPORT.md            # pretrained Ant policy smoke test
│   └── STEP2_GOAL_ANT_REPORT.md   # goal-conditioned from-scratch (failed)
│
├── mujoco_ant_hierarchical/       # Phase 2 final architecture (main results)
│   ├── STEP1_STEP2_REPORT.md      # frozen locomotion + goal routing
│   ├── STEP2_REFINEMENT_REPORT.md
│   ├── STEP3_1_VISION_REPORT.md   # EGL vision rendering
│   ├── STEP3_2_UNCERTAINTY_REPORT.md
│   ├── STEP3_3_*.md               # waypoint completion / scaling sweeps
│   ├── STEP3_4_DEPTH_OCCUPANCY_SWEEP_REPORT.md
│   ├── STEP3_4_VISION_UNCERTAINTY_REPORT.md   # signal comparison (key result)
│   ├── STEP3_4_FINAL_40EPS_REPORT.md          # final comparison (key result)
│   └── artifacts/                 # plots, trajectories, demo videos
│
├── mujoco_sim2sim/                # Isaac Lab → MuJoCo porting attempt (abandoned)
├── mujoco_playground/             # alternative vision-ready env investigation
└── src/system2_nav/               # shared utilities
```

> **Note on Phase 1 (PointMaze):** the initial proof-of-concept was
> developed in a separate GPU workspace that was reset mid-session
> before being pushed to version control, so its source code is not
> included in this repo. Phase 1 results (tables, figures, and
> analysis) are preserved in the final report.

---

## Phase 1 — PointMaze (Proof of Concept)

- **Environment:** `PointMaze-UMazeDense-v3` (MuJoCo)
- **Low-level policy:** goal-conditioned SAC
- **Uncertainty signal:** progress-based (5-step goal-distance change)
- **High-level policy:** GPT-5.4, given a top-down grid image, returns
  an intermediate waypoint

| Condition | Success Rate | LLM Call Rate |
|---|---|---|
| SAC only | 0.74 | 0% |
| Oracle switching (upper bound) | 0.88 | — |
| **Uncertainty switching** | **0.88** | **14%** (7/50 episodes) |

Uncertainty switching matched oracle-level performance while calling
the LLM in only 7 out of 50 episodes.

---

## Phase 2 — MuJoCo Ant (Hierarchical Architecture + Vision)

### Why we extended beyond PointMaze

Phase 1 used coordinate-based input only. To test whether a
**vision**-based uncertainty signal could outperform a non-visual one,
we needed real egocentric/overhead camera rendering — which required
moving to a locomotion-capable environment.

### Exploration path (what didn't work, and why)

| Attempt | Result |
|---|---|
| Isaac Lab H1 (velocity-command locomotion) | ✅ physics validated, ❌ vision blocked (Vulkan graphics capability unavailable on the VESSL container) |
| Isaac Lab → MuJoCo sim2sim porting | ❌ dynamics mismatch (PhysX vs. MuJoCo contact/actuator), policy collapses within ~1.5s |
| Ant goal-conditioned, trained from scratch | ❌ 500k steps, success rate = 0 (jointly learning locomotion + navigation is too sample-inefficient) |
| **Frozen Ant-v4 locomotion + observation-frame rotation** | ✅ **adopted** — zero additional training, vision works via standard EGL rendering |

### Final architecture

The pretrained Ant-v4 locomotion policy (forward-only, no notion of
"goal") is used **frozen**. Its input observation (root quaternion +
root xy velocity) is rotated at every step so that the target is
always perceived as directly ahead:

```
Δ = target_xy - current_xy
θ_target = atan2(Δ_y, Δ_x)
obs_quat' = yaw_quat(-θ_target) ⊗ obs_quat
obs_vel'  = R(-θ_target) · obs_vel
```

This requires **zero training steps** and makes the policy
goal-directed without modifying it.

### Uncertainty signal design (core contribution)

Three candidate signals were measured against ground-truth obstacle
distance (20 episodes, measurement-only, no LLM calls):

| Signal | Corr(distance) | False Positive | Generalizable? |
|---|---|---|---|
| Progress-based (baseline) | +0.047 (≈ none) | 0.43 | ✅ but uninformative — actually detects locomotion wobble, not obstacles |
| Red occlusion (color-based) | +0.116 | 0.45 | ❌ depends on obstacle color, discarded |
| **Depth occupancy** | **−0.326** | **0.29** | ✅ color-independent, physically grounded |

**Final signal:** mean depth over the central 20%×30% region of the
egocentric depth image, threshold 0.25.

### Final result (40 episodes per condition)

| Condition | Success Rate | 95% CI | LLM Calls | Timeouts |
|---|---|---|---|---|
| Policy only | 0.025 | [0.000, 0.073] | 0 | 0 |
| Progress switching | 0.050 | [0.000, 0.118] | 62 | 41 |
| **Depth switching** | **0.225** | **[0.096, 0.354]** | 119 | 69 |

$z \approx 2.35$ — the success-rate advantage of depth switching is
unlikely to be due to chance.

**Same-seed comparison (seed 6001):** progress switching's first LLM
call occurs at step 109 (after extensive wandering near the obstacle,
→ fails); depth switching's first call occurs at step 17 (as soon as
the obstacle enters view, → succeeds). This is the clearest single
piece of evidence that vision-based uncertainty is *proactive* rather
than *reactive*.

---

## Trade-offs & Limitations

- **Accuracy vs. cost:** depth switching is more accurate but also
  more sensitive — it triggers more LLM calls and more timeouts than
  progress switching. This trade-off is tunable via the uncertainty
  threshold.
- **Low-level follow-through:** the largest remaining bottleneck is
  that the low-level controller does not always reliably reach an
  LLM-proposed waypoint (gate completion 28/40, timeout 69/40
  episodes). We attribute this to the controllability limits of
  steering a frozen policy purely through observation rotation, not
  to the quality of the uncertainty signal.
- **Environment simplicity:** validated on a single rectangular
  obstacle; maze-like, multi-obstacle environments are untested.

---

## Related Work

- **HRL switching:** HIRO (fixed $H$-step), Option-Critic (learned
  termination function) — this project proposes uncertainty gating as
  a third alternative.
- **Uncertainty + LLM + navigation:** REAL (resilience/adaptation,
  closest in philosophy), SCOPE (environment uncertainty), TrustNavGPT
  (LLM's own uncertainty) — this project's uncertainty is the
  low-level policy's progress/visual uncertainty.
- **HRM (2025):** same System 1/2 philosophy implemented as two
  recurrent modules within one network; this project instead combines
  two already-pretrained modules without retraining either.
- **VLA models (SayCan, RT-2):** call an LLM/VLM at nearly every
  step; this project gates calls on uncertainty for efficiency.

---

## Citation / Background

- Daw, N. D., Niv, Y., & Dayan, P. (2005). Uncertainty-based
  competition between prefrontal and dorsolateral striatal systems
  for behavioral control. *Nature Neuroscience*, 8(12), 1704–1711.
- Nachum, O. et al. (2018). Data-efficient hierarchical reinforcement
  learning. *NeurIPS*.
- Bacon, P. L. et al. (2017). The option-critic architecture. *AAAI*.
- Tagliabue, A. et al. (2023). REAL: Resilience and Adaptation using
  LLMs on Autonomous Aerial Robots. *arXiv:2311.01403*.
