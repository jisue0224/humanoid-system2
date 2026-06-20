# MuJoCo Ant Step 1 - Pretrained Policy Smoke Test

Date: 2026-06-20

## Goal

Verify that a native MuJoCo Ant pretrained policy can be loaded and rolled out on a flat `Ant-v4` environment before rebuilding System 1/System 2 switching on top of Ant.

## Environment

- Gymnasium: `1.2.3`
- Stable-Baselines3: `2.8.0`
- Environment: `Ant-v4`
- Policy source: HuggingFace `jren123/sac-ant-v4`
- Policy file: `SAC-Ant-v4.zip`
- Rollout length: 100 steps
- MuJoCo timestep exposed by env: `dt = 0.05`
- Simulated duration: 5.0 seconds

## Result

The pretrained SAC policy loaded and completed the full 100-step rollout without termination or truncation.

| Metric | Value |
| --- | ---: |
| requested steps | 100 |
| actual steps | 100 |
| terminated | false |
| truncated | false |
| start xy | `(0.0274, -0.0460)` |
| end xy | `(22.3545, 1.8779)` |
| delta xy | `(22.3271, 1.9239)` |
| xy displacement | `22.4098 m` |
| path length | `22.8189 m` |
| mean xy velocity | `(4.4654, 0.3848) m/s` |
| torso z min/max/final | `0.4053 / 0.7930 / 0.7916 m` |
| reward sum | `455.3462` |
| stable_100_step_rollout | true |

Artifacts:

- `mujoco_ant/artifacts/step1/step1_pretrained_ant_summary.json`
- `mujoco_ant/artifacts/step1/step1_pretrained_ant_trajectory.png`

## Interpretation

This satisfies the main Step 1 gate: the pretrained native MuJoCo Ant locomotion policy is stable on flat ground for 100 steps.

Important limitation: this policy is not goal-conditioned. It is a standard forward-reward SAC policy for `Ant-v4`, so its direct output is an 8D joint action for the current Ant observation. It does not accept a goal coordinate, waypoint, desired heading, or velocity command. Goal navigation in Step 2 will therefore need a high-level controller/wrapper around the policy, or a separate goal-conditioned training setup.

## Code Path

Smoke test script:

```bash
python mujoco_ant/step1_pretrained_ant_smoke.py
```

Core loading path:

```python
from huggingface_sb3 import load_from_hub
from stable_baselines3 import SAC

checkpoint_path = load_from_hub(
    repo_id="jren123/sac-ant-v4",
    filename="SAC-Ant-v4.zip",
)
model = SAC.load(checkpoint_path)
```

Core rollout path:

```python
env = gym.make("Ant-v4")
obs, _ = env.reset(seed=0)

for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break
```

## Decision

Proceeding to Step 2 is reasonable, but Step 2 should explicitly treat the current pretrained Ant as a locomotion primitive rather than a goal-aware controller.
