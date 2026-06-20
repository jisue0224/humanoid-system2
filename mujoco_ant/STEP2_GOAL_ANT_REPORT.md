# MuJoCo Ant Goal System 1 Smoke Test

Date: 2026-06-20

## Goal

Check whether the native `Ant-v4` route can quickly produce a goal/waypoint-following System 1 policy. This is the required gate before obstacle uncertainty and vision-based LLM waypoint experiments, because Step 4 needs a low-level controller that can actually move toward an LLM-proposed waypoint.

## Implementation

Added a small goal-conditioned wrapper around `Ant-v4`:

- Base env: `Ant-v4`
- Action: unchanged 8D Ant action
- Observation: original Ant observation plus 5 goal features:
  - clipped `dx / 10`
  - clipped `dy / 10`
  - clipped `distance / 10`
  - `sin(bearing)`
  - `cos(bearing)`
- Reward:
  - `10.0 * progress_to_goal`
  - `+0.5` healthy reward
  - `-0.02 * sum(action^2)`
  - `+50.0` success bonus
- Success radius: `0.6 m`
- Max episode length: `300`

Code:

- `mujoco_ant/ant_goal_env.py`
- `mujoco_ant/step2_goal_ant_training_smoke.py`

## Training Run

Command:

```bash
source env_isaaclab/bin/activate
python mujoco_ant/step2_goal_ant_training_smoke.py --total_timesteps 25000 --eval_episodes 10
```

SAC config:

```python
model = SAC(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    buffer_size=200_000,
    batch_size=256,
    learning_starts=2_000,
    train_freq=1,
    gradient_steps=1,
    gamma=0.98,
    policy_kwargs={"net_arch": [256, 256]},
)
```

## Result

| Metric | Value |
| --- | ---: |
| total timesteps | `25,000` |
| train time | `276.49 s` |
| effective speed | `90.42 steps/s` |
| eval episodes | `10` |
| eval success rate | `0.0` |
| mean final distance | `4.03 m` |
| median final distance | `4.55 m` |

Artifacts:

- `mujoco_ant/artifacts/step2_goal_ant/goal_ant_training_summary.json`
- `mujoco_ant/artifacts/step2_goal_ant/goal_ant_eval_trajectories.png`
- `mujoco_ant/artifacts/step2_goal_ant/sac_ant_goal_smoke.zip`

## Interpretation

This failed the Step 1/2 gate for the Ant route. The policy shows some motion and reward improvement during training, but after 25k steps it still cannot reliably follow arbitrary sampled goals.

This matters because the requested Step 4 experiment depends on:

1. LLM proposes a waypoint.
2. System 1 moves to that waypoint.
3. The final behavior improves over `policy_only`.

With the current Ant System 1, step 2 is not solved. Running LLM comparisons now would mainly measure a broken waypoint follower, not uncertainty-based System 1/System 2 switching.

## Decision

Do not proceed to obstacle uncertainty or LLM waypoint experiments on this Ant policy yet.

Reasonable next options:

1. Train the goal-conditioned Ant policy much longer and monitor success rate checkpoints.
2. Use an environment with an already goal/velocity-commanded locomotion policy.
3. Return to the stable Ant-v4 forward policy only for vision/rendering demos, but not for waypoint navigation claims.
