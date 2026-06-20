# Hierarchical MuJoCo Ant Step 3-3/3-4 LLM Status

Date: 2026-06-20

## Implemented

Added the LLM waypoint experiment script:

- `mujoco_ant_hierarchical/step3_3_llm_waypoint_experiment.py`

Supported conditions:

```text
policy_only
always_llm
uncertainty_switching
```

The LLM conditions:

- capture real MuJoCo EGL-rendered `overhead` frames at the call step
- optionally capture `ego` frames with `--include_ego`
- send image(s) and numeric state to the configured model
- parse JSON waypoint:

```json
{
  "reasoning": "short",
  "waypoint_x": 0.0,
  "waypoint_y": 0.0
}
```

No waypoint postprocess is applied. If the model returns a waypoint, the hierarchical controller temporarily uses that waypoint as the target. Once the Ant reaches the waypoint radius, it resumes the final goal.

## Baseline Run Completed

Command:

```bash
source env_isaaclab/bin/activate
python mujoco_ant_hierarchical/step3_3_llm_waypoint_experiment.py --condition policy_only --episodes 20
```

Result:

| Condition | Episodes | Success rate | Mean final distance | LLM calls |
| --- | ---: | ---: | ---: | ---: |
| policy_only | `20` | `0.0` | `3.153 m` | `0` |

Artifact:

- `mujoco_ant_hierarchical/artifacts/step3_3_llm/policy_only/summary.json`
- `mujoco_ant_hierarchical/artifacts/step3_3_llm/policy_only/trajectories.png`

## Blocker

The actual LLM comparison could not be run because `OPENAI_API_KEY` is not set in this environment.

Checked state:

```text
OPENAI_API_KEY set: false
```

The `openai` package was installed into `env_isaaclab`, so once the API key is available the remaining two conditions can be run directly:

```bash
source env_isaaclab/bin/activate
python mujoco_ant_hierarchical/step3_3_llm_waypoint_experiment.py \
  --condition always_llm \
  --episodes 20 \
  --include_ego

python mujoco_ant_hierarchical/step3_3_llm_waypoint_experiment.py \
  --condition uncertainty_switching \
  --episodes 20 \
  --include_ego
```

## Stop Point

Per the requested workflow, stop before claiming Step 3-4 results. The comparison table is incomplete until the actual `always_llm` and `uncertainty_switching` runs execute with a valid OpenAI API key.
