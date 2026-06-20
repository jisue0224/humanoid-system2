# MuJoCo Playground Investigation

Date: 2026-06-20

## Summary

MuJoCo Playground can be installed in this container only after adding Python 3.11. The project imports successfully, JAX sees the RTX 3090 as a GPU device, and the `Go1JoystickFlatTerrain` locomotion environment loads and steps with the JAX MJX implementation.

The current blocker is the vision/batch-renderer path. The current MuJoCo Playground README describes vision support through the MJWarp Batch Renderer. In this environment, any `impl="warp"` environment load fails before rollout/rendering with:

```text
AttributeError: type object 'int' has no attribute 'WARP'
```

That means the Playground vision path is not usable yet in this container without resolving the MuJoCo MJX/Warp compatibility issue.

## Step 1 - Install And Capability Check

Installed into an isolated Python 3.11 venv:

```bash
python3.11 -m venv mujoco_playground/.venv
source mujoco_playground/.venv/bin/activate
python -m pip install -e mujoco_playground/external/mujoco_playground
python -m pip install -U "jax[cuda12]" --index-url https://pypi.org/simple
```

Important details:

| Check | Result |
| --- | --- |
| Python 3.10 install | failed, package requires `>=3.11` |
| Python 3.11 install | success |
| JAX backend | `gpu` |
| JAX device | `cuda:0` |
| MuJoCo | `3.9.0` |
| MuJoCo Playground import | success |
| `Go1JoystickFlatTerrain`, `impl="jax"` | success |
| `Go1JoystickFlatTerrain`, `impl="warp"` | failed |

Go1 environment:

| Field | Value |
| --- | ---: |
| `dt` | `0.02` |
| action size | `12` |
| actor obs `state` | `48` |
| critic obs `privileged_state` | `123` |
| command shape | `3`: `(vx, vy, yaw_rate)` |

Locomotion environments found:

```text
ApolloJoystickFlatTerrain
BarkourJoystick
BerkeleyHumanoidJoystickFlatTerrain
BerkeleyHumanoidJoystickRoughTerrain
G1JoystickFlatTerrain
G1JoystickRoughTerrain
Go1JoystickFlatTerrain
Go1JoystickRoughTerrain
Go1Getup
Go1Handstand
Go1Footstand
H1InplaceGaitTracking
H1JoystickGaitTracking
Op3Joystick
SpotFlatTerrainJoystick
SpotGetup
SpotJoystickGaitTracking
T1JoystickFlatTerrain
T1JoystickRoughTerrain
```

Artifacts:

- `mujoco_playground/artifacts/step1_playground_capability_summary.json`

## Step 2 - Pretrained Policy Check

No ready-to-download official pretrained locomotion checkpoint catalog was found in the cloned MuJoCo Playground repo. The repo provides training scripts and checkpoint restore flags, but not a bundled Go1/G1/Berkeley Humanoid pretrained checkpoint path.

The best candidate environment remains `Go1JoystickFlatTerrain` because it is quadruped, relatively simple, and command-conditioned. However, without a checkpoint, the requested 100-step pretrained rollout could not be performed.

Training from scratch is not a light Step 2 substitute here. The tuned Go1 PPO config in the repo uses `200_000_000` timesteps, so starting training would be a separate training task rather than a quick validation.

## Step 3 - Vision Rendering Test

Two different rendering paths were tested.

### 3.1 MuJoCo EGL Render

Plain MuJoCo EGL rendering through `env.render(...)` works.

Generated images:

- `mujoco_playground/artifacts/go1_top_egl.png`
- `mujoco_playground/artifacts/go1_back_egl.png`

Summary:

| Camera | Shape | Range |
| --- | --- | --- |
| `top` | `(480, 640, 3)` | `7..255` |
| `back` | `(480, 640, 3)` | `8..255` |

This confirms headless EGL camera rendering can work for Playground's MuJoCo model path.

### 3.2 MJWarp Vision Batch Renderer

The official Playground vision path fails before image generation.

Smoke test:

```bash
python mujoco_playground/vision_batch_renderer_smoke.py
```

Observed error:

```text
AttributeError: type object 'int' has no attribute 'WARP'
```

Failure location:

```text
mujoco/mjx/_src/io.py, line 548
graph_mode = graph_mode or getattr(mjxw.types.GraphMode, 'WARP')
```

This happens when loading a vision-enabled environment:

```python
env_cfg = registry.get_default_config("CartpoleBalance")
env_cfg.vision = True
env_cfg.vision_config.nworld = 1
env = registry.load("CartpoleBalance", config=env_cfg)
```

It also happens when loading locomotion directly with `impl="warp"`:

```python
registry.load("Go1JoystickFlatTerrain", config_overrides={"impl": "warp"})
```

Artifact:

- `mujoco_playground/artifacts/step3_mjwarp_vision_summary.json`

## Comparison Table

| Environment | Stability | Goal-aware / command-aware | Vision availability |
| --- | --- | --- | --- |
| Isaac Lab H1 | physics stable | velocity command via H1 controller | blocked by Vulkan graphics device |
| H1 Isaac to MuJoCo sim2sim | unstable, collapses around 1.5s | policy command path loads, dynamics mismatch | MuJoCo EGL can render, but policy unusable |
| MuJoCo Ant-v4 SAC | stable 100-step rollout | no, forward-reward only | MuJoCo EGL likely usable |
| MuJoCo Playground Go1 | physics loads/steps with JAX MJX | yes, joystick `(vx, vy, yaw_rate)` in obs | plain MuJoCo EGL works; MJWarp vision path currently blocked |

## Decision

Do not continue with MuJoCo Playground as the main vision-ready locomotion option until the MJWarp/Warp compatibility error is resolved. It is promising structurally because the Go1/G1 locomotion tasks are command-conditioned, but the key reason for investigating it was batch vision rendering, and that path currently fails in this container.
