# Humanoid System 2 Navigation

Research prototype for adding sparse LLM intervention to a pretrained humanoid locomotion policy.

The project follows a two-system architecture:

- System 1: low-level velocity-command locomotion policy.
- System 2: vision-language model that intervenes only when progress uncertainty is high.

Primary target:

- Isaac Lab + Unitree H1, task `Isaac-Velocity-Flat-H1-v0`.

Fallback:

- MuJoCo `Humanoid-v4` with a command-conditioned locomotion policy.

See:

- `docs/SETUP.md`
- `docs/EXPERIMENT_LOG.md`
- `docs/GITHUB_RECOVERY.md`

