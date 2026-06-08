# GitHub Recovery Workflow

Use GitHub as the durable source of truth because `/workspace` is ephemeral.

## Recommended Push Points

- After project skeleton changes.
- After each completed step.
- Before and after long training runs.
- After generating result summaries.

Do not commit large artifacts such as videos, TensorBoard logs, and checkpoints unless there is a deliberate storage plan.

## Token

If HTTPS push asks for credentials, create a GitHub fine-grained personal access token:

1. GitHub profile photo -> Settings.
2. Developer settings -> Personal access tokens -> Fine-grained tokens.
3. Generate new token.
4. Repository access: either this repository only, or selected target repository.
5. Permissions:
   - Contents: Read and write.
   - Metadata: Read-only.
6. Generate token and provide it only when needed for the push command.

Prefer using the token once through Git credential prompt. Do not save it in files inside this repo.

