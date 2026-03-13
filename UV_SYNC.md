# UV Sync Setup

This repository can now be recreated with `uv sync` using the root `pyproject.toml` and `uv.lock`.
The current lock is targeted at Linux `x86_64` because that matches the training machine and server workflow.

## Scope

`uv sync` reconstructs the Python environment, including:

- Isaac Sim pip packages
- Isaac Lab local editable packages under `source/`
- RL frameworks installed by `./isaaclab.sh -i`
  - `rl_games`
  - `rsl_rl`
  - `sb3`
  - `skrl`
  - `robomimic`

It does **not** install:

- OS packages such as `cmake` and `build-essential`
- local USD assets under `my_assets/` if they are not committed to git

## One-time machine prerequisites

```bash
sudo apt-get update
sudo apt-get install -y cmake build-essential
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Recreate the environment

From the repository root:

```bash
uv sync --locked
source .venv/bin/activate
```

If the server does not already have Python 3.11:

```bash
uv python install 3.11
uv sync --locked
```

## Notes

- `uv sync` creates `.venv/` in the repository root.
- the lock currently targets Linux `x86_64`
- `isaaclab.sh` already detects a uv virtual environment and uses `uv pip` when appropriate.
- If your task config references files in `my_assets/`, those assets must exist on the target machine too.

## Updating the lock file

When dependencies change:

```bash
uv lock
```

Then commit both:

- `pyproject.toml`
- `uv.lock`
