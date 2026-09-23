# Setup

Checked on 2026-09-23.

| Tool | Status |
|------|--------|
| python3 | 3.14.4 (OK, >= 3.12) |
| git | 2.53.0 (OK) |
| uv | **missing** |
| numpy, matplotlib, pytest, hypothesis, ruff, mypy | installed per-project by `uv sync` (no system install needed) |

## Manual actions (no sudo needed)

1. Install uv (user-local, goes to `~/.local/bin`):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Make sure `~/.local/bin` is on PATH (the installer usually does this; open a new shell after):

   ```bash
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
   ```

3. Verify:

   ```bash
   uv --version
   ```

## Nothing needs sudo

Python, git and a matplotlib-compatible environment (Agg backend, headless) are all
handled inside the project venv created by uv.

## Git repository

`~/robocell` currently sits inside a git repo rooted at your home directory
(`/home/suhavyy`). The project gets its own repo instead:

```bash
git -C ~/robocell init -b main
```

It reuses your global git identity (`suhanee-gupta`); no `git config` changes.
