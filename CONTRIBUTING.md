# Contributing

Research-lab tool (AnkiLab, FAU), developed alongside our paper.

## Setup

```bash
git clone https://github.com/neuluna/registration_gui.git
cd registration_gui

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124  # adjust to your driver
pip install -e .
```

Usage: see [README](README.md).

## Code style

[black](https://black.readthedocs.io/), line length 88
(`pyproject.toml` → `[tool.black]`). Comments explain *why*, not *what*;
match the existing terse style rather than adding narrative docstrings.

## Licensing

MIT License (`LICENSE`), except `src/CoReg/cotracker/` (vendored
Meta CoTracker source), under a separate CC BY-NC 4.0 license — see
`src/CoReg/cotracker/LICENSE` and `NOTICE`. Treat that directory as
read-only vendored code; do not add to it as if MIT-licensed.

Submitted changes are assumed distributable under the repository's
license terms (MIT outside `cotracker/`).

## Issues / changes

GitHub issue or PR against `main`. State motivation and validation;
no PR template.
