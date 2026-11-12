# DEAD-CELL

[![Stars](https://img.shields.io/github/stars/kapildhinwa2010-dev/DEAD-CELL?style=flat-square)](https://github.com/kapildhinwa2010-dev/DEAD-CELL/stargazers) [![Forks](https://img.shields.io/github/forks/kapildhinwa2010-dev/DEAD-CELL?style=flat-square)](https://github.com/kapildhinwa2010-dev/DEAD-CELL/network/members) [![Issues](https://img.shields.io/github/issues/kapildhinwa2010-dev/DEAD-CELL?style=flat-square)](https://github.com/kapildhinwa2010-dev/DEAD-CELL/issues) [![Top language](https://img.shields.io/github/languages/top/kapildhinwa2010-dev/DEAD-CELL?style=flat-square)](https://github.com/kapildhinwa2010-dev/DEAD-CELL)

The internal graveyard: dead weight rows, dead columns, and neurons that never fire.

Point DEAD-CELL at weight matrices, and optionally at activation snapshots, and it maps the waste inside: exact zero rows, exact zero columns, the fraction of dead parameters, and (with activations) neurons that never rise above a threshold on any sample.

## What it does

- Per-layer zero rows, zero columns, and zero parameter counts.
- Waste fraction per layer and aggregated across the whole model.
- With activation snapshots: never-active neurons and occupancy rate per layer.

## Install

```bash
pip install -e .
```

## Quick start

```python
import numpy as np
from dead_cell.census import dead_cells, total_waste

weights = [np.load(f) for f in ["w0.npy", "w1.npy"]]
acts = [np.load(f) for f in ["a0.npy", "a1.npy"]]
entries = dead_cells(weights, activations=acts)
print(total_waste(entries))
```

## How it works

Zero structure is measured exactly: a row or column is dead only if every entry is exactly zero. Activation death is a threshold question, so neurons count as never active only when they stay at or below the threshold on every recorded sample.

## Honest scope

- Exact-zero tests are strict; small-but-nonzero weights do not count as dead.
- Activation conclusions are only as good as the sample set you record.
- The tool measures waste, not safety; a dead map is a lead, not a verdict.

## License

MIT