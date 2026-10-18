"""Markdown rows from several runs' results.json: python scripts/results_table.py checkpoints/synthetic-*"""

import json
import sys
from pathlib import Path

for run in sorted(Path(p) for p in sys.argv[1:]):
    r = json.loads((run / "results.json").read_text())

    print(f"| `{r['model']}` | {r['parameters'] / 1e6:.2f} M | {100 * r['wer']:.1f}% | {100 * r['cer']:.1f}% | "
          f"{r['train_seconds'] / 60:.0f} min |")
