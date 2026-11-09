"""Snapshot good runs from data/runs.json into seed_runs.json (the sample runs shipped with the app).
Usage: py make_seed.py   (keeps the newest run per prospect that has a result)"""
import json
from pathlib import Path
here = Path(__file__).parent
runs = json.loads((here / "data" / "runs.json").read_text(encoding="utf8"))
best = {}
for r in runs:  # newest first; skip runs that ended needs_edit/error when a cleaner one exists
    key = (r["input"]["name"].lower(), (r["input"].get("company") or "").lower())
    if not r.get("result"):
        continue
    clean = r["status"] not in ("needs_edit", "error")
    if key not in best or (clean and not best[key][0]):
        best[key] = (clean, r)
out = [v[1] for v in best.values()]
(here / "seed_runs.json").write_text(json.dumps(out, ensure_ascii=False, default=str), encoding="utf8")
print(len(out), "sample runs:", ", ".join(r["input"]["name"] + " -> " + r["status"] for r in out))
