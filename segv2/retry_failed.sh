#!/bin/bash
# Second pass over codes that errored in build_log.jsonl (transient Zenodo 504s → no_full_gpkg).
# Used to chain after an already-running build; new runs do this in-process (build_dataset.main).
cd "$(dirname "$0")/.."
while pgrep -f "python3 segv2/build_dataset.py" >/dev/null; do sleep 60; done
CODES=$(python3 -c '
import json,pathlib
ok=set();bad={}
for l in open("data/segv2/build_log.jsonl"):
    m=json.loads(l); c=m["code"]
    if m.get("error") and m["error"] not in ("no_parcels","no_tiles"): bad[c]=m["error"]
    else: ok.add(c); bad.pop(c,None)
print(" ".join(c for c in bad if not pathlib.Path(f"data/segv2/dataset/{c}.parquet").exists()))')
echo "retry pass: $CODES" >> data/segv2/build_stdout.log
[ -n "$CODES" ] && python3 segv2/build_dataset.py $CODES >> data/segv2/build_stdout.log 2>&1
echo FINISHED_RETRY >> data/segv2/build_stdout.log
