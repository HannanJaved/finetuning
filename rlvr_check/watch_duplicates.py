#!/usr/bin/env python3
"""Keep whichever copy of each RL run starts first and cancel its twin on the other account.

A copy counts as started once it has been RUNNING for GRACE seconds, so a node that fails at
launch does not cost the twin (same rule as finetuning/qwen3/watch_account_duplicates.sh).
Exits once no run in starts.tsv has more than one copy queued.

Usage: setsid nohup python3 watch_duplicates.py >> watch_duplicates.log 2>&1 < /dev/null &
"""
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
INTERVAL, GRACE = 30, 120
NAMES = {"rlvr-" + line.split("\t")[1] for line in (HERE / "starts.tsv").read_text().splitlines()[1:]}


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def queue():
    out = subprocess.run(["squeue", "-u", os.environ["USER"], "-h", "-o", "%i|%T|%S|%a|%j"],
                         capture_output=True, text=True, check=True).stdout
    jobs = {}
    for line in out.splitlines():
        jid, state, start, account, name = line.split("|", 4)
        if name in NAMES:
            jobs.setdefault(name, []).append((jid, state, start, account))
    return jobs


def running_for(state, start):
    if state != "RUNNING":
        return None
    return time.time() - datetime.strptime(start, "%Y-%m-%dT%H:%M:%S").timestamp()


log(f"watching {len(NAMES)} runs: {sorted(NAMES)}")
while True:
    try:
        jobs = queue()
    except subprocess.CalledProcessError as err:  # slurmctld timeouts happen; retry next round
        log(f"squeue failed: {err.stderr.strip()}")
        time.sleep(INTERVAL)
        continue
    dupes = {name: js for name, js in jobs.items() if len(js) > 1}
    if not dupes:
        log("no duplicates left, exiting")
        break
    for name, js in dupes.items():
        survivors = [(running_for(s, st), jid, acc) for jid, s, st, acc in js]
        survivors = [x for x in survivors if x[0] is not None and x[0] >= GRACE]
        if not survivors:
            continue
        _, keep, keep_acc = max(survivors)  # running longest = started first
        for jid, state, _, acc in js:
            if jid != keep:
                subprocess.run(["scancel", jid])
                log(f"{name}: keep {keep} ({keep_acc}), cancel {jid} ({acc}, {state})")
    time.sleep(INTERVAL)
