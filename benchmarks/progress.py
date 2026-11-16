#!/usr/bin/env python3
"""benchmarks/progress.py - show what the benchmark runs are doing, with time left. Refreshes every 10 s.
  python benchmarks/progress.py          (Ctrl+C to stop)"""
import glob
import os
import re
import time

LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "logs")


def snapshot():
    lines = [time.strftime("%H:%M:%S") + "  benchmark progress"]
    run = os.path.join(LOGS, "run_v5.log")
    if os.path.exists(run):
        steps = [l.strip() for l in open(run, encoding="utf-8", errors="ignore") if l.startswith("[step]")]
        lines += ["  " + s for s in steps]
    for f in sorted(glob.glob(os.path.join(LOGS, "*.log")), key=os.path.getmtime)[-4:]:
        if time.time() - os.path.getmtime(f) > 3600:
            continue
        txt = open(f, encoding="utf-8", errors="ignore").read()
        prog = re.findall(r"\[progress\] .*", txt)
        done = re.findall(r"^EER .*", txt, re.M)
        state = prog[-1].replace("[progress] ", "") if prog else "starting"
        if "saved" in txt.splitlines()[-1:][0] if txt.strip() else False:
            state = "finished"
        lines.append(f"  {os.path.basename(f):24s} {state}" + (f"   | last result: {done[-1][:60]}" if done else ""))
    return "\n".join(lines)


if __name__ == "__main__":
    try:
        while True:
            os.system("cls" if os.name == "nt" else "clear")
            print(snapshot(), flush=True)
            time.sleep(10)
    except KeyboardInterrupt:
        pass
