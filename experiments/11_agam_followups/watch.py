"""Count-based watcher: waits for agam_main, then spawns agam_nla, waits, collects both summaries and
downloads the result files from rfm-results:/11_agam_followups into results/phaseB/."""
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
LOG = Path("results/phaseB/watch.log")
FILES = ("acts.pt", "map_raw.pt", "jl_top10.pt", "ol_sweep.pt", "ol_gens.pt", "nla_sweep.pt", "nla_gens.pt",
         "ol_bank.pt", "summary_main.json", "summary_nla.json", "summary_bank.json")


def log(s: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(f"{time.strftime('%H:%M:%S', time.gmtime())} {s}\n")


def run(*a: str) -> str:
    r = subprocess.run([PY, "spawn_driver.py", *a], capture_output=True, text=True)
    lines = r.stdout.strip().splitlines()
    return lines[-1] if lines else (r.stderr.strip()[-200:] or "")


def wait(fn: str) -> bool:
    fails = 0
    while True:
        s = run("status", fn)
        log(f"{fn}: {s}")
        if s.startswith("done"):
            return True
        if s.startswith("running"):
            fails = 0
        else:
            fails += 1
            if fails >= 40:
                log(f"GAVE UP on {fn}: {s}")
                return False
        time.sleep(90)


ok = wait("agam_main")
log(run("collect", "agam_main"))
if ok:
    log(run("spawn", "agam_nla"))
    wait("agam_nla")
    log(run("collect", "agam_nla"))
for f in FILES:
    r = subprocess.run(["modal", "volume", "get", "--force", "rfm-results", f"11_agam_followups/{f}", f"results/phaseB/{f}"],
                       capture_output=True, text=True)
    log(f"get {f}: rc={r.returncode} {r.stderr.strip()[-200:]}")
log("WATCH_DONE")
