"""Count-based watcher for stage 2a: waits for jpp_download and operand_stage2a, then spawns jpp_enrich,
waits for it, collects every summary and downloads results/stage2a/{stage2a.json, jpp.json, dirs_fewshot.pt}.
Tolerates transient status failures (40 in a row before giving up)."""
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
LOG = Path("results/stage2a/watch.log")


def log(s: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(f"{time.strftime('%H:%M:%S', time.gmtime())} {s}\n")


def run(*a: str) -> str:
    return subprocess.run([PY, "spawn_driver2.py", *a], capture_output=True, text=True).stdout.strip().splitlines()[-1:] or [""]


def wait(fn: str) -> bool:
    fails = 0
    while True:
        s = run("status", fn)[0]
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
        time.sleep(60)


ok_main = wait("operand_stage2a")
ok_dl = wait("jpp_download")
log(run("collect", "operand_stage2a")[0])
log(run("collect", "jpp_download")[0])
if ok_main and ok_dl:
    log(run("spawn", "jpp_enrich")[0])
    wait("jpp_enrich")
    log(run("collect", "jpp_enrich")[0])
for f in ("stage2a.json", "jpp.json", "dirs_fewshot.pt"):
    r = subprocess.run(["modal", "volume", "get", "--force", "rfm-results", f"10_operand_gate/stage2a/{f}", f"results/stage2a/{f}"],
                       capture_output=True, text=True)
    log(f"get {f}: rc={r.returncode} {r.stderr.strip()[-200:]}")
log("WATCH_DONE")
