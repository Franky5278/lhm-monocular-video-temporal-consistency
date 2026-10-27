"""Write RUN_STATE.json for the experiment monitor while a run is in progress.

Reads stage records only; never touches the experiment process.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(os.environ.get("LHM_ROOT", "/workspace/lhm"))
RESULTS = Path(os.environ.get("LHM_RESULTS", ROOT / "results" / "experiment_v1"))
STATE = ROOT / "RUN_STATE.json"


def gpu() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip()
        used, total, util = [x.strip() for x in out.split(",")]
        return f"{used}/{total} MiB · {util}%"
    except Exception:
        return "n/a"


def collect() -> dict:
    stages, done, failed, running = [], 0, 0, ""
    if RESULTS.exists():
        for record in sorted(RESULTS.rglob("stage.json")):
            try:
                data = json.loads(record.read_text(encoding="utf8"))
            except Exception:
                continue
            name = str(record.parent.relative_to(RESULTS)).replace("\\", "/")
            status = data.get("status", "?")
            secs = data.get("seconds")
            stages.append({"name": name,
                           "status": status if secs is None else f"{status} {secs:.0f}s"})
            if status == "success":
                done += 1
            elif status == "failed":
                failed += 1
            elif status == "running":
                running = name
    started = ROOT / "GPU_STARTED_UNIX"
    deadline = ROOT / "GPU_DEADLINE_UNIX"
    return {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z",
                                    time.localtime(int(started.read_text().strip())))
                      if started.exists() else "",
        "stage": running or ("finished" if (ROOT / "GPU_JOB_EXITED").exists() else "starting"),
        "detail": f"success {done} · failed {failed} · running {running or 'none'}",
        "stages": stages[-14:],
        "gpu_hours": round((time.time() - int(started.read_text())) / 3600, 3)
                     if started.exists() else 0,
        "deadline_left_s": int(deadline.read_text()) - int(time.time())
                           if deadline.exists() else None,
        "gpu": gpu(),
    }


def main() -> None:
    while True:
        try:
            payload = collect()
            tmp = STATE.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf8")
            tmp.replace(STATE)
        except Exception:
            pass
        time.sleep(15)


if __name__ == "__main__":
    main()
