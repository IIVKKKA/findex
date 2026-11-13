"""Lost-update demo: threads incrementing shared state without a lock.

Run on 3.13 (GIL) and on 3.13t (free-threaded) and compare:
  uv run python scripts/race_demo.py
  .venv-ft/Scripts/python.exe scripts/race_demo.py
"""

import sys
import threading
from collections import Counter

THREADS, ITERS = 8, 200_000


def run(use_lock: bool) -> tuple[int, int, int]:
    n = 0
    shared: Counter[str] = Counter()
    lock = threading.Lock()

    def work() -> None:
        nonlocal n
        for _ in range(ITERS):
            if use_lock:
                with lock:
                    n += 1
                    shared["x"] += 1
            else:
                n += 1
                shared["x"] += 1

    ts = [threading.Thread(target=work) for _ in range(THREADS)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return THREADS * ITERS, n, shared["x"]


gil = getattr(sys, "_is_gil_enabled", lambda: True)()
print(f"python {sys.version.split()[0]}  GIL enabled: {gil}  switchinterval={sys.getswitchinterval()}")
for interval in (sys.getswitchinterval(), 1e-6):
    sys.setswitchinterval(interval)
    for use_lock in (False, True):
        expected, n, c = run(use_lock)
        print(
            f"switchinterval={interval:<8} lock={use_lock!s:<5} expected={expected} "
            f"int n={n} (lost {expected - n}) Counter={c} (lost {expected - c})"
        )
