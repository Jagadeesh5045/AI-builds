"""Run every red-team probe under both agent policies and print the scoreboard."""

import shutil
import sys
import tempfile

sys.path.insert(0, __import__("os").path.dirname(__file__))

from probes import PROBES, run_all


def main():
    tmp = tempfile.mkdtemp(prefix="ml4-harness-")
    try:
        for policy in ("naive", "guarded"):
            print("policy: %s" % policy)
            print("-" * 64)
            contained = 0
            for pid, name, outcome, detail in run_all(tmp, policy):
                mark = "CONTAINED" if outcome == "contained" else "MISSED   "
                contained += outcome == "contained"
                print("  [%s] %-20s %s" % (mark, pid, detail))
            print("  score: %d/%d contained" % (contained, len(PROBES)))
            print()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
