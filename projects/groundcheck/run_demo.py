"""Demo: evaluate groundcheck and write report.md with the real numbers."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from groundcheck.evaluate import run_eval

if __name__ == "__main__":
    r = run_eval()
    lines = ["# groundcheck evaluation report", "",
             f"Cases: {r['cases']} (4 grounded, 4 with planted unsupported claims)", "",
             "| case | label | groundedness | unsupported claims | decision |",
             "|---|---|---|---|---|"]
    for i, (label, g, u, d) in enumerate(r["rows"], 1):
        lines.append(f"| {i} | {label} | {g} | {u} | {d} |")
    lines += ["", f"Hallucination detection: precision {r['precision']}, recall {r['recall']} "
              f"(TP {r['tp']}, FP {r['fp']}, FN {r['fn']}, TN {r['tn']})."]
    out = "\n".join(lines) + "\n"
    pathlib.Path("report.md").write_text(out)
    print(out)
