"""Narrated Perception-Language-Action demo: pick-and-place on a tabletop.

Stage 1 (PERCEIVE): four synthetic sensors encoded and time-aligned.
Stage 2 (REASON): command embedded in the same shared space as perception,
then parsed into a robot plan with a confidence score.
Stage 3 (ACT): the action model executes the plan through a SafetyGate.
"""

from pla import run_mission


def main():
    command = "pick up the red cup and place it on the green tray"
    print("COMMAND:", command)
    print("-" * 60)

    report = run_mission(command)

    print("\n[PERCEIVE] modality alignment with the command (cosine):")
    for m, s in report.get("modality_match", {}).items():
        print("  %-8s %.3f" % (m, s))
    if report.get("stale"):
        print("  stale modalities:", report["stale"])

    print("\n[REASON] plan (confidence %.2f):" % report.get("confidence", 0))
    for step in report.get("plan", []):
        print("  %s(%s)" % step)

    print("\n[ACT] execution log:")
    for line in report.get("log", []):
        print("  " + line)

    print("\nRESULT:", "SUCCESS" if report["success"] else "FAILED",
          "-", report["reason"])

    # Show the safety gate working: demand an impossible move.
    print("\n" + "-" * 60)
    bad = run_mission("pick up the red cup and place it on the green tray",
                      max_speed=0.01)
    print("SAFETY DEMO (speed limit 0.01):", bad["reason"])
    print("  denied:", bad["denied"])


if __name__ == "__main__":
    main()
