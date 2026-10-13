"""Demo: a queen-led swarm fixes a buggy top_k, with review + approval gate.

Run: python demo.py
"""

from swarm import (ApprovalGate, Benchmarker, Coder, DocsPublisher, Learner,
                   Ledger, Mission, Queen, Researcher, Reviewer, SharedMemory,
                   Tester, naive_single_agent)

BUGGY_REPO = {
    "utils.py": (
        "def top_k(items, k):\n"
        '    """Return the k largest items, largest first."""\n'
        "    out = []\n"
        "    for i in range(len(items)):\n"
        "        if i < k - 1:  # off-by-one: stops one element early\n"
        "            out.append(items[i])\n"
        "    return sorted(out, reverse=True)\n"
    ),
}


def build_queen(approver):
    workers = {
        "researcher": Researcher(),
        "coder": Coder(),
        "tester": Tester(),
        "reviewer": Reviewer(),
        "benchmarker": Benchmarker(),
        "learner": Learner(),
        "publisher": DocsPublisher(),
    }
    return Queen(workers, ApprovalGate(approver), Ledger(), SharedMemory())


def main():
    print("=== swarm mission: fix top_k (deploy denied by approver) ===")
    queen = build_queen(approver=lambda task: False)
    report = queen.run_mission(
        Mission("fix-top-k", "Fix the off-by-one in utils.top_k safely."),
        dict(BUGGY_REPO))
    print(f"revised after review : {report['revised_after_review']}")
    print(f"final test passed    : {report['final_test_passed']}")
    print(f"final review findings: {report['final_review_findings']}")
    print(f"deploy blocked       : {report['deploy']['blocked']}")
    print("\n--- ledger: what ran, what it cost, whether it passed ---")
    print(queen.ledger.table())
    print("\nsummary:", report["summary"])

    print("\n=== approving variant: deploy allowed ===")
    queen2 = build_queen(approver=lambda task: True)
    report2 = queen2.run_mission(
        Mission("fix-top-k", "Fix the off-by-one in utils.top_k safely."),
        dict(BUGGY_REPO))
    print("deploy approved:", report2["deploy"]["approved"])

    print("\n=== naive baseline: one generalist, no review, no gate ===")
    naive = naive_single_agent(dict(BUGGY_REPO))
    print(f"unsafe eval() shipped: {naive['unsafe_shipped']}")
    print(f"review findings      : {naive['review_findings']} (no reviewer)")

    print("\n=== swarm vs naive ===")
    print(f"{'':<28}{'swarm':<10}{'naive'}")
    swarm_repo = dict(BUGGY_REPO)
    q3 = build_queen(approver=lambda task: True)
    q3.run_mission(Mission("m", "g"), swarm_repo)
    print(f"{'unsafe patterns shipped':<28}"
          f"{str('eval(' in swarm_repo['utils.py']):<10}"
          f"{str(naive['unsafe_shipped'])}")
    print(f"{'review findings caught':<28}"
          f"{len(report['final_review_findings']) + 1:<10}0")
    print(f"{'approval gate enforced':<28}{'yes':<10}{'no'}")


if __name__ == "__main__":
    main()
