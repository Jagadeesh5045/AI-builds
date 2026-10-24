"""Demo: route a mixed workload across Standard / Highspeed, gate outputs.

Run:  python3 demo.py
"""

from router import (
    STANDARD, HIGHSPEED, Task, ModelOutput, route, cost, latency_seconds,
    check_output, generation_cost_factor, compare_k3_k31,
)

WORKLOAD = [
    Task("coding", 20_000, 8_000, 5_000, urgency=0.90, deadline_seconds=5),
    Task("finance", 200_000, 4_000, 150_000, urgency=0.10, deadline_seconds=60),
    Task("cad", 500_000, 20_000, 400_000, urgency=0.40, deadline_seconds=30),
    Task("agent", 50_000, 10_000, 0, urgency=0.50, deadline_seconds=3),
    Task("coding", 10_000, 3_000, 0, urgency=0.00, deadline_seconds=600),
    Task("finance", 30_000, 5_000, 10_000, urgency=0.85, deadline_seconds=10),
    Task("cad", 100_000, 6_000, 20_000, urgency=0.20, deadline_seconds=60),
    Task("agent", 400_000, 10_000, 300_000, urgency=0.35, deadline_seconds=60),
]

OUTPUTS = [
    ModelOutput("coding", "def add(a, b): return a + b",
                {"language": "python", "code": "def add(a, b): return a + b"}),
    ModelOutput("finance", "Q3 summary for ACME",
                {"ticker": "ACME"},  # 'total' missing -> gate must fail
                numeric_checks=[("revenue", 100.0, 100.0, 0.01)]),
    ModelOutput("cad", "Bracket, TODO finish dimensions",
                {"units": "mm", "dimensions": "40x20"}),  # placeholder -> fail
    ModelOutput("finance", "Q3 summary for ACME, total 100",
                {"ticker": "ACME", "total": "100"},
                numeric_checks=[("revenue", 100.0, 118.0, 0.01)]),  # fail
]


def main():
    routed_cost = routed_lat = std_cost = hs_cost = 0.0
    counts = {}
    print("Routing a mixed 8-task workload:")
    for i, task in enumerate(WORKLOAD, 1):
        decision = route(task)
        c = cost(decision.variant, task.input_tokens, task.output_tokens,
                 task.cached_tokens)
        lat = latency_seconds(decision.variant, task.input_tokens,
                              task.output_tokens)
        routed_cost += c
        routed_lat += lat
        std_cost += cost(STANDARD, task.input_tokens, task.output_tokens,
                         task.cached_tokens)
        hs_cost += cost(HIGHSPEED, task.input_tokens, task.output_tokens,
                        task.cached_tokens)
        counts[decision.variant.name] = counts.get(decision.variant.name, 0) + 1
        print(f"  {i}. {task.kind:8s} -> {decision.variant.name:28s} "
              f"${c:.4f} {lat:.2f}s ({decision.reason})")
    print(f"\nRouted total: ${routed_cost:.4f}, {routed_lat:.2f}s, {counts}")
    print(f"All-Standard would cost ${std_cost:.4f}; "
          f"all-Highspeed would cost ${hs_cost:.4f}")
    print(f"Routing saves ${hs_cost - routed_cost:.4f} vs all-Highspeed "
          f"({100 * (hs_cost - routed_cost) / hs_cost:.1f}%)")

    print("\nQuality gate on 4 sample outputs:")
    passed = 0
    for i, out in enumerate(OUTPUTS, 1):
        result = check_output(out)
        passed += result.passed
        print(f"  {i}. {out.kind:8s} passed={result.passed} "
              f"reasons={result.reasons}")
    print(f"Gate: {passed}/4 passed, {len(OUTPUTS) - passed} rejected")

    print("\nPer-token price vs per-task cost (K3 -> K3.1 Pro):")
    print(f"  output-cost multiplier: "
          f"{generation_cost_factor(15.0, 10.0, 0.14):.3f} "
          "(price factor 0.667 x 1.14 token growth)")
    cmp_ = compare_k3_k31(2_000, 10_000)
    print(f"  example task (2k in, 10k out): K3 ${cmp_['k3_cost']:.4f} vs "
          f"K3.1 Pro ${cmp_['k31_cost']:.4f} at "
          f"{cmp_['k31_output_tokens']} output tokens")


if __name__ == "__main__":
    main()
