"""Toy queen-led multi-agent orchestrator, Ruflo-style. Stdlib only.

Illustrates the ideas behind Cognitum One's Ruflo (announced 13 Oct 2026):
an open-source agent meta-harness / orchestration layer for coding agents.
Per the announcement, Ruflo turns one assistant into a coordinated team of
specialized agents with mission-driven Mods (research, build, test, validate,
secure, benchmark, learn), queen-led swarms, persistent shared memory, routing
of each task to the right model, a run ledger (what ran, what it cost,
whether it passed), and human approval where it matters.

Everything here is a deterministic toy stand-in: the "models" are small
rule-based workers, tokens and costs are simulated, and the repo under test
is a synthetic in-memory fixture. What is real is the *mechanism*: task DAG
decomposition, shared memory, review loop, approval gate, ledger.
"""

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Shared memory: the swarm's persistent, namespaced scratchpad.
# ---------------------------------------------------------------------------
class SharedMemory:
    """Namespaced key-value store shared by every worker in the swarm."""

    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, object]] = {}

    def write(self, namespace: str, key: str, value: object) -> None:
        self._store.setdefault(namespace, {})[key] = value

    def read(self, namespace: str, key: str, default: object = None) -> object:
        return self._store.get(namespace, {}).get(key, default)

    def read_all(self, namespace: str) -> Dict[str, object]:
        return dict(self._store.get(namespace, {}))

    def export(self) -> Dict[str, Dict[str, object]]:
        return {ns: dict(keys) for ns, keys in self._store.items()}


# ---------------------------------------------------------------------------
# Tasks, risk, results.
# ---------------------------------------------------------------------------
class Risk(Enum):
    LOW = "low"
    HIGH = "high"


@dataclass
class Task:
    name: str
    worker_role: str
    payload: Dict[str, object] = field(default_factory=dict)
    risk: Risk = Risk.LOW
    depends_on: List[str] = field(default_factory=list)


@dataclass
class TaskResult:
    task_name: str
    worker_role: str
    success: bool
    output: Dict[str, object] = field(default_factory=dict)
    tokens: int = 0
    findings: List[str] = field(default_factory=list)
    approved: Optional[bool] = None  # set by the gate for HIGH risk tasks


# ---------------------------------------------------------------------------
# Workers: deterministic stand-ins for real specialist models.
# ---------------------------------------------------------------------------
class Worker:
    role = "worker"

    def run(self, task: Task, memory: SharedMemory, repo: Dict[str, str]) -> TaskResult:
        raise NotImplementedError


class Researcher(Worker):
    """Finds the buggy function in the repo and records it in memory."""

    role = "researcher"

    def run(self, task, memory, repo):
        hits = []
        for fname, src in repo.items():
            for i, line in enumerate(src.splitlines(), 1):
                if "def top_k" in line:
                    hits.append({"file": fname, "line": i})
        memory.write("research", "bug_location", hits)
        memory.write("research", "bug_hypothesis",
                     "off-by-one: loop stops one element early")
        return TaskResult(task.name, self.role, success=bool(hits),
                          output={"locations": hits}, tokens=420)


class Coder(Worker):
    """Writes a fix for top_k. First attempt ships an unsafe helper on
    purpose (the review stage exists to catch exactly this); the revision
    pass reads the review findings from memory and ships a clean fix."""

    role = "coder"

    CLEAN_FIX = (
        "import heapq\n\n\n"
        "def top_k(items, k):\n"
        '    """Return the k largest items, largest first."""\n'
        "    if k <= 0:\n"
        "        return []\n"
        "    return heapq.nlargest(k, items)\n"
    )
    # Revision 1 is functionally correct (tests pass) but smuggles in an
    # unsafe eval() helper. The review stage exists to catch exactly this.
    UNSAFE_FIX = (
        "def top_k(items, k, filter_expr='True'):\n"
        '    """Return the k largest items, largest first."""\n'
        "    items = [x for x in items if eval(filter_expr)]\n"
        "    return sorted(items, reverse=True)[:k]\n"
    )

    def run(self, task, memory, repo):
        findings = memory.read("secure", "review_findings", [])
        revision = 1 if not findings else 2
        body = self.CLEAN_FIX if findings else self.UNSAFE_FIX
        repo["utils.py"] = body
        memory.write("build", f"fix_revision_{revision}", body)
        return TaskResult(task.name, self.role, success=True,
                          output={"revision": revision,
                                  "unsafe": not bool(findings)},
                          tokens=780)


class Tester(Worker):
    """Runs deterministic checks against the patched top_k."""

    role = "tester"

    def run(self, task, memory, repo):
        ns: Dict[str, object] = {}
        exec(compile(repo["utils.py"], "utils.py", "exec"), ns)  # noqa: S102
        top_k = ns["top_k"]
        cases = [
            ([5, 1, 9, 3, 7], 3, [9, 7, 5]),
            ([4, 4, 4], 2, [4, 4]),
            ([1, 2], 5, [2, 1]),
            ([], 3, []),
            ([3, 1, 2], 0, []),
        ]
        failures = []
        for items, k, want in cases:
            try:
                got = list(top_k(items, k))
            except Exception as exc:  # noqa: BLE001
                failures.append(f"top_k({items}, {k}) raised {exc!r}")
                continue
            if got != want:
                failures.append(f"top_k({items}, {k}) = {got}, want {want}")
        memory.write("test", "failures", failures)
        return TaskResult(task.name, self.role, success=not failures,
                          output={"failures": failures}, tokens=350,
                          findings=[] if not failures else ["tests failed"])


class Reviewer(Worker):
    """Security/style review. Flags banned patterns in the patched file."""

    role = "reviewer"
    BANNED = ["eval(", "exec(", "os.system", "subprocess", "__import__"]

    def run(self, task, memory, repo):
        src = repo["utils.py"]
        findings = [f"banned pattern {p!r} in utils.py"
                    for p in self.BANNED if p in src]
        memory.write("secure", "review_findings", findings)
        return TaskResult(task.name, self.role, success=True,
                          output={"findings": findings}, tokens=510,
                          findings=findings)


class Benchmarker(Worker):
    """Compares the naive full-sort approach with the heap-based fix."""

    role = "benchmarker"

    @staticmethod
    def _naive_ops(n: int) -> int:
        return n * max(1, n.bit_length())  # ~ n log n comparisons

    @staticmethod
    def _heap_ops(n: int, k: int) -> int:
        return n * max(1, k.bit_length())  # ~ n log k

    def run(self, task, memory, repo):
        n, k = 100_000, 10
        naive, heap = self._naive_ops(n), self._heap_ops(n, k)
        memory.write("benchmark", "op_counts", {"naive": naive, "heap": heap})
        return TaskResult(task.name, self.role, success=True,
                          output={"naive_ops": naive, "heap_ops": heap,
                                  "speedup": round(naive / heap, 1)},
                          tokens=260)


class Learner(Worker):
    """Writes a lesson from this mission into shared memory."""

    role = "learner"

    def run(self, task, memory, repo):
        lesson = ("off-by-one fixed via nlargest; review caught eval() in "
                  "revision 1; always run the reviewer before benchmark.")
        memory.write("lessons", "top_k_mission", lesson)
        return TaskResult(task.name, self.role, success=True,
                          output={"lesson": lesson}, tokens=180)


# ---------------------------------------------------------------------------
# Approval gate: HIGH risk tasks need a human decision.
# ---------------------------------------------------------------------------
class ApprovalGate:
    """LOW risk tasks auto-pass; HIGH risk tasks go to the approver
    callback, which returns True (approve) or False (deny)."""

    def __init__(self, approver: Callable[[Task], bool]) -> None:
        self.approver = approver
        self.decisions: List[Dict[str, object]] = []

    def request(self, task: Task) -> bool:
        if task.risk is Risk.LOW:
            decision = True
        else:
            decision = bool(self.approver(task))
        self.decisions.append({"task": task.name, "risk": task.risk.value,
                               "approved": decision})
        return decision


# ---------------------------------------------------------------------------
# Ledger: what ran, what it cost, whether it passed.
# ---------------------------------------------------------------------------
TOKEN_RATE_USD_PER_M = 3.0  # toy rate, simulated tokens


class Ledger:
    def __init__(self) -> None:
        self.entries: List[Dict[str, object]] = []

    def record(self, result: TaskResult, duration_ms: float) -> None:
        self.entries.append({
            "task": result.task_name,
            "worker": result.worker_role,
            "passed": result.success,
            "tokens": result.tokens,
            "cost_usd": round(result.tokens / 1_000_000 * TOKEN_RATE_USD_PER_M, 6),
            "duration_ms": round(duration_ms, 1),
            "approved": result.approved,
            "findings": result.findings,
        })

    def summary(self) -> Dict[str, object]:
        passed = sum(1 for e in self.entries if e["passed"])
        blocked = sum(1 for e in self.entries if e["approved"] is False)
        return {
            "total_tasks": len(self.entries),
            "passed": passed,
            "failed": len(self.entries) - passed,
            "blocked_by_gate": blocked,
            "total_tokens": sum(e["tokens"] for e in self.entries),
            "total_cost_usd": round(sum(e["cost_usd"] for e in self.entries), 6),
            "approval_decisions": [d for d in self.entries
                                   if d["approved"] is not None],
        }

    def table(self) -> str:
        rows = [f"{'task':<22}{'worker':<12}{'pass':<6}{'tokens':<8}"
                f"{'cost_usd':<10}{'approved'}"]
        for e in self.entries:
            rows.append(f"{e['task']:<22}{e['worker']:<12}"
                        f"{str(e['passed']):<6}{e['tokens']:<8}"
                        f"{e['cost_usd']:<10}{e['approved']}")
        return "\n".join(rows)

    def to_json(self) -> str:
        return json.dumps({"entries": self.entries,
                           "summary": self.summary()}, indent=2)


class DocsPublisher(Worker):
    """Stands in for the HIGH-risk deploy step: publishing docs."""

    role = "publisher"

    def run(self, task, memory, repo):
        repo["CHANGELOG.md"] = "# Changelog\n\n## fixed\n- top_k off-by-one fixed\n"
        return TaskResult(task.name, self.role, success=True,
                          output={"published": "CHANGELOG.md"}, tokens=150)


# ---------------------------------------------------------------------------
# The Queen: decomposes a mission into Ruflo-Mod-style stages and runs them:
# research -> build -> test -> secure(review) -> benchmark -> learn, with
# one revise loop when the review flags something, then a HIGH-risk deploy
# step behind the approval gate.
# ---------------------------------------------------------------------------
@dataclass
class Mission:
    name: str
    goal: str


class Queen:
    def __init__(self, workers: Dict[str, Worker], gate: ApprovalGate,
                 ledger: Ledger, memory: SharedMemory) -> None:
        self.workers = workers
        self.gate = gate
        self.ledger = ledger
        self.memory = memory

    def decompose(self, mission: Mission) -> List[Task]:
        return [
            Task("research", "researcher"),
            Task("build", "coder", depends_on=["research"]),
            Task("test", "tester", depends_on=["build"]),
            Task("secure", "reviewer", depends_on=["test"]),
            Task("benchmark", "benchmarker", depends_on=["secure"]),
            Task("learn", "learner", depends_on=["benchmark"]),
            Task("deploy_docs", "publisher", {"action": "publish docs"},
                 risk=Risk.HIGH, depends_on=["learn"]),
        ]

    def _run_task(self, task: Task, repo: Dict[str, str]) -> TaskResult:
        approved: Optional[bool] = None
        if task.risk is Risk.HIGH:
            approved = self.gate.request(task)
        start = time.perf_counter()
        if approved is False:
            result = TaskResult(task.name, task.worker_role, success=False,
                                output={"blocked": "approval denied"},
                                approved=False)
        else:
            result = self.workers[task.worker_role].run(task, self.memory, repo)
            result.approved = approved
        self.ledger.record(result, (time.perf_counter() - start) * 1000)
        return result

    def run_mission(self, mission: Mission,
                    repo: Dict[str, str]) -> Dict[str, object]:
        tasks = {t.name: t for t in self.decompose(mission)}
        done: Dict[str, TaskResult] = {}

        def run(name: str) -> TaskResult:
            result = self._run_task(tasks[name], repo)
            done[name] = result
            return result

        run("research")
        run("build")                       # revision 1: correct but unsafe
        test_result = run("test")          # passes: the bug is safety, not function
        secure_result = run("secure")      # flags eval()
        revised = False
        if secure_result.findings:
            # Review flagged something: coder revises with the findings in
            # memory, then test + review run again on the revision.
            revised = True
            run("build")                   # revision 2: clean
            test_result = run("test")
            secure_result = run("secure")
        run("benchmark")
        run("learn")
        deploy_result = run("deploy_docs")  # HIGH risk: behind the gate
        return {"mission": mission.name,
                "goal": mission.goal,
                "revised_after_review": revised,
                "final_test_passed": test_result.success,
                "final_review_findings": secure_result.findings,
                "deploy": {"approved": deploy_result.approved,
                           "blocked": deploy_result.approved is False},
                "summary": self.ledger.summary()}


# ---------------------------------------------------------------------------
# Naive baseline: one generalist, no memory sharing, no review, no gate.
# ---------------------------------------------------------------------------
def naive_single_agent(repo: Dict[str, str]) -> Dict[str, object]:
    """A single worker doing research+fix+test inline. It ships the first
    fix it writes, which here carries the unsafe eval() helper, and it has
    no reviewer or approval gate to stop it."""
    memory, ledger = SharedMemory(), Ledger()
    researcher, coder = Researcher(), Coder()
    start = time.perf_counter()
    r1 = researcher.run(Task("research", "researcher"), memory, repo)
    ledger.record(r1, (time.perf_counter() - start) * 1000)
    start = time.perf_counter()
    r2 = coder.run(Task("build", "coder"), memory, repo)  # revision 1: unsafe
    ledger.record(r2, (time.perf_counter() - start) * 1000)
    unsafe_shipped = "eval(" in repo["utils.py"]
    return {"unsafe_shipped": unsafe_shipped,
            "review_findings": 0,  # no reviewer in the loop
            "summary": ledger.summary()}
