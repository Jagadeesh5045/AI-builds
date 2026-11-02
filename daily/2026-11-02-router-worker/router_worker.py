"""
Router/Worker super-agent simulation, in the spirit of DeepSeek Harness's
dsh-superagent-prototype (open-sourced 31 Oct 2026, reported 1 Nov 2026).

The prototype's thesis, as reported by Zhidx / PoYo (1 Nov 2026):
  * an always-online agent that runs tasks in parallel, proactively finds
    work, and recovers from failures;
  * the human role moves up to decision and acceptance;
  * five pieces: dual-layer Router/Worker, event-driven design, async
    multi-channel interaction, separation of compute / state / environment,
    and an external Watchdog;
  * the Router does resident global routing, human interaction and dispatch
    ONLY -- it never does heavy task execution;
  * Workers run isolated sessions for single tasks and report to the Router.

This file is a deterministic, stdlib-only, simulated-clock model of those
ideas. No real time passes, no network, no LLM. Time is ticks.

Reported figures from the prototype's preliminary (synthetic / simplified)
experiments, quoted in the README, are the lab's, not this toy's.
"""

from dataclasses import dataclass, field

# ---------------------------------------------------------------- events

class EventBus:
    """Append-only event log. Everything observable happens as an event."""

    def __init__(self):
        self.events = []  # list of (tick, kind, detail)

    def publish(self, tick, kind, detail):
        self.events.append((tick, kind, detail))

    def kinds(self):
        return [k for _, k, _ in self.events]

    def count(self, kind):
        return sum(1 for _, k, _ in self.events if k == kind)


# ---------------------------------------------------------------- tickets

PENDING, READY, RUNNING, DONE, FAILED, AWAITING, ACCEPTED, BLOCKED = (
    "pending", "ready", "running", "done", "failed",
    "awaiting_acceptance", "accepted", "blocked",
)


@dataclass
class Ticket:
    tid: str
    title: str
    duration: int                 # simulated ticks of worker time
    deps: tuple = ()
    flaky_failures: int = 0       # fail this many attempts before succeeding
    stall: bool = False           # worker hangs (no heartbeat) on attempt 1
    status: str = PENDING
    attempts: int = 0
    started_at: int = -1
    finished_at: int = -1
    result: str = ""


# ---------------------------------------------------------------- workers

class Worker:
    """An isolated session. Its scratch state is its own: another worker
    cannot see it, and it is wiped when the watchdog restarts the worker."""

    def __init__(self, wid, bus):
        self.wid = wid
        self.bus = bus
        self.scratch = {}          # isolated per-worker state
        self.current = None        # Ticket or None
        self.progress = 0
        self.last_heartbeat = 0
        self.executed = 0          # tickets actually executed by this worker
        self.alive = True

    def heartbeat(self, tick):
        self.last_heartbeat = tick

    def start(self, ticket, tick):
        self.current = ticket
        self.progress = 0
        ticket.status = RUNNING
        ticket.attempts += 1
        ticket.started_at = tick
        self.heartbeat(tick)
        self.scratch[ticket.tid] = "working"
        self.bus.publish(tick, "worker_started", f"{self.wid}:{ticket.tid}")

    def step(self, tick):
        """Advance one tick. Returns the finished ticket or None."""
        t = self.current
        if t is None:
            self.heartbeat(tick)
            return None
        if t.stall and t.attempts == 1:
            # Hung session: no heartbeat, no progress. The watchdog's job.
            return None
        self.heartbeat(tick)
        self.progress += 1
        if self.progress < t.duration:
            return None
        # Finishing this tick.
        self.current = None
        self.executed += 1
        if t.attempts <= t.flaky_failures:
            t.status = FAILED
            self.bus.publish(tick, "ticket_failed", t.tid)
        else:
            t.status = AWAITING
            t.finished_at = tick
            t.result = f"done:{t.tid}"
            self.bus.publish(tick, "ticket_done", t.tid)
        return t


# ---------------------------------------------------------------- router

class Router:
    """Dispatch only. By construction it has no execute() method and its
    `executed` counter must stay at zero -- a test enforces this split."""

    UNBATCHED_DECISION_S = 6.4   # reported amortised figure, unbatched
    BATCHED_TARGET_S = 1.6       # reported amortised figure, batched

    def __init__(self, bus, workers):
        self.bus = bus
        self.workers = workers
        self.tickets = {}
        self.executed = 0
        self.decisions = 0

    def submit(self, ticket, tick=0):
        if ticket.tid in self.tickets:
            raise ValueError(f"duplicate ticket id {ticket.tid}")
        self.tickets[ticket.tid] = ticket
        self.bus.publish(tick, "ticket_submitted", ticket.tid)

    def check_cycles(self):
        """Raise ValueError if the dependency graph has a cycle."""
        visiting, done = set(), set()

        def visit(tid, path):
            if tid in done:
                return
            if tid in visiting:
                raise ValueError(f"dependency cycle: {'->'.join(path + [tid])}")
            visiting.add(tid)
            for dep in self.tickets[tid].deps:
                if dep not in self.tickets:
                    raise ValueError(f"{tid} depends on unknown {dep}")
                visit(dep, path + [tid])
            visiting.discard(tid)
            done.add(tid)

        for tid in self.tickets:
            visit(tid, [])

    def ready(self):
        out = []
        for t in self.tickets.values():
            if t.status in (PENDING, FAILED) and all(
                self.tickets[d].status in (DONE, AWAITING, ACCEPTED)
                for d in t.deps
            ):
                out.append(t)
        return out

    def blocked(self):
        return [t for t in self.tickets.values()
                if t.status == PENDING and t.deps and not all(
                    self.tickets[d].status in (DONE, AWAITING, ACCEPTED)
                    for d in t.deps)]

    def dispatch_batch(self, tick):
        """One batched dispatch round: assign every ready ticket that fits
        an idle worker. Returns how many decisions this round made.
        Batching is the point: one inference serves the whole round."""
        idle = [w for w in self.workers if w.alive and w.current is None]
        batch = []
        for ticket in self.ready():
            if not idle:
                break
            worker = idle.pop(0)
            worker.start(ticket, tick)
            batch.append(ticket.tid)
            self.decisions += 1
        if batch:
            self.bus.publish(tick, "router_batch", ",".join(batch))
        return len(batch)

    @staticmethod
    def amortised_decision_s(batch_size):
        """Reported effect, modelled exactly: batching 4 decisions into one
        inference cuts amortised time per decision from 6.4s to 1.6s.
        amortised = 6.4 / batch_size, floored at the reported 1.6s target
        is not assumed beyond batch 4 -- larger batches keep dividing the
        single shared call in this toy, which the README flags as a toy
        simplification."""
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        return Router.UNBATCHED_DECISION_S / batch_size

    def accept(self, tid, tick, approve=True):
        """The human's job in this architecture: decision and acceptance."""
        t = self.tickets[tid]
        if t.status != AWAITING:
            raise ValueError(f"{tid} is not awaiting acceptance ({t.status})")
        if approve:
            t.status = ACCEPTED
            self.bus.publish(tick, "human_accepted", tid)
        else:
            t.status = FAILED  # rejected work goes back into the pool
            self.bus.publish(tick, "human_rejected", tid)


# ---------------------------------------------------------------- watchdog

class Watchdog:
    """External to both Router and Workers. If a worker stops heartbeating,
    the watchdog kills it, requeues its ticket, and spawns a replacement."""

    def __init__(self, bus, timeout=3):
        self.bus = bus
        self.timeout = timeout
        self.restarts = 0

    def check(self, router, tick):
        for i, w in enumerate(router.workers):
            if not w.alive:
                continue
            if tick - w.last_heartbeat > self.timeout:
                stuck = w.current
                w.alive = False
                self.restarts += 1
                self.bus.publish(tick, "watchdog_restart", w.wid)
                if stuck is not None:
                    stuck.status = FAILED  # requeueable via Router.ready()
                    stuck.stall = False    # the hang was the session, not the job
                    self.bus.publish(tick, "ticket_requeued", stuck.tid)
                router.workers[i] = Worker(f"{w.wid}r{self.restarts}", router.bus)
                router.workers[i].last_heartbeat = tick
        return self.restarts


# ---------------------------------------------------------------- scanner

def proactive_scan(repo_state):
    """Proactively find work, no human request needed.
    repo_state: {path: {"todo": bool, "tests_pass": bool}}"""
    tickets = []
    for path, state in sorted(repo_state.items()):
        slug = path.replace("/", "_").replace(".", "_")
        if state.get("todo"):
            tickets.append(Ticket(f"todo-{slug}", f"Resolve TODO in {path}", 2))
        if state.get("tests_pass") is False:
            tickets.append(Ticket(f"fix-{slug}", f"Fix failing tests in {path}", 3))
    return tickets


# ---------------------------------------------------------------- simulation

def run_simulation(router, watchdog, max_ticks=200):
    """Discrete-tick loop: watchdog, then dispatch, then workers step.
    Returns the tick at which every ticket reached a terminal-for-now state
    (awaiting acceptance / accepted), i.e. the parallel makespan."""
    router.check_cycles()
    tick = 0
    while tick <= max_ticks:
        watchdog.check(router, tick)
        router.dispatch_batch(tick)
        for w in list(router.workers):
            if w.alive:
                w.step(tick)
        if all(t.status in (AWAITING, ACCEPTED)
               for t in router.tickets.values()):
            return tick
        tick += 1
    raise RuntimeError("simulation did not finish within max_ticks")


def sequential_makespan(router):
    """Baseline: one worker, no parallelism -- sum of all durations,
    counting a flaky ticket once per attempt it needs."""
    total = 0
    for t in router.tickets.values():
        total += t.duration * (t.flaky_failures + 1)
    return total


def build_demo():
    bus = EventBus()
    workers = [Worker(f"w{i+1}", bus) for i in range(3)]
    router = Router(bus, workers)
    watchdog = Watchdog(bus, timeout=3)
    # A small repo job: scaffold first, two independent modules in parallel,
    # an integration step that needs both, a flaky test-fix, and docs.
    router.submit(Ticket("scaffold", "Scaffold repo layout", 2))
    router.submit(Ticket("api", "Build API module", 4, deps=("scaffold",)))
    router.submit(Ticket("store", "Build storage module", 3, deps=("scaffold",)))
    router.submit(Ticket("integ", "Integrate API + storage", 3,
                         deps=("api", "store")))
    router.submit(Ticket("fixtests", "Fix flaky integration tests", 2,
                         deps=("integ",), flaky_failures=1))
    router.submit(Ticket("docs", "Write README section", 1, deps=("integ",)))
    # Proactive work found by scanning, not requested by anyone.
    for t in proactive_scan({
        "src/auth.py": {"todo": True, "tests_pass": True},
        "src/billing.py": {"todo": False, "tests_pass": False},
        "src/clean.py": {"todo": False, "tests_pass": True},
    }):
        router.submit(t)
    makespan = run_simulation(router, watchdog)
    for tid in list(router.tickets):
        if router.tickets[tid].status == AWAITING:
            router.accept(tid, makespan)
    return bus, router, watchdog, makespan


if __name__ == "__main__":
    bus, router, watchdog, makespan = build_demo()
    seq = sequential_makespan(router)
    print(f"tickets: {len(router.tickets)}  workers: 3")
    print(f"parallel makespan: {makespan} ticks   sequential baseline: {seq} ticks")
    print(f"speedup: {seq / max(makespan, 1):.2f}x (simulated ticks)")
    print(f"router executed tasks itself: {router.executed} (must be 0)")
    print(f"worker executions: {sum(w.executed for w in router.workers)}")
    print(f"watchdog restarts: {watchdog.restarts}")
    print(f"router decisions: {router.decisions} in "
          f"{bus.count('router_batch')} batched rounds")
    print(f"amortised decision time: batch of 1 = "
          f"{Router.amortised_decision_s(1):.1f}s, batch of 4 = "
          f"{Router.amortised_decision_s(4):.1f}s (reported prototype effect)")
    print(f"accepted: {sum(1 for t in router.tickets.values() if t.status == ACCEPTED)}"
          f"/{len(router.tickets)}")
    print(f"events: {len(bus.events)}")
