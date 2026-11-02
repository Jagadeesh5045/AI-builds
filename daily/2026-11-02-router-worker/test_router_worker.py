"""Tests for router_worker.py -- run: python3 test_router_worker.py"""
import unittest

from router_worker import (
    EventBus, Ticket, Worker, Router, Watchdog,
    proactive_scan, run_simulation, sequential_makespan, build_demo,
    AWAITING, ACCEPTED, FAILED,
)


def make_router(n_workers=2):
    bus = EventBus()
    router = Router(bus, [Worker(f"w{i+1}", bus) for i in range(n_workers)])
    return bus, router, Watchdog(bus, timeout=3)


class TestRouterWorkerSplit(unittest.TestCase):
    def test_router_never_executes(self):
        _, router, _, _ = build_demo()
        self.assertEqual(router.executed, 0)
        self.assertGreater(sum(w.executed for w in router.workers), 0)
        self.assertFalse(hasattr(router, "execute"))

    def test_worker_isolation(self):
        bus = EventBus()
        a, b = Worker("a", bus), Worker("b", bus)
        a.scratch["secret"] = 1
        self.assertNotIn("secret", b.scratch)


class TestScheduling(unittest.TestCase):
    def test_dependency_order_respected(self):
        bus, router, wd = make_router()
        router.submit(Ticket("a", "a", 2))
        router.submit(Ticket("b", "b", 1, deps=("a",)))
        makespan = run_simulation(router, wd)
        self.assertGreaterEqual(router.tickets["b"].started_at,
                                router.tickets["a"].finished_at)
        self.assertEqual(router.tickets["b"].status, AWAITING)
        self.assertGreaterEqual(makespan, 2)

    def test_parallel_is_faster_than_sequential(self):
        _, router, _, makespan = build_demo()
        self.assertLess(makespan, sequential_makespan(router))

    def test_cycle_detected(self):
        _, router, _ = make_router()
        router.submit(Ticket("x", "x", 1, deps=("y",)))
        router.submit(Ticket("y", "y", 1, deps=("x",)))
        with self.assertRaises(ValueError):
            router.check_cycles()

    def test_unknown_dependency_detected(self):
        _, router, _ = make_router()
        router.submit(Ticket("x", "x", 1, deps=("ghost",)))
        with self.assertRaises(ValueError):
            router.check_cycles()

    def test_duplicate_ticket_rejected(self):
        _, router, _ = make_router()
        router.submit(Ticket("x", "x", 1))
        with self.assertRaises(ValueError):
            router.submit(Ticket("x", "again", 1))


class TestRecovery(unittest.TestCase):
    def test_flaky_ticket_recovers_on_retry(self):
        bus, router, wd = make_router(n_workers=1)
        router.submit(Ticket("flaky", "flaky", 1, flaky_failures=1))
        run_simulation(router, wd)
        t = router.tickets["flaky"]
        self.assertEqual(t.attempts, 2)
        self.assertEqual(t.status, AWAITING)
        self.assertEqual(bus.count("ticket_failed"), 1)

    def test_watchdog_restarts_stalled_worker_and_ticket_completes(self):
        bus, router, wd = make_router(n_workers=1)
        router.submit(Ticket("hang", "hangs first attempt", 2, stall=True))
        run_simulation(router, wd)
        self.assertEqual(wd.restarts, 1)
        self.assertEqual(bus.count("ticket_requeued"), 1)
        self.assertEqual(router.tickets["hang"].status, AWAITING)


class TestProactiveAndHuman(unittest.TestCase):
    def test_scanner_finds_only_real_work(self):
        found = proactive_scan({
            "a.py": {"todo": True, "tests_pass": True},
            "b.py": {"todo": False, "tests_pass": False},
            "c.py": {"todo": False, "tests_pass": True},
        })
        self.assertEqual([t.tid for t in found], ["todo-a_py", "fix-b_py"])

    def test_human_acceptance_gate(self):
        bus, router, wd = make_router(n_workers=1)
        router.submit(Ticket("job", "job", 1))
        run_simulation(router, wd)
        self.assertEqual(router.tickets["job"].status, AWAITING)
        router.accept("job", 99)
        self.assertEqual(router.tickets["job"].status, ACCEPTED)

    def test_human_rejection_requeues(self):
        bus, router, wd = make_router(n_workers=1)
        router.submit(Ticket("job", "job", 1))
        run_simulation(router, wd)
        router.accept("job", 99, approve=False)
        self.assertEqual(router.tickets["job"].status, FAILED)


class TestBatching(unittest.TestCase):
    def test_reported_batching_arithmetic(self):
        self.assertEqual(Router.amortised_decision_s(1), 6.4)
        self.assertEqual(Router.amortised_decision_s(4), 1.6)

    def test_batching_rejects_zero(self):
        with self.assertRaises(ValueError):
            Router.amortised_decision_s(0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
