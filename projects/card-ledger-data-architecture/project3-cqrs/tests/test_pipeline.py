import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from events import FileLogTransport, make_event
from projector import Projector
from run_pipeline import generate, malformed_events
from write_model import WriteModel


def build(tmp_path):
    base = str(tmp_path)
    rows = generate(200, seed=42)
    write = WriteModel(base)
    write.insert_transactions(rows)
    events = [make_event(r) for r in rows]
    transport = FileLogTransport(base)
    transport.publish(events)
    transport.publish(malformed_events()[:5])
    stream = transport.consume_all(len(events) + 5)
    proj = Projector(base)
    proj.apply(stream)
    return write, proj, stream


def test_totals_match(tmp_path):
    write, proj, _ = build(tmp_path)
    assert write.totals_by_member_period() == proj.totals_by_member_period()
    assert write.total_pence() == proj.total_pence()


def test_idempotent_replay(tmp_path):
    write, proj, stream = build(tmp_path)
    before = proj.total_pence()
    proj.apply(stream[:50])
    assert proj.duplicates >= 50
    assert proj.total_pence() == before


def test_dead_letter_capture(tmp_path):
    _, proj, _ = build(tmp_path)
    assert proj.dead == 5
    reasons = proj.conn.execute("SELECT DISTINCT reason FROM dead_letters").fetchall()
    assert len(reasons) >= 2
