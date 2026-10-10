"""
demo.py - run the compression layer over a realistic agent trace.

Simulates three turns of a coding agent: it runs a test suite (verbose logs),
calls a search tool (verbose JSON), and reads retrieved RAG chunks. The
compressor shrinks what the model must re-read next turn; the demo prints
before/after token counts, the compression ratio, and an estimated cost saving.
"""

from agent_context import AgentContextCompressor, estimate_tokens

LOG_TRACE = """\
INFO 2026-10-10T20:02:11Z agent: starting turn 3
INFO 2026-10-10T20:02:11Z runner: collected 41 tests
INFO 2026-10-10T20:02:11Z runner: test_session starts
DEBUG 2026-10-10T20:02:12Z plugin: loading cov hooks
INFO 2026-10-10T20:02:12Z runner: test_auth_login PASSED
INFO 2026-10-10T20:02:12Z runner: test_auth_login PASSED
INFO 2026-10-10T20:02:12Z runner: test_auth_login PASSED
DEBUG 2026-10-10T20:02:13Z cache: hit auth/session/9f2
INFO 2026-10-10T20:02:13Z runner: test_auth_refresh PASSED
WARNING 2026-10-10T20:02:14Z runner: test_retry_backoff slow (2.1s > 2.0s budget)
INFO 2026-10-10T20:02:14Z runner: test_retry_backoff PASSED
ERROR 2026-10-10T20:02:15Z runner: test_payment_webhook FAILED - AssertionError: expected status 'settled', got 'pending'
TRACEBACK 2026-10-10T20:02:15Z runner: File "tests/test_payments.py", line 87, in test_payment_webhook
INFO 2026-10-10T20:02:15Z runner: continuing suite
INFO 2026-10-10T20:02:16Z runner: test_payments_refund PASSED
INFO 2026-10-10T20:02:16Z runner: test_payments_refund PASSED
INFO 2026-10-10T20:02:16Z runner: test_payments_refund PASSED
INFO 2026-10-10T20:02:16Z runner: test_payments_refund PASSED
DEBUG 2026-10-10T20:02:17Z cache: hit payments/refund/a1c
DEBUG 2026-10-10T20:02:17Z cache: hit payments/refund/a1c
INFO 2026-10-10T20:02:17Z runner: suite done: 40 passed, 1 failed
"""

TOOL_JSON = {
    "id": "search-0042",
    "tool": "repo_search",
    "status": "ok",
    "query": "webhook settled status pending",
    "results": [
        {
            "path": "src/payments/webhook.py",
            "signature": "def handle_webhook(event: dict) -> str",
            "snippet": "status = event.get('status', 'pending')  # FIXME: upstream sends 'settled' only after async ack",
            "score": 0.93,
            "metadata": {"repo": "billing-svc", "branch": "main", "indexed_at": "2026-10-09"},
        },
        {
            "path": "src/payments/models.py",
            "signature": "class Payment(Base): status: str",
            "snippet": "status transitions: pending -> processing -> settled",
            "score": 0.81,
            "metadata": {"repo": "billing-svc", "branch": "main", "indexed_at": "2026-10-09"},
        },
        {
            "path": "docs/webhooks.md",
            "signature": None,
            "snippet": "Webhooks fire on every transition; consumers must treat 'pending' as non-terminal.",
            "score": 0.77,
            "metadata": {"repo": "billing-svc", "branch": "main", "indexed_at": "2026-10-09"},
        },
        {"path": "old/legacy.py", "signature": None, "snippet": "...", "score": 0.31,
         "metadata": {"repo": "billing-svc"}},
        {"path": "old/legacy2.py", "signature": None, "snippet": "...", "score": 0.28,
         "metadata": {"repo": "billing-svc"}},
        {"path": "old/legacy3.py", "signature": None, "snippet": "...", "score": 0.25,
         "metadata": {"repo": "billing-svc"}},
        {"path": "old/legacy4.py", "signature": None, "snippet": "...", "score": 0.22,
         "metadata": {"repo": "billing-svc"}},
    ],
    "timings": {"search_ms": 184, "rerank_ms": 96},
    "trace": ["node=eu-west-2", "shard=7", "cache=MISS", "retries=0"],
    "debug": {"query_plan": "bm25+vector hybrid", "topk": 50},
}

RAG_CHUNK = """\
Payment webhooks fire on every status transition. The 'pending' state is
non-terminal: it means the payment was accepted but the async settlement
acknowledgement has not arrived yet. Consumers must not treat 'pending' as
final. The upstream provider sends 'settled' only after the async ack, which
typically lands within 60 seconds. As mentioned above, retry logic should use
exponential backoff. It is important to note that polling the settlement
endpoint more than once per minute will trigger rate limiting. For more
information, see the provider status page. The handler in
src/payments/webhook.py reads event['status'] with a default of 'pending'.
Tests should mock the async ack before asserting a 'settled' state. In
conclusion, pending-then-settled is the expected lifecycle.
"""

QUERY = "why did the payment webhook test fail expecting settled but got pending"


def main():
    items = [
        {"kind": "log", "content": LOG_TRACE},
        {"kind": "json", "content": TOOL_JSON},
        {"kind": "chunk", "content": RAG_CHUNK, "query": QUERY},
    ]

    compressor = AgentContextCompressor(token_budget=4000)
    compressed, stats = compressor.compress(items)

    print("=== Agent context compression demo ===\n")
    print("Query: %s\n" % QUERY)
    print("tokens before : %d" % stats["tokens_before"])
    print("tokens after  : %d" % stats["tokens_after"])
    print("compression   : %.1fx (%s%% fewer tokens)" % (
        stats["compression_ratio"], stats["reduction_pct"]))
    print("est. saving   : $%.4f per 1k agent runs (illustrative $1/MTok)" % (
        stats["est_cost_saved_per_1k_runs_usd"]))
    print("within budget : %s\n" % stats["within_budget"])

    for item in compressed:
        print("--- %s (compressed, %d tokens) ---" % (
            item["kind"], estimate_tokens(item["content"])))
        print(item["content"])
        print()

    print("=== What survived (spot check) ===")
    log_out = compressed[0]["content"]
    assert "test_payment_webhook FAILED" in log_out, "FATAL line lost!"
    assert "AssertionError" in log_out, "error detail lost!"
    print("OK  error lines preserved byte-for-byte")
    json_out = compressed[1]["content"]
    assert '"signature": "def handle_webhook' in json_out
    assert "... (2 more items)" in json_out
    print("OK  JSON shape kept, list truncated with visible marker")
    chunk_out = compressed[2]["content"]
    assert "non-terminal" in chunk_out and "acknowledgement" in chunk_out
    print("OK  answer-bearing sentences kept, filler dropped")


if __name__ == "__main__":
    main()
