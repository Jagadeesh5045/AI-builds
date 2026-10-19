"""Agent ticket-router: System One decisions vs the old chat-parse-JSON path.

Spins up server.py in-process, then routes three synthetic support tickets
two ways:

NEW WAY (/v1/systemone): one request asks three typed questions -- a choice
(department), a noul (page a human?) and a score (urgency). The agent branches
in code on the returned values and probabilities. Nothing is generated, so
output_tokens is 0 and there is nothing to parse.

OLD WAY (/v1/chat/completions): "reply with ONLY a JSON object". The stub
chat server buries the decision in varied prose; the client tries
json.loads, then falls back to a regex. Watch what happens on ticket 3.

Standard library only. Run:  python3 agent_demo.py
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.request

import server as srv

PORT = 8765

TICKETS = [
    "Production Postgres is down since 02:00 and checkout is failing for every "
    "EU customer. No workaround.",
    "I was charged twice for my October invoice. Please refund the duplicate charge.",
    "Quick question: how do I reset my password? The reset email never arrives.",
]

QUESTIONS = {
    "dept": {
        "type": "choice",
        "instructions": "Which team should own this ticket?",
        "criteria": {
            "engineering": "code, bugs, outages, infrastructure, servers, databases, production incidents",
            "billing": "invoices, charges, refunds, payments, pricing, duplicates",
            "account": "login, password, signup, profile, access, reset email",
        },
    },
    "needs_human": {
        "type": "noul",
        "instructions": "Does this ticket describe a production outage affecting customers?",
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this ticket?",
        "levels": ["low", "medium", "high"],
        "criteria": {
            "low": "routine question, no deadline, workaround exists",
            "medium": "affects one customer, money involved, needs action today",
            "high": "production down, many customers affected, no workaround, money at risk",
        },
    },
}


def _post(path: str, payload: dict):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = json.loads(resp.read().decode())
    return (time.perf_counter() - t0) * 1000.0, body


def new_way(ticket: str) -> dict:
    ms, resp = _post("/v1/systemone", {
        "model": "proxy-systemone-1", "state": ticket, "questions": QUESTIONS})
    a = resp["answers"]
    dept, conf = a["dept"]["choice"], a["dept"]["confidence"]
    page_human = a["needs_human"]["noul"] > 0.65
    score = a["urgency"]["score"]
    escalate = conf < 0.6  # uncertain routing -> human review, policy lives in code
    priority = "p1" if score > 1.5 else "p2" if score > 0.7 else "p3"
    action = "PAGE_ONCALL" if page_human else ("HUMAN_REVIEW" if escalate else f"QUEUE:{dept}:{priority}")
    return {"ms": ms, "route": dept, "confidence": conf,
            "page_human": page_human, "urgency_score": score,
            "action": action, "output_tokens": resp["usage"]["output_tokens"]}


def old_way(ticket: str) -> dict:
    prompt = (
        "Classify the following support ticket. Reply with ONLY a JSON object "
        'like {"department": "...", "urgency": "low|medium|high"}.\nTicket: ' + ticket
    )
    ms, resp = _post("/v1/chat/completions", {
        "model": "proxy-systemone-1",
        "messages": [{"role": "user", "content": prompt}]})
    content = resp["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(content)  # the "ONLY a JSON object" hope
        return {"ms": ms, "parsed": parsed, "method": "json.loads", "ok": True,
                "output_tokens": resp["usage"]["completion_tokens"]}
    except ValueError:
        pass
    m = re.search(r'"department"\s*:\s*"(\w+)"', content)
    if m:
        return {"ms": ms, "parsed": {"department": m.group(1), "urgency": "?"},
                "method": "regex salvage", "ok": True,
                "output_tokens": resp["usage"]["completion_tokens"]}
    return {"ms": ms, "parsed": None, "method": "FAILED TO PARSE", "ok": False,
            "output_tokens": resp["usage"]["completion_tokens"]}


def main():
    httpd = srv.create_server(PORT)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.2)

    print("=" * 78)
    print("NEW WAY: POST /v1/systemone  (typed decisions, nothing generated)")
    print("=" * 78)
    for i, ticket in enumerate(TICKETS, 1):
        r = new_way(ticket)
        print(f"\nTicket {i}: {ticket[:60]}...")
        print(f"  dept={r['route']} (confidence {r['confidence']:.3f})  "
              f"page_human={r['page_human']}  urgency_score={r['urgency_score']:.2f}")
        print(f"  -> action: {r['action']}   ({r['ms']:.1f} ms, "
              f"output_tokens={r['output_tokens']})")

    print("\n" + "=" * 78)
    print("OLD WAY: POST /v1/chat/completions  (parse the prose, hope the JSON holds)")
    print("=" * 78)
    for i, ticket in enumerate(TICKETS, 1):
        r = old_way(ticket)
        status = "OK " if r["ok"] else "FAIL"
        print(f"\nTicket {i} [{status}] via {r['method']}: {r['parsed']}   "
              f"({r['ms']:.1f} ms, output_tokens={r['output_tokens']})")

    print("\nTakeaway: the System One path returns values code can branch on -- "
          "choice, probability, confidence -- with zero output tokens and no "
          "parsing. The chat path needed two parse strategies and still "
          "failed outright on one ticket. (Timings are local-stub, illustrative.)")
    httpd.shutdown()


if __name__ == "__main__":
    main()
