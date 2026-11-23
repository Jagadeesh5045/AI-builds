# Agent-setup scanner: audit the config your agent loads, not just your code

**Trending (22-23 Nov 2026):** safe-anything appeared in the 23 Nov
"Weights & Takes" roundup of the last 24 hours: a scanner that audits your
*agent setup* rather than your source code, looking for permission bypasses,
unpinned MCP servers, plaintext MCP tokens, curl-to-shell hooks and hidden
prompt-injection text, across the config formats used by Claude Code, Codex,
Cursor, Gemini and OpenCode. The same edition led with Z.ai's GLM-5.4, its
first looped transformer; looped blocks were covered here on 21 Nov, so
today's build takes the scanner instead.

## Why this matters

An agent reads its config with the user's full trust and often with shell
access. A setup file is therefore an attack surface of its own:

- A permission bypass turns every later mistake into an unreviewed action.
- An unpinned MCP server means the code that runs tomorrow is whatever the
  registry serves tomorrow, not what was reviewed today.
- A plaintext token in a config file is one leaked dotfiles repo away from
  a compromised account.
- A curl-pipe-shell hook executes remote code at session start.
- Invisible unicode or an instruction override buried in a shared AGENTS.md
  can steer the agent without a human ever seeing the text.

## What this code does

`scanner.py` is a stdlib-only miniature of that audit. It walks a directory
of agent configuration (JSON, YAML, Markdown, TOML, shell) and applies six
explicit rules:

| Rule | Severity | What it catches |
| --- | --- | --- |
| PERMISSION_BYPASS | critical | bypass permission modes, dangerously-skip-permissions flags, auto-approve |
| PLAINTEXT_SECRET | critical | literal credentials in config; the report shows key name and length only, never the value |
| UNPINNED_MCP | high | npx/uvx packages with no version pin (@latest does not count), docker images with no tag or digest |
| CURL_PIPE_SHELL | high | curl or wget piped straight into sh, bash or zsh |
| HIDDEN_INJECTION | high | zero-width and bidi-control unicode, instruction-override phrases |
| WILDCARD_TOOLS | medium | allowlists granting "*" or "Bash(*)" |

Two implementation details found by the tests: JSON formatting splits
arrays across lines, so allowlists and MCP launch commands are matched on
the whole file text with line numbers recovered from match offsets; and
secrets are redacted at finding-creation time, so no code path can print
one later.

## Demo results (verified locally, 23 Nov 2026)

- Vulnerable fixture (bypass mode, wildcard allowlist, curl-pipe hook,
  unpinned filesystem server, unpinned docker image, @latest notes server,
  literal token, zero-width character plus an override comment in
  AGENTS.md): **9 findings, all six rules, verdict FAIL**.
- Clean fixture (default permissions, scoped allowlist, pinned versions,
  env-reference token, local check script): **0 findings, verdict PASS**.
- Tests: **25/25 checks passing**, covering the pinning parser (scoped
  names, @latest), each rule firing and not firing, secret redaction, and
  both fixtures end to end.

## Run it

```bash
python3 scanner.py            # demo on the built-in fixtures
python3 scanner.py ~/.config/agent   # scan a real directory (JSON report)
python3 test_scanner.py       # 25 checks
```

Exit code is 1 when any critical or high finding exists, else 0, so the
scanner can gate a dotfiles commit or a CI step.

## Honest limits

This is a text scanner, not a parser: it does not evaluate JSON logic,
follow symlinks out of the tree, or judge whether a pinned version is a
*good* version. It also cannot see instructions hidden in images or in
files with extensions outside its scan list. Treat a PASS as "none of
these six known-bad patterns", not as a proof that a setup is safe.
