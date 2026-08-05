# deepseek-pi

A coding agent for the DeepSeek v4 series, in Go. Single binary, zero runtime
services, one dependency.

The architecture is borrowed from the [Pi agent harness][pi] — its loop
contracts, tool-output discipline, and prompt-cache invariants are the parts of
a coding agent that are expensive to rediscover. The specialization is
DeepSeek-only, which is what lets this be a few thousand lines instead of tens
of thousands: one wire protocol, one model family, no provider matrix.

[pi]: https://github.com/earendil-works/pi

```
go install github.com/thevibeworks/deepseek-pi/cmd/deepseek-pi@latest
export DEEPSEEK_API_KEY=sk-...
deepseek-pi                 # interactive; asks before it changes anything
```

## What it does

```
$ deepseek-pi --yolo -p "buggy.go prints 9 but should print 10. Fix it and verify."
read /tmp/demo/buggy.go
  ok 1  package main (+15 more lines)
The bug: the loop starts at index 1 instead of 0, skipping the first element.
edit /tmp/demo/buggy.go (1 edit(s))
  ok Applied 1 edit(s) to buggy.go (1 line(s) changed).
bash cd /tmp/demo && go run buggy.go
  ok 10
Fixed. The loop started at index 1, skipping the first element, so it summed
2+3+4=9 instead of 1+2+3+4=10. Running the program now prints 10.
41113 in / 344 out - cache 30464 (74%) - $0.0017 - saved $0.0042
```

Four tools: `read`, `bash`, `edit`, `write`. `bash` absorbs search and
navigation (`rg`, `fd`, `ls`, `sed`) rather than shipping a tool per verb —
every schema lives in the cached prompt prefix, so a 4-schema tool set is
materially cheaper per turn than a 40-schema one, and the model already knows
shell better than it knows any bespoke search schema.

## Usage

```
deepseek-pi                      interactive session
deepseek-pi -p "prompt"          run once and exit
deepseek-pi "prompt"             same, positional
deepseek-pi -c                   continue the most recent session here
deepseek-pi -m pro -e high ...   pro model, high reasoning effort
deepseek-pi -plan                read-only: investigate and propose
deepseek-pi -sessions            list stored sessions
deepseek-pi -show-prompt         print the assembled system prompt
```

Slash commands in an interactive session: `/mode`, `/model`, `/effort`,
`/status`, `/prompt`, `/session`, `/thinking`, `/clear`, `/exit`.

## Permissions

Reading and read-only shell always run. Anything that can change the machine
asks first:

```
edit wants to run:
  main.go (1 change(s))
    1. for i := 1; i < len(nums) -> for i := 0; i < len(nums)
[y] once  [a] always for edit  [n] no >
```

Three modes:

| mode | what runs |
|---|---|
| `default` | reads and read-only shell freely; asks before edit, write, or shell that can mutate |
| `-plan` | read-only. Mutating tools are refused outright, with a message telling the model to describe the change instead |
| `-yolo` | everything, no questions. For sandboxes, disposable checkouts, and CI that has accepted the blast radius |

`--allow edit,write` (repeatable, or comma-separated) pre-approves specific
tools. Plan mode ignores both `--allow` and remembered approvals — a mode that
another setting can quietly override is not a mode.

A headless run (`-p`) has nobody to ask, so **it refuses instead of assuming
consent** and tells the model to say so. Pass `-yolo` or `--allow` to opt in.
Reading and searching still work unprompted, so `deepseek-pi -p "what does this
package do"` needs no flags at all.

Whether shell is "read-only" is decided by a deliberately conservative
classifier: the command is split on every chaining operator, each segment must
lead with a program on a short allowlist, and any redirection, substitution,
backgrounding, inline assignment, or path-qualified program disqualifies the
whole command. Unknown means unsafe. `rg foo | head` runs; `rg foo > out` asks.
Its failure mode should be an unnecessary question, never an unintended write —
`TestBashClassifierRejectsBypasses` is the adversarial pass over that.

Skills are progressive-disclosure: only a one-line index sits in the cached
prefix, and a body is read on the turn it is needed. A large personal
collection can still dominate the prompt — `/status` reports what the index
costs, and `-no-skills` turns discovery off.

Configuration is environment only — `DEEPSEEK_API_KEY` (required),
`DEEPSEEK_PI_HOME` (session storage, default `~/.deepseek-pi`), `NO_COLOR`.
There is no config file, because no option here has had a user who needed one.

Project instructions come from `DEEPSEEK.md`, `AGENTS.md` or `CLAUDE.md`,
discovered from the repository root down to the working directory. Skills are
`SKILL.md` files under `.agents/skills/` or `.deepseek-pi/skills/`, project
first, then `~`.

## Why these decisions

**One protocol.** DeepSeek v4 emits DSML, an Anthropic-shaped tool grammar, so
the Anthropic-compatible Messages endpoint is the shape the model is trained
toward. It carries tool use, parallel tool calls, thinking blocks and
cache-read reporting correctly. The OpenAI-style route brings extra hazards
(DSML envelope leakage into visible content, `reasoning_content` replay rules,
`tool_choice`/thinking conflicts) for no demonstrated benefit. One state
machine to keep correct beats two.

**Real numbers, not advertised ones.** v4 advertises a 1M context window; about
616k of it is usable input. All budget math uses 616k. Cost is computed from
provider-reported token counts against our own rate card — never from a
provider or harness cost field. Retargeted Claude harnesses misprice DeepSeek
by roughly 13x because they apply Claude rate cards to DeepSeek tokens.

**The prompt never drifts.** Cached input runs 0.0028/M against 0.14/M for a
miss — a 50x swing — and the cache re-bills the entire prompt when one byte
before the change point differs. So the system prompt carries no date, no
time, no counters; tool docs are assembled deterministically from the tools
themselves; skills sort by name; and the working directory goes on the last
line because it is the only per-machine value. A byte-comparison test enforces
this in CI.

**A transcript is pinned to one model.** Nothing verifies that thinking blocks
generated by flash replay correctly under pro, so `/model` starts a fresh
context rather than pretending. That single rule deletes an entire cross-model
replay layer (thinking-to-text rewrites, ID normalization) from the codebase.

**Spill, don't truncate.** Oversized command output goes to a temp file whose
path the model is told, decided *during* streaming rather than after — by the
time a command finishes, the bytes you would have wanted are already gone.

## Loop contracts

These are ported from pi deliberately and in full. They are what separates an
agent loop that works from one that mostly works:

- A stream function never fails. Request, transport and runtime failures arrive
  as a final assistant message with `StopError`, so the loop has one failure
  path instead of two.
- A response truncated by the output token limit fails its **entire** tool
  batch. Streamed arguments are finalized with a best-effort salvage parse, so
  a truncated call can parse *and* validate while being silently incomplete.
  None are safe to run.
- Tool end-events fire in **completion** order; tool result messages are
  appended in the model's **source** order. Different consumers, different
  contracts.
- One tool declaring sequential execution forces the whole batch sequential —
  the shared state it protects is not confined to itself.
- Early termination requires **every** result in a batch to request it. One
  tool cannot end a run it lacks the full picture for.
- A crashed turn is scrubbed from the next provider payload, and every orphaned
  tool call gets a synthesized result. Replaying a crashed turn poisons the
  conversation.
- Everything that happens between turns goes through one named seam
  (`PrepareNextTurn` / `ShouldStopAfterTurn`) rather than being inlined in the
  loop body.
- Two retry layers, and only one is on by default. The transport retries
  nothing; the session layer classifies the failure, pops the failed turn, and
  replays. Retrying in both places multiplies attempts and hides real failures.

## Packages

| Package   | Role | Pi analogue |
|---|---|---|
| `ai`      | wire protocol, streaming, model catalog, cost | `@earendil-works/pi-ai` |
| `agent`   | loop, tool runtime, queues, session retry | `@earendil-works/pi-agent-core` |
| `tools`   | read, bash, edit, write + output bounding | (part of pi-coding-agent) |
| `harness` | prompt assembly, project context, skills, sessions | `@earendil-works/pi-coding-agent` |

`ai` and `agent` are usable on their own if you want to build something else on
DeepSeek:

```go
client := ai.NewClient(os.Getenv("DEEPSEEK_API_KEY"))
a := agent.New(
    &agent.Context{SystemPrompt: "You are helpful.", Tools: myTools},
    agent.LoopConfig{Model: ai.ModelFlash, Effort: ai.EffortHigh},
    client.StreamFunc(ctx),
)
msgs, err := a.Prompt(ctx, "do the thing")
```

## Development

```
make test          # unit tests
make test-race     # race detector
make test-live     # hits the real API, needs DEEPSEEK_API_KEY, costs a cent
make lint          # golangci-lint
make build         # ./bin/deepseek-pi
```

The only runtime dependency is `golang.org/x/text`, for Unicode normalization
in the edit tool's fuzzy matching. Everything else is standard library, and it
should stay that way.

## Status

Early. The loop, tools, permissions, sessions and accounting work and are
tested against the live API. Not yet built: context compaction (a session
currently ends at the context limit), sub-agents, an eval harness, a scheduler,
and a full-screen TUI.

## License

MIT. See [LICENSE](LICENSE).

The Pi harness (MIT, Mario Zechner / earendil-works) is the architectural
source for the loop contracts, tool-output discipline and prompt-assembly
invariants ported here. This is an independent Go implementation, not a
translation.
