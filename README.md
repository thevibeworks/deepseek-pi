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

Five tools: `read`, `bash`, `edit`, `write`, `task`. `bash` absorbs search and
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
deepseek-pi -max-cost 2.50       stop the run after $2.50
deepseek-pi -max-turns 40        stop one request after 40 turns
deepseek-pi -sessions            list stored sessions
deepseek-pi -prune               drop checkpoint data no session refers to
deepseek-pi -show-prompt         print the assembled system prompt
```

Slash commands in an interactive session: `/mode`, `/model`, `/effort`,
`/compact`, `/cache`, `/budget`, `/turns`, `/rewind`, `/fork`, `/status`,
`/prompt`, `/session`, `/thinking`, `/clear`, `/exit`.

## Branching

A session that went down a wrong path has two moves. `/rewind` drops the last
turn (or `/rewind N`, any turn) from the conversation; `/fork` copies the
session at that point into a new one and continues there, leaving the branch
you left untouched. `/turns` lists the points you can cut at.

```
$ /turns
  1   what does the parser do?
  2 * fix the off-by-one
  3   now add a benchmark
* changed files or ran commands

$ /rewind 2
rewound to before turn 2 — 1 turn(s) kept, 2 dropped
the conversation went back; the workspace did not:
  turn 2 wrote parser.go
  turn 2 ran  go test ./...
```

Three things make this correct rather than a text-editor undo:

**You can only cut at a turn boundary.** A tool call and its result are a
matched pair the provider rejects if split, so a user message is the only place
the transcript is unambiguously complete. That is the same rule compaction's
`findCutPoint` follows.

**The file is never rewritten.** A rewind appends a marker and `LoadSession`
replays it, so the transcript still records the path you abandoned while the
context the model sees goes back. Storage and view are separate, exactly as
they already are for compaction.

**The workspace goes back too — for the files the agent wrote.** Every `write`
and `edit` is checkpointed at the tool-call seam, so a rewind restores the
files that turn changed and deletes the ones it created. Shell commands are
not checkpointed and cannot be: a command may touch anything, and guessing
would produce a restore that is confidently wrong. What could not be undone is
listed rather than omitted:

```
$ /rewind 2
rewound to before turn 2 — 1 turn(s) kept, 2 dropped, 1 file(s) restored
  removed  extra.go
not undone:
  kept demo.go — it changed outside this conversation since the agent wrote it
  turn 3 ran go generate ./...
```

That `kept` line is the safety property. A restore only proceeds when what is
on disk is byte-for-byte what the agent last wrote there; anything you edited
by hand in the meantime is left alone, because destroying work the
conversation never knew about is worse than not restoring at all. Snapshots
persist in the session file, so `-c` can rewind work from an earlier run.

`/clear` is a rewind to nothing, so a cleared conversation stays cleared across
`-c` — but it deliberately does *not* revert files. Clearing is about the
context; silently undoing a day of accepted edits because you dropped the
conversation would be a shock.

Content is stored content-addressed under the workspace's session directory,
so a repeated edit and a fork cost nothing extra. Nothing collects it
automatically — `deepseek-pi -prune` drops what no session can still rewind to.

Branching is nearly free, because a truncated prefix is still a cached prefix.
Measured against the live API, the turn after a cut read 17792 of 17809 input
tokens from cache; the 17 uncached tokens were the new prompt itself.

## Sub-agents

`task(role, prompt)` delegates self-contained work to a sub-agent with its own
context. The parent receives only the final report — never the child's tool
output — which is the entire point: a broad search costs the parent one report
instead of forty file reads.

| role | can modify | model |
|---|---|---|
| `explorer` | no | flash |
| `tester` | no | flash |
| `reviewer` | no | **pro** — judgment is the one job where the cheap model is a false economy |
| `implementer` | yes | flash |

Sub-agents run **concurrently** when the model batches them, get their own
transcript on disk, and are bounded by a budget (30 turns, 200k tokens, 5
minutes). A child that hits one dies quietly and the parent still gets a
partial report, clearly marked as partial — a truncated report that reads as
complete is worse than none, because the parent acts on it.

Two safety properties are structural rather than configured. Recursion is
impossible because a child's tool set simply does not contain `task` — no depth
counter to tune or get wrong. And a child is bounded by the **stricter** of its
role and its parent, so an `implementer` under a `-plan` parent still cannot
write; a mode another setting can escape is not a mode.

## Budgets

Sub-agents have always been bounded; the session itself was not. `-max-cost`
and `-max-turns` bound it, and `/budget` shows or changes them mid-session.

```
$ /budget
cost   $0.0012 of $2.5000 (0% used)
turns  40 per request

$ /budget cost 5
```

The two axes have **different scopes on purpose**. `-max-cost` is cumulative
over the run, because money only accumulates and the question it answers is how
much you are willing to spend today. `-max-turns` bounds one request, because a
session that has answered forty questions is working while one request that took
forty turns is looping — making it cumulative would stop a productive session at
a point cost already covers better, and would say nothing about loops.

**Sub-agent spend counts.** A loop that delegates is still a loop. Measured on a
single delegating turn: parent $0.0002, child $0.0010 — a budget watching only
the parent would have missed 83% of the money.

Both stop at a turn boundary, the same seam compaction uses, because mid-turn
the transcript holds an assistant message whose tool calls are unanswered and a
provider rejects that outright. So a stop can overshoot by at most one turn.
That is the price of never producing a broken transcript.

A stop is not a wall. Interactively it prints what fired and how to raise it,
and raising it clears the stop so the next prompt continues. Under `-p` there is
nobody to ask, so the run exits non-zero — a script piping the answer somewhere
must not treat a truncated one as complete. The default is no budget: a limit
nobody asked for that silently truncates a long task is worse than none.

Being over budget also refuses to *start* a request, rather than buying one more
turn to discover it is over.

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

**The cache is watched, not assumed.** Every arrangement above exists to
protect the prefix cache, which used to be an assumption. `/cache` makes it an
observation: the prompt on turn N is the turn N-1 prompt plus what the turn
appended, so the *entire* previous prompt should come back as `cache_read`.
When it does not, the tracker names the first component that differs — system
prompt, tool schemas, or a specific message index — because a cache breaks at
the first differing byte and everything after it is a consequence, not a cause.
Compaction declares its break in advance, so the one prefix break we choose
reads as chosen rather than as a defect.

```
$ /cache
no prompt-cache breaks recorded
```

**Compaction is the one sanctioned prefix break.** A long session eventually
exceeds the window. Two things happen before the model is asked for anything:
reclaim, which costs nothing, then summarization, which costs one flash call.

Reclaim replaces a file read that a *later* read of the same file supersedes,
leaving a stub rather than a hole so the model understands why it is gone.
Paged reads are exempt — two windows of one file are not interchangeable.

Summarization cuts the transcript at a boundary where a real user message
begins. That constraint is not stylistic: a `tool_result` whose `tool_use` was
cut away is rejected by the API, and rejected requests are how a compaction bug
becomes a dead session. The retained tail keeps more than its budget rather
than cut unsafely.

Two details make it terminate and survive. Usage on retained messages is
cleared, because provider usage describes the *pre-compaction* prompt and
leaving it means the size estimate never falls, so every subsequent turn
compacts again forever. And a deterministic summary — files touched, commands
run, requests made, reconstructed from the transcript alone — is always
available, because compaction runs exactly when the context is fullest and a
request is most likely to fail.

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
- Compaction runs only from the between-turns seam. Mid-turn the transcript
  holds an assistant message whose tool calls are still unanswered, and
  rewriting there orphans them.
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

## Evals

Engine changes are supposed to be gated on numbers rather than argument, so
there is a harness for producing them.

```
make eval          # run the suite
make eval-gate     # run it and compare against the committed baseline
make eval-baseline # record a new baseline from a known-good tree
```

A task is a directory: a workspace to copy, a prompt, and a command that must
exit zero. The verifier is a real command on purpose — "did the tests pass" is
a fact, "does the diff look right" is an opinion, and an eval built on opinions
cannot gate anything. Verifiers restore their own canonical test files from
`$EVAL_TASK_DIR` before checking, so a task cannot be passed by deleting the
test.

The gate calibrates itself to noise, and the noise is much larger than it
looks. Two consecutive runs of an *unchanged* agent differed by 24% on input
tokens. Calibrating a tolerance from that measurement then failed on the next
run, which showed a **101-133% within-task spread** — the same task finishing in
5 turns or in 15, depending on nothing.

That is turn-count variance, and it is heavy-tailed, so:

- A single sample per task cannot gate efficiency. `-repeat 5` compares medians.
- A task counts as passed only if **every** repeat passed. Flaky is not passing.
- The tolerance is the largest spread either run revealed, floored at 25% —
  calibrating from the baseline alone leaves the gate most confident exactly
  when it should not be, which is when the *current* run is the one that
  wandered.
- Correctness has no tolerance at all.

The practical consequence: this suite can catch a correctness regression or a
gross efficiency regression, and it cannot resolve a 20% token improvement. If
you want that resolution, raise `-repeat` — more samples, not a tighter number.

Current baseline (flash, 5 repeats, medians):

```
3/3 passed · 41701 in / 3042 out · cache 83% · $0.0019 · 29s
widest input-token spread within a task: 44%
```

A worked example of the limit. Adding the `task` tool moved the suite's input
tokens by +15.7%, comfortably inside the noise floor, so the gate passed and
told us nothing about efficiency. The structural cost of that change is not a
question the eval can answer — but it is deterministic, so measure it directly:

```
$ deepseek-pi -no-skills -show-prompt | wc -c              # 2206
$ deepseek-pi -no-skills -no-subagents -show-prompt | wc -c # 1824
```

95 tokens per turn, 0.3% of the prompt. The +15.7% was noise. Reach for the
eval to answer "did this still work"; reach for a deterministic measurement to
answer "what did this cost".

## Development

```
make test          # unit tests, no network
make test-race     # race detector
make test-live     # hits the real API; needs DEEPSEEK_API_KEY, costs a few cents
make lint          # golangci-lint
make build         # ./bin/deepseek-pi
```

Live tests need `DEEPSEEK_PI_LIVE=1` as well as a key. Exporting a key is
normal; a plain `go test ./...` quietly spending money is not.

The only runtime dependency is `golang.org/x/text`, for Unicode normalization
in the edit tool's fuzzy matching. Everything else is standard library, and it
should stay that way.

## Status

Early. The loop, tools, permissions, compaction, sessions, accounting and the
eval harness, cache-break attribution, sub-agents, session branching, workspace
checkpointing and session budgets work and are tested against the live API. Not
yet built: readline-style input (a pasted block is still one turn per line), a
scheduler and a full-screen TUI.

## License

MIT. See [LICENSE](LICENSE).

The Pi harness (MIT, Mario Zechner / earendil-works) is the architectural
source for the loop contracts, tool-output discipline and prompt-assembly
invariants ported here. This is an independent Go implementation, not a
translation.
