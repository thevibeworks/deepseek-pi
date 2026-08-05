# deepseek-pi

A coding agent for the DeepSeek v4 series. Go, standard library plus
`golang.org/x/text`.

## Layout

- `ai/` — wire protocol, streaming, model catalog, cost. No knowledge of tools.
- `agent/` — the loop, tool runtime, queues, session retry. No knowledge of the
  filesystem or of any specific tool.
- `tools/` — read, bash, edit, write, and the output-bounding primitives.
- `harness/` — prompt assembly, project context, skills, session persistence.
- `cmd/deepseek-pi/` — CLI and rendering.

Dependencies point one way: `harness` -> `tools` -> `agent` -> `ai`. Keep it
that way; a cycle here means an abstraction is in the wrong package.

## Rules that are not negotiable

**The system prompt must be byte-stable across turns.** No dates, no times, no
counters, no map iteration without sorting. Cached input is 0.0028/M against
0.14/M for a miss, and one differing byte re-bills the whole prefix.
`TestPrefixStabilityAcrossTurns` guards this and runs in CI.

**Cost is computed from token counts, never read from a field.** Provider and
harness cost fields are somebody else's arithmetic; token counts are facts.

**Budget against 616k, not 1M.** The advertised context window is not the
usable one.

**Tool documentation lives with the tool.** `PromptSnippet` and
`PromptGuidelines` on the tool definition; the prompt is assembled from them.
Never hand-maintain a tool list in `harness/prompt.go`.

**Compaction must terminate.** Clear provider usage from any message that
survives compaction. `Usage.Input` describes the prompt *before* the head was
dropped, and `EstimateTokens` anchors on the newest such figure — leave it and
the estimate never falls, so every following turn compacts again while the
transcript stops shrinking. `TestCompactionTerminates` guards this. Resume has
the same hazard and the same fix.

**Never cut a transcript where a tool result would lose its call.** The API
rejects a `tool_result` without a preceding `tool_use` outright. Cut only where
a real user message begins; retaining more than the budget is fine, retaining
an orphan is a dead session. Compaction (`findCutPoint`) and branching
(`harness.Turns`) are the two places that cut, and they follow the same rule.

**The session file is append-only.** Compaction and rewind shrink the context
*view* and never storage: both append a marker that `LoadSession` replays.
Deleting lines would destroy the record of what was tried, which is the only
thing a transcript is for.

**A stream function never returns an error.** Failures are a final assistant
message with `StopError`. This is what keeps the loop to one failure path.

**Tools return an error to fail.** Do not encode failure inside a successful
result; the model reacts to the error flag and routinely misses failure text
buried in output.

## Changing permissions

`harness/permission.go` gates every tool call. Two things are load-bearing:

- The order inside `Policy.Decide`. Plan mode is checked before the allow-list
  and before remembered approvals. A mode another setting can override is not a
  mode.
- `classifyBash` fails toward asking. If you add to `safeCommands`, the
  question is not "is this usually safe" but "can this write, execute
  arbitrary text, or be chained into something that can". Add an adversarial
  case to `TestBashClassifierRejectsBypasses` for anything you are unsure of.

New tools that can change the machine must be added to `mutatingTools` in the
same commit. Nothing else classifies them.

## Changing the loop

`agent/loop.go` implements contracts documented in the README and enforced in
`agent/loop_test.go`. Before changing behaviour there, read the tests — each one
names the failure it prevents. If a change makes a contract test fail, the
contract is the thing to argue with first, not the test.

## Branching

`harness/branch.go` turns a transcript into addressable branch points. A turn is
one user message and everything through to the next; nothing else is a legal
cut. Two rules to keep in mind when touching it:

Route session writes through `h.session()`, never a captured `*Session`. `Fork`
swaps the session under the agent's event goroutine, and a callback holding the
old pointer appends to a closed file — silently, since the error only surfaces
as a warning.

Report what a discarded turn changed on disk. `Turn.Changed` reads the tool
*calls*, because a call records what was attempted even when the result errored,
and shell safety comes from `classifyBash` so a turn that only ran `rg` is not
flagged. A rewind looks like undo, and a user who believes the files went back
too will act on a workspace that does not match the conversation.

Branching does not break the prefix cache and must not start to: the retained
prefix is byte-identical, and the live test asserts no cache break is recorded
across a rewind or a fork.

## The prompt cache

`/cache` reports why the prefix cache missed and what it cost. It should say
"no prompt-cache breaks recorded" for an ordinary session — a real six-turn run
holds 99%.

If you are about to change the system prompt, the tool set, or how history is
serialized, run a few turns and check `/cache` afterwards. An unsanctioned
break means the prefix drifted, and on a long session that is a 50x price
difference on every turn that follows.

Anything that rewrites history on purpose must call `CacheTracker.ExpectBreak`
first, or it reports as a defect and the report starts crying wolf.

## Evals

`make eval-gate` before anything that touches the loop, the prompt, the tools
or the context engine. It costs a few cents.

Do not tighten `MinTolerance` because a number looks loose. It is 25% because
the measured spread between two runs of an unchanged agent was 24-42%, and a
gate that fires on noise gets ignored. If you want a tighter gate, raise
`-repeat` — more samples, not a smaller tolerance.

Only record a new baseline from a tree you believe is good, and say in the
commit message why the numbers moved.

## Before committing

```
make check      # fmt, vet, test
make test-race  # the loop runs tools in parallel; races here are real
```

`make test-live` hits the real API and costs about a cent. Run it when
touching `ai/`.

## Style

Comments explain **why**, not what. If a line needs a comment to say what it
does, rename something instead. Prefer the smallest change that fully solves
the problem, and leave unrelated code alone.
