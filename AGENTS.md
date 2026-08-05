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

**A stream function never returns an error.** Failures are a final assistant
message with `StopError`. This is what keeps the loop to one failure path.

**Tools return an error to fail.** Do not encode failure inside a successful
result; the model reacts to the error flag and routinely misses failure text
buried in output.

## Changing the loop

`agent/loop.go` implements contracts documented in the README and enforced in
`agent/loop_test.go`. Before changing behaviour there, read the tests — each one
names the failure it prevents. If a change makes a contract test fail, the
contract is the thing to argue with first, not the test.

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
