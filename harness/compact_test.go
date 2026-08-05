package harness

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/thevibeworks/deepseek-pi/agent"
	"github.com/thevibeworks/deepseek-pi/ai"
)

func readCall(id, path string) ai.Content {
	args, _ := json.Marshal(map[string]any{"path": path})
	return ai.Content{Type: ai.ContentToolCall, ID: id, Name: "read", Arguments: args}
}

func pagedReadCall(id, path string, offset int) ai.Content {
	args, _ := json.Marshal(map[string]any{"path": path, "offset": offset})
	return ai.Content{Type: ai.ContentToolCall, ID: id, Name: "read", Arguments: args}
}

func readResult(id, body string) ai.Message {
	return ai.Message{
		Role: ai.RoleToolResult, ToolCallID: id, ToolName: "read",
		Content: []ai.Content{ai.TextContent(body)},
	}
}

func assistantWith(content ...ai.Content) ai.Message {
	return ai.Message{Role: ai.RoleAssistant, StopReason: ai.StopToolUse, Content: content}
}

// assertNoOrphanToolResults is the correctness property compaction must never
// break. A tool_result whose tool_use id does not appear earlier is not a
// degraded transcript — the provider rejects the request outright.
func assertNoOrphanToolResults(t *testing.T, msgs []ai.Message) {
	t.Helper()
	seen := map[string]bool{}
	for _, m := range msgs {
		if m.Role == ai.RoleAssistant {
			for _, c := range m.Content {
				if c.Type == ai.ContentToolCall {
					seen[c.ID] = true
				}
			}
		}
		if m.Role == ai.RoleToolResult && !seen[m.ToolCallID] {
			t.Fatalf("orphaned tool result %q: its tool call was cut away", m.ToolCallID)
		}
	}
}

func TestEstimateTokensAnchorsOnProviderUsage(t *testing.T) {
	// Provider usage is truth; only what comes after it is estimated. Without
	// the anchor, error compounds every turn.
	msgs := []ai.Message{
		ai.UserMessage(strings.Repeat("x", 4000)),
		{
			Role: ai.RoleAssistant, StopReason: ai.StopEnd,
			Content: []ai.Content{ai.TextContent("ok")},
			Usage:   ai.Usage{Input: 100_000, Output: 500},
		},
	}
	if got := EstimateTokens(msgs); got != 100_500 {
		t.Errorf("EstimateTokens = %d, want 100500 (anchor only)", got)
	}

	// Appending after the anchor adds an estimate, not a re-count.
	msgs = append(msgs, ai.UserMessage(strings.Repeat("y", 4000)))
	got := EstimateTokens(msgs)
	if got <= 100_500 || got > 102_000 {
		t.Errorf("EstimateTokens = %d, want ~101500 (anchor + tail estimate)", got)
	}
}

func TestFindCutPointNeverOrphansToolResults(t *testing.T) {
	// Build a transcript whose natural cut lands in the middle of a tool batch.
	big := strings.Repeat("z", 8000)
	msgs := []ai.Message{
		ai.UserMessage("first request"),
		assistantWith(readCall("c1", "a.go"), readCall("c2", "b.go")),
		readResult("c1", big),
		readResult("c2", big),
		ai.UserMessage("second request"),
		assistantWith(readCall("c3", "c.go")),
		readResult("c3", big),
		ai.UserMessage("third request"),
		assistantWith(readCall("c4", "d.go")),
		readResult("c4", big),
	}

	// Sweep every plausible retain budget; each must produce a safe cut.
	for retain := 100; retain < 12_000; retain += 137 {
		cut := findCutPoint(msgs, retain)
		if cut == 0 {
			continue // "no safe cut" is a legitimate answer
		}
		if msgs[cut].Role != ai.RoleUser {
			t.Fatalf("retain=%d cut at index %d (role %s), which is not a clean boundary",
				retain, cut, msgs[cut].Role)
		}
		assertNoOrphanToolResults(t, msgs[cut:])
	}
}

func TestSummarizeProducesValidTranscript(t *testing.T) {
	big := strings.Repeat("q", 6000)
	msgs := []ai.Message{
		ai.UserMessage("do the first thing"),
		assistantWith(readCall("c1", "a.go")),
		readResult("c1", big),
		ai.UserMessage("do the second thing"),
		assistantWith(readCall("c2", "b.go")),
		readResult("c2", big),
		ai.UserMessage("do the third thing"),
		assistantWith(readCall("c3", "c.go")),
		readResult("c3", big),
	}

	c := &Compactor{
		Model: ai.MustLookup(ai.ModelFlash), RetainTail: 1000,
		// No Stream: exercises the deterministic fallback, which is the path
		// that must work when the context is too full for a model call.
	}
	out, usedLLM, err := c.summarize(t.Context(), msgs)
	if err != nil {
		t.Fatalf("summarize: %v", err)
	}
	if usedLLM {
		t.Error("no stream was configured, so no model call should have happened")
	}
	if out == nil {
		t.Fatal("summarize returned nothing")
	}
	if len(out) >= len(msgs) {
		t.Errorf("compaction did not shrink the transcript: %d -> %d", len(msgs), len(out))
	}
	if out[0].Role != ai.RoleUser || !strings.Contains(out[0].Text(), "summarized") {
		t.Errorf("first message should be the summary, got %+v", out[0])
	}
	assertNoOrphanToolResults(t, out)
}

func TestReclaimStubsSupersededReads(t *testing.T) {
	big := strings.Repeat("k", 4000)
	msgs := []ai.Message{
		ai.UserMessage("go"),
		assistantWith(readCall("c1", "same.go")),
		readResult("c1", big),
		ai.UserMessage("again"),
		assistantWith(readCall("c2", "same.go")),
		readResult("c2", big),
		assistantWith(readCall("c3", "other.go")),
		readResult("c3", big),
	}

	out, reclaimed := Reclaim(msgs)
	if reclaimed < 3000 {
		t.Errorf("reclaimed %d bytes, expected the superseded read to be cleared", reclaimed)
	}

	// The earlier read of same.go is stubbed...
	if len(out[2].Text()) > 500 || !strings.Contains(out[2].Text(), "read again later") {
		t.Errorf("earlier read not stubbed: %q", firstLine(out[2].Text(), 120))
	}
	// ...the latest read of same.go survives intact...
	if len(out[5].Text()) != len(big) {
		t.Error("the most recent read of a file must survive")
	}
	// ...and an unrelated file is untouched.
	if len(out[7].Text()) != len(big) {
		t.Error("a read of a different file was cleared")
	}
	// Reclaim must not mutate its input.
	if len(msgs[2].Text()) != len(big) {
		t.Error("Reclaim mutated the caller's slice")
	}
}

func TestReclaimSkipsPagedReads(t *testing.T) {
	// Two windows of one file are not interchangeable; treating them as
	// superseding each other would silently discard content.
	big := strings.Repeat("k", 4000)
	msgs := []ai.Message{
		ai.UserMessage("go"),
		assistantWith(pagedReadCall("c1", "big.go", 1)),
		readResult("c1", big),
		assistantWith(pagedReadCall("c2", "big.go", 500)),
		readResult("c2", big),
	}
	out, reclaimed := Reclaim(msgs)
	if reclaimed != 0 {
		t.Errorf("reclaimed %d bytes from paged reads; windows are not interchangeable", reclaimed)
	}
	if len(out[2].Text()) != len(big) {
		t.Error("a paged read was cleared")
	}
}

func TestDeterministicSummaryIsDeterministic(t *testing.T) {
	// It feeds a prompt, so unstable output is a prompt-cache break as well as
	// a reproducibility problem. Map iteration order is the usual culprit.
	msgs := []ai.Message{
		ai.UserMessage("fix the parser"),
		{Role: ai.RoleToolResult, ToolName: "write", Content: []ai.Content{
			ai.TextContent("Created zebra.go (10 lines, 1KB)")}},
		{Role: ai.RoleToolResult, ToolName: "edit", Content: []ai.Content{
			ai.TextContent("Applied 1 edit(s) to alpha.go (2 line(s) changed)")}},
		{Role: ai.RoleToolResult, ToolName: "edit", Content: []ai.Content{
			ai.TextContent("Applied 1 edit(s) to middle.go (1 line(s) changed)")}},
	}

	first := DeterministicSummary(msgs, "")
	for range 20 {
		if got := DeterministicSummary(msgs, ""); got != first {
			t.Fatalf("summary is not deterministic:\n%s\n---\n%s", first, got)
		}
	}
	for _, want := range []string{"fix the parser", "alpha.go", "middle.go", "zebra.go"} {
		if !strings.Contains(first, want) {
			t.Errorf("summary missing %q:\n%s", want, first)
		}
	}
	// Sorted, so a reader can diff two summaries.
	alpha, zebra := strings.Index(first, "alpha.go"), strings.Index(first, "zebra.go")
	if alpha > zebra {
		t.Error("file list is not sorted")
	}
}

func TestDeterministicSummaryFoldsInPrevious(t *testing.T) {
	out := DeterministicSummary([]ai.Message{ai.UserMessage("new work")}, "EARLIER CONTEXT")
	if !strings.Contains(out, "EARLIER CONTEXT") {
		t.Error("previous summary was dropped; the oldest history would be lost")
	}
	if !strings.Contains(out, "new work") {
		t.Error("new content missing")
	}
}

func TestPrepareNextTurnNoOpBelowThreshold(t *testing.T) {
	c := NewCompactor(ai.MustLookup(ai.ModelFlash), nil, "")
	actx := &agent.Context{Messages: []ai.Message{ai.UserMessage("small")}}
	if got := c.PrepareNextTurn(agent.TurnContext{Context: actx}); got != nil {
		t.Error("compacted a transcript that is nowhere near the limit")
	}
}

func TestPrepareNextTurnCompactsAboveThreshold(t *testing.T) {
	model := ai.MustLookup(ai.ModelFlash)
	big := strings.Repeat("w", 40_000)

	// Anchor the estimate above the threshold with reported usage.
	msgs := []ai.Message{ai.UserMessage("start")}
	for i := range 6 {
		msgs = append(msgs,
			assistantWith(readCall(fmt.Sprintf("c%d", i), fmt.Sprintf("f%d.go", i))),
			readResult(fmt.Sprintf("c%d", i), big),
			ai.UserMessage(fmt.Sprintf("next step %d", i)),
		)
	}
	msgs = append(msgs, ai.Message{
		Role: ai.RoleAssistant, StopReason: ai.StopEnd,
		Content: []ai.Content{ai.TextContent("done")},
		Usage:   ai.Usage{Input: int(float64(model.ContextWindow) * 0.9), Output: 100},
	})

	var events []CompactionEvent
	c := NewCompactor(model, nil, "")
	c.RetainTail = 5000
	c.OnEvent = func(ev CompactionEvent) { events = append(events, ev) }

	actx := &agent.Context{Messages: msgs}
	update := c.PrepareNextTurn(agent.TurnContext{Context: actx})
	if update == nil || update.Context == nil {
		t.Fatal("no compaction above the threshold")
	}
	if len(update.Context.Messages) >= len(msgs) {
		t.Errorf("transcript did not shrink: %d -> %d", len(msgs), len(update.Context.Messages))
	}
	assertNoOrphanToolResults(t, update.Context.Messages)

	if len(events) != 1 {
		t.Fatalf("got %d compaction events, want 1", len(events))
	}
	if events[0].AfterToken >= events[0].BeforeToken {
		t.Errorf("token estimate did not fall: %d -> %d", events[0].BeforeToken, events[0].AfterToken)
	}
}

func TestCompactionSurvivesSummarizationFailure(t *testing.T) {
	// Compaction runs when the context is nearly full, which is exactly when a
	// request is most likely to fail. A compactor that can fail leaves a
	// session that cannot make progress, so the fallback must engage.
	model := ai.MustLookup(ai.ModelFlash)
	failing := func(ai.Context, ai.StreamOptions) *ai.Stream {
		return ai.ErrorStream(model.ID, "provider error (503): unavailable")
	}

	big := strings.Repeat("e", 20_000)
	msgs := []ai.Message{ai.UserMessage("start")}
	for i := range 4 {
		msgs = append(msgs,
			assistantWith(readCall(fmt.Sprintf("c%d", i), fmt.Sprintf("f%d.go", i))),
			readResult(fmt.Sprintf("c%d", i), big),
			ai.UserMessage("continue"),
		)
	}
	msgs = append(msgs, ai.Message{
		Role: ai.RoleAssistant, StopReason: ai.StopEnd,
		Usage: ai.Usage{Input: int(float64(model.ContextWindow) * 0.9)},
	})

	c := NewCompactor(model, failing, "sys")
	c.RetainTail = 2000
	var ev CompactionEvent
	c.OnEvent = func(e CompactionEvent) { ev = e }

	actx := &agent.Context{Messages: msgs}
	update := c.PrepareNextTurn(agent.TurnContext{Context: actx})
	if update == nil {
		t.Fatal("compaction gave up entirely when summarization failed")
	}
	if ev.UsedLLM {
		t.Error("event claims a model summary despite the failure")
	}
	if ev.Err == nil {
		t.Error("the failure should be reported, not swallowed")
	}
	assertNoOrphanToolResults(t, update.Context.Messages)
	if !strings.Contains(update.Context.Messages[0].Text(), "Recovered summary") {
		t.Error("deterministic fallback did not run")
	}
}

func TestCompactionTerminates(t *testing.T) {
	// The failure this guards: usage on a retained assistant message describes
	// the PRE-compaction prompt. EstimateTokens anchors on the newest such
	// figure, so if compaction leaves it in place the estimate never falls,
	// the next turn compacts again, and so does every turn after that — each
	// one paying for a summary while the transcript stops shrinking.
	model := ai.MustLookup(ai.ModelFlash)
	big := strings.Repeat("t", 30_000)

	msgs := []ai.Message{ai.UserMessage("start")}
	for i := range 8 {
		msgs = append(msgs,
			assistantWith(readCall(fmt.Sprintf("c%d", i), fmt.Sprintf("f%d.go", i))),
			readResult(fmt.Sprintf("c%d", i), big),
			ai.UserMessage("continue"),
		)
	}
	msgs = append(msgs, ai.Message{
		Role: ai.RoleAssistant, StopReason: ai.StopEnd,
		Content: []ai.Content{ai.TextContent("done")},
		Usage:   ai.Usage{Input: int(float64(model.ContextWindow) * 0.95), Output: 100},
	})

	c := NewCompactor(model, nil, "")
	c.RetainTail = 5000
	compactions := 0
	c.OnEvent = func(CompactionEvent) { compactions++ }

	actx := &agent.Context{Messages: msgs}
	// Drive several turns with no new content. After the first compaction the
	// transcript is small, so nothing further should fire.
	for turn := range 5 {
		if update := c.PrepareNextTurn(agent.TurnContext{Context: actx}); update != nil {
			actx = update.Context
		}
		if turn == 0 && compactions != 1 {
			t.Fatalf("first turn produced %d compactions, want 1", compactions)
		}
	}
	if compactions != 1 {
		t.Errorf("compaction ran %d times over 5 idle turns; it should run once and settle", compactions)
	}
	assertNoOrphanToolResults(t, actx.Messages)

	// And the result is genuinely under the threshold, not merely different.
	if got := EstimateTokens(actx.Messages); got >= c.Threshold() {
		t.Errorf("post-compaction estimate %d still at or above the threshold %d", got, c.Threshold())
	}
}
