package harness

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/thevibeworks/deepseek-pi/ai"
)

// bulkFile generates a large but plausible Go source file.
func bulkFile(lines int) string {
	var b strings.Builder
	b.WriteString("package demo\n\n")
	for i := range lines {
		fmt.Fprintf(&b, "// Item%d documents the %dth generated helper in this file.\n", i, i)
		fmt.Fprintf(&b, "func Item%d(in int) int { return in*%d + %d }\n\n", i, i+1, i)
	}
	return b.String()
}

// requireLive gates the tests that hit the real API. A key alone is not
// enough: see the note in ai/live_test.go.
func requireLive(t *testing.T) {
	t.Helper()
	if os.Getenv("DEEPSEEK_PI_LIVE") == "" {
		t.Skip("set DEEPSEEK_PI_LIVE=1 to run tests against the real API (they cost money)")
	}
	if os.Getenv("DEEPSEEK_API_KEY") == "" {
		t.Skip("DEEPSEEK_API_KEY not set; skipping live API test")
	}
}

func liveWorkspace(t *testing.T) string {
	t.Helper()
	requireLive(t)
	dir := t.TempDir()
	t.Setenv("DEEPSEEK_PI_HOME", t.TempDir())
	t.Setenv("HOME", t.TempDir()) // no personal skills in the prompt

	files := map[string]string{
		"alpha.go": "package demo\n\n// AlphaSecret is 4711.\nconst AlphaSecret = 4711\n",
		"beta.go":  "package demo\n\n// BetaSecret is 1337.\nconst BetaSecret = 1337\n",
		// Large enough that reading it leaves a substantial retained tail after
		// compaction. With a tiny fixture the post-compaction prompt is almost
		// entirely the still-cached system prompt, so there is no shortfall to
		// observe and the test would be measuring nothing.
		"bulk.go": bulkFile(600),
	}
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

// TestLiveSummarizationProducesUsableSummary checks the summarization prompt
// against the real model. A summary that omits what was asked for or what was
// done is worse than useless: it silently replaces the transcript.
func TestLiveSummarizationProducesUsableSummary(t *testing.T) {
	requireLive(t)
	client := ai.NewClient(os.Getenv("DEEPSEEK_API_KEY"))
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	model := ai.MustLookup(ai.ModelFlash)
	c := NewCompactor(model, client.StreamFunc(ctx), "You are a coding agent.")

	// A well-formed transcript: every tool result follows its call, because
	// that is what the real cut point guarantees and what the API demands.
	head := []ai.Message{
		ai.UserMessage("Rename the function Frobnicate to Process across the repo, and keep the tests passing."),
		assistantWith(readCall("c1", "frob.go")),
		readResult("c1", "1\tpackage main\n2\tfunc Frobnicate() {}\n"),
		assistantWith(ai.Content{
			Type: ai.ContentToolCall, ID: "c2", Name: "edit",
			Arguments: []byte(`{"path":"frob.go","edits":[{"oldText":"Frobnicate","newText":"Process"}]}`),
		}),
		{Role: ai.RoleToolResult, ToolName: "edit", ToolCallID: "c2",
			Content: []ai.Content{ai.TextContent("Applied 1 edit(s) to frob.go (1 line(s) changed).")}},
		assistantWith(ai.Content{
			Type: ai.ContentToolCall, ID: "c3", Name: "bash",
			Arguments: []byte(`{"command":"go test ./..."}`),
		}),
		{Role: ai.RoleToolResult, ToolName: "bash", ToolCallID: "c3", IsError: true,
			Content: []ai.Content{ai.TextContent("FAIL demo [build failed]\nfrob_test.go:9: undefined: Frobnicate")}},
		ai.UserMessage("the test file still refers to the old name"),
	}

	summary, usedLLM, err := c.renderSummary(ctx, head, "")
	if err != nil {
		t.Fatalf("renderSummary: %v", err)
	}
	if !usedLLM {
		t.Fatal("fell back to the deterministic summary despite a working client")
	}
	t.Logf("summary:\n%s", summary)

	// The summary must carry the task, the change made, and the outstanding
	// failure. Those three are what let another turn continue the work.
	lower := strings.ToLower(summary)
	for _, want := range []string{"frobnicate", "process", "frob.go", "test"} {
		if !strings.Contains(lower, want) {
			t.Errorf("summary omits %q, which a continuation would need:\n%s", want, summary)
		}
	}
	if len(summary) > 4000 {
		t.Errorf("summary is %d bytes; it is meant to be smaller than what it replaces", len(summary))
	}
}

// TestLiveCompactionKeepsSessionWorking is the end-to-end check: force
// compaction mid-session and confirm the agent still answers correctly using
// information that only existed before the cut.
func TestLiveCompactionKeepsSessionWorking(t *testing.T) {
	dir := liveWorkspace(t)
	ctx, cancel := context.WithTimeout(context.Background(), 6*time.Minute)
	defer cancel()

	h, err := New(ctx, Options{Cwd: dir, Mode: ModeYolo, NoSkills: true})
	if err != nil {
		t.Fatalf("harness: %v", err)
	}
	defer func() { _ = h.Close() }()

	var events []CompactionEvent
	record := h.Compactor.OnEvent
	h.Compactor.OnEvent = func(ev CompactionEvent) {
		if record != nil {
			record(ev)
		}
		events = append(events, ev)
	}
	// Trigger almost immediately so compaction is exercised without generating
	// half a million tokens of filler.
	h.Compactor.Trigger = 0.002 // ~1.2k tokens
	h.Compactor.RetainTail = 400

	if _, err := h.Agent.Prompt(ctx, "Read alpha.go and tell me the value of AlphaSecret. Just the number."); err != nil {
		t.Fatalf("turn 1: %v", err)
	}
	if _, err := h.Agent.Prompt(ctx, "Now read beta.go and tell me BetaSecret. Just the number."); err != nil {
		t.Fatalf("turn 2: %v", err)
	}

	if len(events) == 0 {
		t.Fatalf("compaction never fired; estimate is %d, threshold %d",
			EstimateTokens(h.Agent.Context().Messages), h.Compactor.Threshold())
	}
	for _, ev := range events {
		if ev.Err != nil {
			t.Errorf("compaction reported an error: %v", ev.Err)
		}
	}
	t.Logf("compactions: %d, first: %d -> %d tokens (llm=%v)",
		len(events), events[0].BeforeToken, events[0].AfterToken, events[0].UsedLLM)

	// The transcript was rewritten at least once. It must still be a valid
	// payload, and the model must still be able to work from it.
	assertNoOrphanToolResults(t, h.Agent.Context().Messages)

	msgs, err := h.Agent.Prompt(ctx,
		"Without reading any files again, what was the value of AlphaSecret that you found earlier?")
	if err != nil {
		t.Fatalf("turn 3: %v", err)
	}
	last := msgs[len(msgs)-1]
	if last.StopReason == ai.StopError {
		t.Fatalf("post-compaction turn failed: %s", last.ErrorMessage)
	}
	// 4711 came from a turn that compaction may well have summarized away. If
	// the summary is doing its job, the answer survives.
	answer := ""
	for _, m := range msgs {
		answer += m.Text()
	}
	if !strings.Contains(answer, "4711") {
		t.Errorf("the value from before compaction did not survive; answer was:\n%s", answer)
	}
}

// TestLiveCacheTrackerSeesCompactionBreak closes the loop on cache
// attribution: the unit tests prove it names causes correctly, and the manual
// check proves it stays quiet on a healthy session. This proves it fires on a
// real break, against the real provider, and labels the one break we choose to
// take as chosen rather than broken.
func TestLiveCacheTrackerSeesCompactionBreak(t *testing.T) {
	dir := liveWorkspace(t)
	ctx, cancel := context.WithTimeout(context.Background(), 6*time.Minute)
	defer cancel()

	h, err := New(ctx, Options{Cwd: dir, Mode: ModeYolo, NoSkills: true})
	if err != nil {
		t.Fatalf("harness: %v", err)
	}
	defer func() { _ = h.Close() }()

	// Sized so the retained tail after compaction is real: the shortfall being
	// measured is what a stable prefix would have served but could not.
	h.Compactor.Trigger = 0.02 // ~12k tokens
	h.Compactor.RetainTail = 3000

	if _, err := h.Agent.Prompt(ctx, "Read bulk.go and tell me how many Item functions it defines."); err != nil {
		t.Fatalf("turn 1: %v", err)
	}
	if _, err := h.Agent.Prompt(ctx, "Read alpha.go and state AlphaSecret. Just the number."); err != nil {
		t.Fatalf("turn 2: %v", err)
	}
	if _, err := h.Agent.Prompt(ctx, "Say done."); err != nil {
		t.Fatalf("turn 3: %v", err)
	}

	if len(h.Cache.Breaks) == 0 {
		t.Fatalf("compaction rewrote the transcript but no cache break was recorded\n%s",
			h.Cache.Report())
	}
	var sanctioned int
	for _, b := range h.Cache.Breaks {
		if b.Sanctioned {
			sanctioned++
			continue
		}
		// An UNsanctioned break here means the prefix drifted for a reason we
		// did not choose, which is exactly what this tool exists to surface.
		t.Errorf("unexpected cache break: %s", b)
	}
	if sanctioned == 0 {
		t.Error("the compaction break was not labelled as chosen")
	}
	t.Logf("cache report:\n%s", h.Cache.Report())
}

// TestLiveSubagentDelegation checks the whole sub-agent path against the real
// model: the parent decides to delegate, children run in their own contexts,
// and only their reports come back.
func TestLiveSubagentDelegation(t *testing.T) {
	dir := liveWorkspace(t)
	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Minute)
	defer cancel()

	h, err := New(ctx, Options{Cwd: dir, Mode: ModeYolo, NoSkills: true})
	if err != nil {
		t.Fatalf("harness: %v", err)
	}
	defer func() { _ = h.Close() }()

	msgs, err := h.Agent.Prompt(ctx,
		"Use two explorer sub-agents in a single batch, running concurrently: one to find "+
			"the value of AlphaSecret, one to find the value of BetaSecret. Do not read the "+
			"files yourself. Then report both numbers.")
	if err != nil {
		t.Fatalf("prompt: %v", err)
	}

	children := h.Children()
	if len(children) < 2 {
		t.Fatalf("expected at least 2 sub-agents, got %d", len(children))
	}
	for _, c := range children {
		if c.Truncated != "" {
			t.Errorf("%s sub-agent hit a budget: %s", c.Role, c.Truncated)
		}
		if c.Usage.Input == 0 {
			t.Errorf("%s sub-agent recorded no usage", c.Role)
		}
	}

	answer := ""
	for _, m := range msgs {
		answer += m.Text()
	}
	for _, want := range []string{"4711", "1337"} {
		if !strings.Contains(answer, want) {
			t.Errorf("the parent's answer is missing %s; delegation did not carry the result:\n%s",
				want, answer)
		}
	}

	// The point of delegating is that the children's tool output never reaches
	// the parent. If file contents leaked, the tool costs more than inlining.
	for _, m := range h.Agent.Context().Messages {
		if m.Role == ai.RoleToolResult && m.ToolName == "read" {
			t.Error("the parent ran a read itself; the test asked it to delegate")
		}
	}
	t.Logf("children: %d, parent cost $%.4f", len(children), h.Session.Usage.Cost.Total)
	for _, c := range children {
		t.Logf("  %s: %d turns, %d in / %d out, $%.4f, %s",
			c.Role, c.Turns, c.Usage.Input, c.Usage.Output, c.Usage.Cost.Total,
			c.Duration.Round(time.Millisecond))
	}
}
