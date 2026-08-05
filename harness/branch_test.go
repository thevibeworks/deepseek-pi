package harness

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/thevibeworks/deepseek-pi/ai"
)

func toolCall(id, name, args string) ai.Message {
	return ai.Message{
		Role: ai.RoleAssistant, StopReason: ai.StopToolUse,
		Content: []ai.Content{{
			Type: ai.ContentToolCall, ID: id, Name: name, Arguments: json.RawMessage(args),
		}},
	}
}

func toolResult(id, name, text string) ai.Message {
	return ai.Message{
		Role: ai.RoleToolResult, ToolCallID: id, ToolName: name,
		Content: []ai.Content{ai.TextContent(text)},
	}
}

func assistant(text string) ai.Message {
	return ai.Message{
		Role: ai.RoleAssistant, StopReason: ai.StopEnd,
		Content: []ai.Content{ai.TextContent(text)},
	}
}

// transcript is a three-turn session where the middle turn writes a file and
// runs a command, so the turn model has something real to report.
func transcript() []ai.Message {
	return []ai.Message{
		ai.UserMessage("what does the parser do?"),
		toolCall("r1", "read", `{"path":"parser.go"}`),
		toolResult("r1", "read", "package parser"),
		assistant("It parses."),

		ai.UserMessage("fix the off-by-one"),
		toolCall("w1", "write", `{"path":"parser.go","content":"fixed"}`),
		toolResult("w1", "write", "wrote parser.go"),
		toolCall("b1", "bash", `{"command":"go test ./..."}`),
		toolResult("b1", "bash", "ok"),
		assistant("Fixed."),

		ai.UserMessage("now add a benchmark"),
		assistant("Added."),
	}
}

func TestTurnsSegmentOnUserMessagesOnly(t *testing.T) {
	// Tool results carry the user role in no sane encoding, but if they ever
	// did, every tool call would become a branch point and cutting at one would
	// split a call from its result.
	turns := Turns(transcript())
	if len(turns) != 3 {
		t.Fatalf("found %d turns, want 3: %+v", len(turns), turns)
	}
	want := []struct{ start, end int }{{0, 4}, {4, 10}, {10, 12}}
	for i, w := range want {
		if turns[i].Start != w.start || turns[i].End != w.end {
			t.Errorf("turn %d spans [%d,%d), want [%d,%d)",
				i+1, turns[i].Start, turns[i].End, w.start, w.end)
		}
	}
	if turns[0].Prompt != "what does the parser do?" {
		t.Errorf("prompt = %q", turns[0].Prompt)
	}
}

func TestTurnsReportWhatChanged(t *testing.T) {
	turns := Turns(transcript())

	// A turn that only read must not be flagged, or the warning becomes noise
	// and stops being read.
	if turns[0].Changed() {
		t.Errorf("a read-only turn was reported as changing things: %+v", turns[0])
	}
	if turns[2].Changed() {
		t.Errorf("a turn with no tool calls was reported as changing things: %+v", turns[2])
	}

	if got := turns[1].Wrote; len(got) != 1 || got[0] != "parser.go" {
		t.Errorf("wrote = %v, want [parser.go]", got)
	}
	if got := turns[1].Ran; len(got) != 1 || got[0] != "go test ./..." {
		t.Errorf("ran = %v, want [go test ./...]", got)
	}
}

func TestReadOnlyShellIsNotAChange(t *testing.T) {
	// The same classifier the permission gate uses decides this. A turn that
	// ran `rg` did not change anything, and saying it did would train the user
	// to ignore the warning that matters.
	msgs := []ai.Message{
		ai.UserMessage("find the parser"),
		toolCall("b1", "bash", `{"command":"rg -n parse"}`),
		toolResult("b1", "bash", "parser.go:3"),
		assistant("Found it."),
	}
	if turns := Turns(msgs); turns[0].Changed() {
		t.Errorf("a read-only command was reported as a change: %+v", turns[0])
	}
}

func TestCompactionSummaryIsNotAPrompt(t *testing.T) {
	// A summary enters the transcript as a user message. It is a real branch
	// point, but showing it as something the user typed is a lie.
	msgs := []ai.Message{
		ai.UserMessage(summaryPreamble + "Earlier: we fixed the parser."),
		ai.UserMessage("carry on"),
	}
	turns := Turns(msgs)
	if !turns[0].Synthetic {
		t.Error("compaction summary was not marked synthetic")
	}
	if turns[1].Synthetic {
		t.Error("a real prompt was marked synthetic")
	}
	if strings.HasPrefix(turns[0].Prompt, "Earlier conversation was summarized") {
		t.Errorf("the preamble was not stripped from the display text: %q", turns[0].Prompt)
	}
}

// TestRewindNeverOrphansAToolCall is the invariant that makes branch points
// safe at all. The provider rejects a tool_result whose tool_use id it never
// saw, and equally an assistant turn whose calls are never answered, so a cut
// anywhere but a turn boundary produces a session that cannot make a request.
func TestRewindNeverOrphansAToolCall(t *testing.T) {
	msgs := transcript()
	turns := Turns(msgs)

	for _, turn := range turns {
		retained := msgs[:turn.Start]
		calls := map[string]bool{}
		for _, m := range retained {
			for _, c := range m.Content {
				if c.Type == ai.ContentToolCall {
					calls[c.ID] = false
				}
			}
		}
		for _, m := range retained {
			if m.Role != ai.RoleToolResult {
				continue
			}
			if _, ok := calls[m.ToolCallID]; !ok {
				t.Fatalf("rewind to turn %d left result %q with no call", turn.Number, m.ToolCallID)
			}
			calls[m.ToolCallID] = true
		}
		for id, answered := range calls {
			if !answered {
				t.Fatalf("rewind to turn %d left call %q unanswered", turn.Number, id)
			}
		}
	}
}

func testHarness(t *testing.T) *Harness {
	t.Helper()
	t.Setenv("HOME", t.TempDir())
	t.Setenv("DEEPSEEK_PI_HOME", t.TempDir())
	// New only reads the key; nothing here reaches the network.
	t.Setenv("DEEPSEEK_API_KEY", "test-key")

	h, err := New(context.Background(), Options{Cwd: t.TempDir(), NoSkills: true})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(func() { _ = h.Close() })

	msgs := transcript()
	for _, m := range msgs {
		if err := h.Session.Append(m); err != nil {
			t.Fatalf("Append: %v", err)
		}
	}
	h.Agent.Context().Messages = msgs
	return h
}

func TestRewindCutsTheContextAndKeepsTheRecord(t *testing.T) {
	h := testHarness(t)
	path := h.Session.Path()

	rep, err := h.Rewind(2)
	if err != nil {
		t.Fatalf("Rewind: %v", err)
	}
	if rep.Retained != 1 || len(rep.Discarded) != 2 {
		t.Errorf("report = %d kept / %d dropped, want 1 / 2", rep.Retained, len(rep.Discarded))
	}
	if got := len(h.Agent.Context().Messages); got != 4 {
		t.Errorf("context holds %d messages, want 4", got)
	}
	if err := h.Session.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// The view goes back; the file does not. A transcript that forgets the path
	// you abandoned cannot tell you why you abandoned it.
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading session: %v", err)
	}
	if !strings.Contains(string(raw), "now add a benchmark") {
		t.Error("rewind erased the discarded turns from storage")
	}

	loaded, _, err := LoadSession(path)
	if err != nil {
		t.Fatalf("LoadSession: %v", err)
	}
	if len(loaded) != 4 {
		t.Fatalf("reload gave %d messages, want the rewound 4", len(loaded))
	}
	if loaded[0].Text() != "what does the parser do?" {
		t.Errorf("reload lost the retained prefix: %+v", loaded[0])
	}
}

func TestRewindDefaultsToTheLastTurn(t *testing.T) {
	h := testHarness(t)
	rep, err := h.Rewind(0)
	if err != nil {
		t.Fatalf("Rewind: %v", err)
	}
	if rep.Turn != 3 || rep.Retained != 2 {
		t.Errorf("report = turn %d, %d kept; want turn 3, 2 kept", rep.Turn, rep.Retained)
	}
	if got := len(h.Agent.Context().Messages); got != 10 {
		t.Errorf("context holds %d messages, want 10", got)
	}
}

func TestRewindReportsWhatItCannotUndo(t *testing.T) {
	// The one thing a user must be told: the conversation went back and the
	// workspace did not.
	h := testHarness(t)
	rep, err := h.Rewind(2)
	if err != nil {
		t.Fatalf("Rewind: %v", err)
	}
	changed := rep.Changed()
	if len(changed) != 1 {
		t.Fatalf("reported %d changed turns, want 1: %+v", len(changed), changed)
	}
	if changed[0].Number != 2 {
		t.Errorf("blamed turn %d, want 2", changed[0].Number)
	}
	if len(changed[0].Wrote) != 1 || len(changed[0].Ran) != 1 {
		t.Errorf("lost the detail of what changed: %+v", changed[0])
	}
}

func TestRewindRejectsATurnThatDoesNotExist(t *testing.T) {
	h := testHarness(t)
	_, err := h.Rewind(9)
	if err == nil {
		t.Fatal("rewind to a nonexistent turn was accepted")
	}
	// The error has to say how many there are, or the user guesses again.
	if !strings.Contains(err.Error(), "3") {
		t.Errorf("error does not say how many turns exist: %v", err)
	}
	if got := len(h.Agent.Context().Messages); got != 12 {
		t.Errorf("a failed rewind changed the context: %d messages", got)
	}
}

func TestRewindTwiceComposes(t *testing.T) {
	h := testHarness(t)
	path := h.Session.Path()
	if _, err := h.Rewind(3); err != nil {
		t.Fatalf("first Rewind: %v", err)
	}
	if _, err := h.Rewind(2); err != nil {
		t.Fatalf("second Rewind: %v", err)
	}
	if err := h.Session.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	loaded, _, err := LoadSession(path)
	if err != nil {
		t.Fatalf("LoadSession: %v", err)
	}
	if len(loaded) != 4 {
		t.Errorf("replaying two rewinds gave %d messages, want 4", len(loaded))
	}
}

func TestForkLeavesTheBranchYouLeftIntact(t *testing.T) {
	h := testHarness(t)
	parent := h.Session
	parentPath := parent.Path()

	rep, err := h.Fork(3)
	if err != nil {
		t.Fatalf("Fork: %v", err)
	}
	if !rep.Forked || rep.Session == parentPath {
		t.Fatalf("fork did not create a new session: %+v", rep)
	}

	// The whole point: the original is still there, whole, to come back to.
	original, _, err := LoadSession(parentPath)
	if err != nil {
		t.Fatalf("LoadSession(parent): %v", err)
	}
	if len(original) != 12 {
		t.Errorf("the parent transcript was modified: %d messages, want 12", len(original))
	}

	if err := h.Session.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	branch, info, err := LoadSession(rep.Session)
	if err != nil {
		t.Fatalf("LoadSession(branch): %v", err)
	}
	if len(branch) != 10 {
		t.Errorf("branch holds %d messages, want the retained 10", len(branch))
	}
	if info.Parent != parent.ID() {
		t.Errorf("branch parent = %q, want %q", info.Parent, parent.ID())
	}
	if info.ID == parent.ID() {
		t.Error("branch reused the parent's session id")
	}
}

func TestForkRedirectsSubsequentWrites(t *testing.T) {
	// The harness appends every message through the session it currently holds.
	// A callback that captured the pre-fork session instead would keep writing
	// to a closed file, and the branch would record nothing.
	h := testHarness(t)
	parent := h.Session

	rep, err := h.Fork(2)
	if err != nil {
		t.Fatalf("Fork: %v", err)
	}
	if h.session() == parent {
		t.Fatal("the harness still holds the pre-fork session")
	}
	if err := parent.Append(ai.UserMessage("late")); err == nil {
		t.Error("the parent session is still writable after being forked away from")
	}

	if err := h.session().Append(ai.UserMessage("a different approach")); err != nil {
		t.Fatalf("appending after a fork: %v", err)
	}
	if err := h.Session.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	branch, _, err := LoadSession(rep.Session)
	if err != nil {
		t.Fatalf("LoadSession: %v", err)
	}
	if len(branch) != 5 || branch[4].Text() != "a different approach" {
		t.Errorf("the new message did not land in the branch: %d messages", len(branch))
	}
}

func TestForkFromHereCopiesEverything(t *testing.T) {
	h := testHarness(t)
	rep, err := h.Fork(0)
	if err != nil {
		t.Fatalf("Fork: %v", err)
	}
	if len(rep.Discarded) != 0 || rep.Retained != 3 {
		t.Errorf("report = %d kept / %d dropped, want 3 / 0", rep.Retained, len(rep.Discarded))
	}
	if got := len(h.Agent.Context().Messages); got != 12 {
		t.Errorf("context holds %d messages, want all 12", got)
	}
}

func TestForkCarriesAccountingForward(t *testing.T) {
	// A branch continues the same run's spend. Zeroing it would make forking
	// look like a way to reset the bill.
	h := testHarness(t)
	if err := h.Session.Append(ai.Message{
		Role: ai.RoleAssistant, Usage: ai.Usage{Input: 5000, Output: 100},
	}); err != nil {
		t.Fatalf("Append: %v", err)
	}
	before := h.Session.Usage
	if _, err := h.Fork(0); err != nil {
		t.Fatalf("Fork: %v", err)
	}
	if h.Session.Usage != before {
		t.Errorf("usage after fork = %+v, want %+v", h.Session.Usage, before)
	}
}

func TestListingHonorsRewind(t *testing.T) {
	// A listing that advertises a prompt the session no longer holds sends you
	// into the wrong transcript.
	h := testHarness(t)
	cwd := h.Workspace.Root
	if _, err := h.Rewind(1); err != nil {
		t.Fatalf("Rewind: %v", err)
	}
	if err := h.Session.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	list, err := ListSessions(cwd)
	if err != nil {
		t.Fatalf("ListSessions: %v", err)
	}
	if len(list) != 1 {
		t.Fatalf("listed %d sessions, want 1", len(list))
	}
	if list[0].Messages != 0 {
		t.Errorf("listing reports %d messages after a full rewind, want 0", list[0].Messages)
	}
	if list[0].Preview != "" {
		t.Errorf("listing still shows a discarded prompt: %q", list[0].Preview)
	}
}

func TestAliasedPathArgumentsAreDetected(t *testing.T) {
	// The transcript stores the model's raw arguments; the loop heals aliases
	// into a copy it never writes back. Reading only "path" therefore misses a
	// real write, and a missed write is exactly the warning that matters.
	for _, key := range []string{"path", "file_path", "filepath", "filename", "file"} {
		msgs := []ai.Message{
			ai.UserMessage("fix it"),
			toolCall("w1", "write", `{"`+key+`":"parser.go","content":"x"}`),
			toolResult("w1", "write", "wrote parser.go"),
		}
		turn := Turns(msgs)[0]
		if !turn.Changed() {
			t.Errorf("a write using %q went unreported", key)
			continue
		}
		if turn.Wrote[0] != "parser.go" {
			t.Errorf("%q gave path %q, want parser.go", key, turn.Wrote[0])
		}
	}
}

func TestDelegatedChangesAreReported(t *testing.T) {
	// A child's tool calls never reach this transcript: the parent sees one
	// task call and a text report. Without this, a turn that delegated all its
	// writing reads as having changed nothing.
	cases := []struct {
		role string
		want bool
	}{
		{"implementer", true},
		{"explorer", false},
		{"reviewer", false},
		{"tester", false},
		// Unknown means unsafe, as with the shell classifier: a role added later
		// must not go unreported because this predates it.
		{"demolisher", true},
	}
	for _, c := range cases {
		msgs := []ai.Message{
			ai.UserMessage("fix it"),
			toolCall("t1", "task", `{"role":"`+c.role+`","prompt":"go"}`),
			toolResult("t1", "task", "done"),
		}
		turn := Turns(msgs)[0]
		if got := turn.Changed(); got != c.want {
			t.Errorf("role %q reported changed=%v, want %v", c.role, got, c.want)
		}
	}
}

func TestClearIsPersisted(t *testing.T) {
	// /clear used to change only the in-memory context, so `-c` replayed the
	// file and handed back every message the user had just been told was gone.
	h := testHarness(t)
	path := h.Session.Path()

	rep, err := h.Clear()
	if err != nil {
		t.Fatalf("Clear: %v", err)
	}
	if len(h.Agent.Context().Messages) != 0 {
		t.Error("clear left messages in the context")
	}
	// It still owes the user the truth about what it did not undo.
	if len(rep.Changed()) != 1 {
		t.Errorf("clear reported %d changed turns, want 1", len(rep.Changed()))
	}
	if err := h.Session.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	loaded, _, err := LoadSession(path)
	if err != nil {
		t.Fatalf("LoadSession: %v", err)
	}
	if len(loaded) != 0 {
		t.Errorf("resuming a cleared session restored %d messages", len(loaded))
	}
}

func TestCutForgetsASummaryItDiscarded(t *testing.T) {
	// The running summary is carried forward so a later compaction can update
	// it. Rewind past it and that text describes turns the user deliberately
	// threw away — folding them back in is the opposite of what they asked for.
	h := testHarness(t)
	h.Agent.Context().Messages = []ai.Message{
		ai.UserMessage("first"),
		assistant("ok"),
		ai.UserMessage(summaryPreamble + "we already fixed the parser"),
		assistant("understood"),
	}
	h.Compactor.SyncSummary(h.Agent.Context().Messages)
	if h.Compactor.lastSummary == "" {
		t.Fatal("the summary in the transcript was not picked up")
	}

	if _, err := h.Rewind(2); err != nil {
		t.Fatalf("Rewind: %v", err)
	}
	if h.Compactor.lastSummary != "" {
		t.Errorf("a discarded summary is still being carried forward: %q", h.Compactor.lastSummary)
	}
}

func TestCutKeepsASummaryItRetained(t *testing.T) {
	h := testHarness(t)
	h.Agent.Context().Messages = []ai.Message{
		ai.UserMessage(summaryPreamble + "we already fixed the parser"),
		assistant("understood"),
		ai.UserMessage("now benchmark it"),
		assistant("done"),
	}
	if _, err := h.Rewind(2); err != nil {
		t.Fatalf("Rewind: %v", err)
	}
	if h.Compactor.lastSummary != "we already fixed the parser" {
		t.Errorf("a retained summary was dropped: %q", h.Compactor.lastSummary)
	}
}
