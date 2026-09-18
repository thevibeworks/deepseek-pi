package harness

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/thevibeworks/deepseek-pi/agent"
	"github.com/thevibeworks/deepseek-pi/ai"
)

// scriptedChild returns a stream that replays canned assistant messages, then
// repeats the last one forever. Repeating is what lets a budget test drive a
// child that will not stop on its own.
func scriptedChild(msgs ...ai.Message) (func(context.Context) ai.StreamFunc, *int) {
	var mu sync.Mutex
	calls := 0
	return func(context.Context) ai.StreamFunc {
		return func(_ ai.Context, opts ai.StreamOptions) *ai.Stream {
			mu.Lock()
			i := calls
			calls++
			mu.Unlock()

			if i >= len(msgs) {
				i = len(msgs) - 1
			}
			final := msgs[i]
			final.Model = opts.Model
			final.Usage = ai.Usage{Input: 100, Output: 10}

			st := ai.NewStream()
			go func() { st.Finish(&final) }()
			return st
		}
	}, &calls
}

func childToolCall(id, name, args string) ai.Message {
	return ai.Message{
		Role: ai.RoleAssistant, StopReason: ai.StopToolUse,
		Content: []ai.Content{{
			Type: ai.ContentToolCall, ID: id, Name: name, Arguments: json.RawMessage(args),
		}},
	}
}

func childAnswer(text string) ai.Message {
	return ai.Message{
		Role: ai.RoleAssistant, StopReason: ai.StopEnd,
		Content: []ai.Content{ai.TextContent(text)},
	}
}

func testEnv(t *testing.T, stream func(context.Context) ai.StreamFunc, parentMode Mode) *taskEnv {
	t.Helper()
	t.Setenv("DEEPSEEK_PI_HOME", t.TempDir())
	return &taskEnv{
		cwd: t.TempDir(), newStream: stream,
		parentModel:  ai.MustLookup(ai.ModelFlash),
		parentPolicy: NewPolicy(parentMode, nil),
		budget:       Budget{MaxTurns: 5, MaxTokens: 1_000_000, MaxWall: 30 * time.Second},
	}
}

func runTask(t *testing.T, env *taskEnv, role, prompt string) (agent.ToolResult, error) {
	t.Helper()
	tool := newTaskTool(env)
	args, _ := json.Marshal(map[string]string{"role": role, "prompt": prompt})
	raw := json.RawMessage(args)
	if tool.PrepareArguments != nil {
		raw = tool.PrepareArguments(raw)
	}
	if err := agent.ValidateArguments(tool.Parameters, raw); err != nil {
		return agent.ToolResult{}, err
	}
	return tool.Execute(context.Background(),
		agent.ToolCall{ID: "t1", Name: "task", Arguments: raw}, nil)
}

func resultText(r agent.ToolResult) string {
	var b strings.Builder
	for _, c := range r.Content {
		b.WriteString(c.Text)
	}
	return b.String()
}

func TestSubagentCannotSpawnSubagents(t *testing.T) {
	// Recursion is prevented structurally: the child's tool set does not
	// contain the task tool, so it cannot express a nested spawn. This drives a
	// child that tries anyway and checks it is told the tool does not exist —
	// no depth counter, nothing to tune, nothing to get wrong.
	stream, _ := scriptedChild(
		childToolCall("c1", "task", `{"role":"explorer","prompt":"go deeper"}`),
		childAnswer("could not delegate"),
	)
	env := testEnv(t, stream, ModeDefault)

	res, err := runTask(t, env, "explorer", "find something")
	if err != nil {
		t.Fatalf("task: %v", err)
	}
	if !strings.Contains(resultText(res), "could not delegate") {
		t.Errorf("child did not recover from the refused nested spawn:\n%s", resultText(res))
	}

	// The proof is in the child's own transcript, which records what the tool
	// call actually returned. The parent never sees this, so it is the only
	// place the refusal is observable.
	msgs := childTranscript(t, res)
	var refused bool
	for _, m := range msgs {
		if m.Role == ai.RoleToolResult && m.ToolName == "task" {
			if !m.IsError || !strings.Contains(m.Text(), "not found") {
				t.Errorf("nested task call was not refused as an unknown tool: %q", m.Text())
			}
			refused = true
		}
	}
	if !refused {
		t.Fatal("the child's nested task call left no result; the tool may have been available")
	}
}

// childTranscript loads the session a child wrote, which is the only record of
// what happened inside it.
func childTranscript(t *testing.T, res agent.ToolResult) []ai.Message {
	t.Helper()
	details, ok := res.Details.(map[string]any)
	if !ok {
		t.Fatalf("result details missing: %+v", res.Details)
	}
	path, _ := details["session"].(string)
	if path == "" {
		t.Fatal("no child session path recorded")
	}
	msgs, _, err := LoadSession(path)
	if err != nil {
		t.Fatalf("loading child transcript: %v", err)
	}
	return msgs
}

func TestSubagentBudgetStopsRunawayChild(t *testing.T) {
	// The failure that matters: a child that never decides it is finished.
	// Unbounded it burns the whole session's budget and reports nothing.
	stream, calls := scriptedChild(
		childToolCall("c1", "bash", `{"command":"echo working"}`),
	)
	env := testEnv(t, stream, ModeDefault)
	env.budget.MaxTurns = 3

	res, err := runTask(t, env, "explorer", "loop forever")
	if err != nil {
		t.Fatalf("task: %v", err)
	}

	if *calls > 5 {
		t.Errorf("child made %d provider calls; the turn budget should have stopped it near 3", *calls)
	}
	text := resultText(res)
	// A partial report that reads as complete is worse than no report, because
	// the parent acts on it as if the work were done.
	if !strings.Contains(text, "PARTIAL") {
		t.Errorf("a budget-stopped child did not report as partial:\n%s", text)
	}
	if !strings.Contains(text, "turn budget") {
		t.Errorf("the report does not say which budget stopped it:\n%s", text)
	}
}

func TestSubagentInheritsParentRestriction(t *testing.T) {
	// An implementer normally writes. Under a plan-mode parent it must not,
	// or the parent's mode is decorative.
	stream, _ := scriptedChild(
		childToolCall("c1", "write", `{"path":"new.txt","content":"x"}`),
		childAnswer("blocked from writing"),
	)
	env := testEnv(t, stream, ModePlan)

	res, err := runTask(t, env, "implementer", "write a file")
	if err != nil {
		t.Fatalf("task: %v", err)
	}
	if !strings.Contains(resultText(res), "blocked from writing") {
		t.Errorf("unexpected report:\n%s", resultText(res))
	}
	// The filesystem is the check that matters, not the narration.
	if _, err := os.Stat(filepath.Join(env.cwd, "new.txt")); err == nil {
		t.Fatal("an implementer child wrote a file under a plan-mode parent")
	}
	// And the refusal should name plan mode, so the child knows why.
	var blocked bool
	for _, m := range childTranscript(t, res) {
		if m.Role == ai.RoleToolResult && m.ToolName == "write" && m.IsError {
			blocked = true
			if !strings.Contains(m.Text(), "plan mode") {
				t.Errorf("refusal does not explain the cause: %q", m.Text())
			}
		}
	}
	if !blocked {
		t.Error("no refusal recorded for the write attempt")
	}
}

func TestStricterMode(t *testing.T) {
	cases := []struct{ a, b, want Mode }{
		{ModeYolo, ModePlan, ModePlan},
		{ModePlan, ModeYolo, ModePlan},
		{ModeYolo, ModeDefault, ModeDefault},
		{ModeDefault, ModeYolo, ModeDefault},
		{ModeYolo, ModeYolo, ModeYolo},
	}
	for _, c := range cases {
		if got := stricterMode(c.a, c.b); got != c.want {
			t.Errorf("stricterMode(%s, %s) = %s, want %s", c.a, c.b, got, c.want)
		}
	}
}

func TestTaskRejectsUnknownRole(t *testing.T) {
	stream, _ := scriptedChild(childAnswer("done"))
	env := testEnv(t, stream, ModeDefault)

	_, err := runTask(t, env, "wizard", "do magic")
	if err == nil {
		t.Fatal("unknown role accepted")
	}
	// The error has to list the real roles, or the model guesses again.
	for _, role := range Roles() {
		if !strings.Contains(err.Error(), string(role)) {
			t.Errorf("error does not offer %q: %v", role, err)
		}
	}
}

func TestTaskRejectsEmptyPrompt(t *testing.T) {
	stream, _ := scriptedChild(childAnswer("done"))
	env := testEnv(t, stream, ModeDefault)
	if _, err := runTask(t, env, "explorer", "   "); err == nil {
		t.Error("empty prompt accepted; the child would have nothing to do")
	}
}

func TestReportCarriesOnlyTheAnswerAndAccounting(t *testing.T) {
	// The whole point of delegating is keeping the child's tool traffic out of
	// the parent's context. If the report carried it, the tool would cost more
	// than doing the work inline.
	stream, _ := scriptedChild(
		childToolCall("c1", "bash", `{"command":"echo a-very-distinctive-tool-output"}`),
		childAnswer("The answer is 42."),
	)
	env := testEnv(t, stream, ModeDefault)

	res, err := runTask(t, env, "explorer", "compute the answer")
	if err != nil {
		t.Fatalf("task: %v", err)
	}
	text := resultText(res)

	if !strings.Contains(text, "The answer is 42.") {
		t.Errorf("report is missing the child's answer:\n%s", text)
	}
	if strings.Contains(text, "a-very-distinctive-tool-output") {
		t.Errorf("the child's tool output leaked into the parent's context:\n%s", text)
	}
	// Accounting travels with the report so the parent's cost is not a mystery.
	for _, want := range []string{"explorer sub-agent", "Transcript:"} {
		if !strings.Contains(text, want) {
			t.Errorf("report missing %q:\n%s", want, text)
		}
	}
}

func TestConcurrentChildrenAreCapped(t *testing.T) {
	// A model that decides to spawn twenty explorers is confused; failing
	// visibly is more useful than serving it.
	release := make(chan struct{})
	blocking := func(context.Context) ai.StreamFunc {
		return func(_ ai.Context, _ ai.StreamOptions) *ai.Stream {
			st := ai.NewStream()
			go func() {
				<-release
				st.Finish(&ai.Message{Role: ai.RoleAssistant, StopReason: ai.StopEnd,
					Content: []ai.Content{ai.TextContent("done")}})
			}()
			return st
		}
	}
	env := testEnv(t, blocking, ModeDefault)

	var wg sync.WaitGroup
	for range MaxConcurrentChildren {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _ = runTask(t, env, "explorer", "wait")
		}()
	}
	// Wait for the cap to fill.
	for i := 0; i < 200; i++ {
		env.mu.Lock()
		n := env.spawned
		env.mu.Unlock()
		if n >= MaxConcurrentChildren {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}

	_, err := runTask(t, env, "explorer", "one too many")
	if err == nil {
		t.Error("the concurrency cap did not apply")
	}
	close(release)
	wg.Wait()
}

func TestEveryRoleRunsOnTheParentModel(t *testing.T) {
	// reviewer was pinned to pro until V4.1 Flash (2026-09-10). Every role now
	// inherits the parent's model, so a flash session never pays pro rates
	// behind the user's back and a pro session keeps pro everywhere.
	for _, parent := range []string{ai.ModelFlash, ai.ModelPro} {
		for _, role := range Roles() {
			var mu sync.Mutex
			var sent []string
			stream := func(context.Context) ai.StreamFunc {
				return func(_ ai.Context, opts ai.StreamOptions) *ai.Stream {
					mu.Lock()
					sent = append(sent, opts.Model)
					mu.Unlock()
					final := childAnswer("done")
					final.Model = opts.Model
					st := ai.NewStream()
					go func() { st.Finish(&final) }()
					return st
				}
			}
			env := testEnv(t, stream, ModeYolo)
			env.parentModel = ai.MustLookup(parent)
			if _, err := runTask(t, env, string(role), "do the thing"); err != nil {
				t.Fatalf("%s under %s: %v", role, parent, err)
			}
			if len(sent) == 0 || sent[0] != parent {
				t.Errorf("%s under a %s parent sent model %v", role, parent, sent)
			}
		}
	}
}
