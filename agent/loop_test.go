package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/thevibeworks/deepseek-pi/ai"
)

// scriptedStream replays canned assistant messages, one per call, so loop
// behaviour can be tested without a provider.
func scriptedStream(msgs ...ai.Message) (ai.StreamFunc, *int) {
	var mu sync.Mutex
	calls := 0
	fn := func(_ ai.Context, _ ai.StreamOptions) *ai.Stream {
		mu.Lock()
		i := calls
		calls++
		mu.Unlock()

		st := ai.NewStream()
		var m ai.Message
		if i < len(msgs) {
			m = msgs[i]
		} else {
			m = ai.Message{Role: ai.RoleAssistant, StopReason: ai.StopEnd,
				Content: []ai.Content{ai.TextContent("done")}}
		}
		final := m
		go func() {
			st.Push(ai.Event{Type: ai.EventStart, Partial: &final})
			st.Finish(&final)
		}()
		return st
	}
	return fn, &calls
}

func toolCallMsg(stop ai.StopReason, calls ...ai.Content) ai.Message {
	return ai.Message{Role: ai.RoleAssistant, StopReason: stop, Content: calls}
}

func call(id, name, args string) ai.Content {
	return ai.Content{Type: ai.ContentToolCall, ID: id, Name: name, Arguments: json.RawMessage(args)}
}

// recordingTool notes each invocation and optionally delays, so completion
// order can be made to differ from source order.
func recordingTool(name string, delay time.Duration, log *[]string, mu *sync.Mutex) Tool {
	return Tool{
		Name:       name,
		Label:      name,
		Parameters: ai.Object(nil),
		Execute: func(ctx context.Context, c ToolCall, _ UpdateFunc) (ToolResult, error) {
			if delay > 0 {
				select {
				case <-time.After(delay):
				case <-ctx.Done():
				}
			}
			mu.Lock()
			*log = append(*log, name)
			mu.Unlock()
			return Text("ran " + name), nil
		},
	}
}

func TestLengthTruncationFailsEntireToolBatch(t *testing.T) {
	// A response cut off by the token limit may carry tool calls whose
	// arguments parse and validate but are silently incomplete. Executing any
	// of them risks acting on half a command.
	var log []string
	var mu sync.Mutex
	tool := recordingTool("danger", 0, &log, &mu)

	stream, _ := scriptedStream(
		toolCallMsg(ai.StopLength, call("a", "danger", `{}`), call("b", "danger", `{}`)),
		ai.Message{Role: ai.RoleAssistant, StopReason: ai.StopEnd, Content: []ai.Content{ai.TextContent("ok")}},
	)

	c := &Context{Tools: []Tool{tool}}
	msgs := Run(context.Background(), []ai.Message{ai.UserMessage("go")}, c,
		LoopConfig{Model: ai.ModelFlash}, stream, func(Event) {})

	mu.Lock()
	executed := len(log)
	mu.Unlock()
	if executed != 0 {
		t.Fatalf("executed %d tools from a truncated message; want 0", executed)
	}

	var results int
	for _, m := range msgs {
		if m.Role == ai.RoleToolResult {
			results++
			if !m.IsError {
				t.Error("truncated tool call result should be an error")
			}
			if !strings.Contains(m.Text(), "output token limit") {
				t.Errorf("error text should explain the cause, got %q", m.Text())
			}
		}
	}
	if results != 2 {
		t.Errorf("got %d tool results, want 2 (every call must be answered)", results)
	}
}

func TestToolResultsAppendInSourceOrder(t *testing.T) {
	// Completion order and transcript order are different contracts. The
	// provider requires results in the order the model requested them; a
	// transcript ordered by timing is not reproducible.
	var log []string
	var mu sync.Mutex
	slow := recordingTool("slow", 60*time.Millisecond, &log, &mu)
	fast := recordingTool("fast", 0, &log, &mu)

	stream, _ := scriptedStream(
		toolCallMsg(ai.StopToolUse, call("1", "slow", `{}`), call("2", "fast", `{}`)),
		ai.Message{Role: ai.RoleAssistant, StopReason: ai.StopEnd, Content: []ai.Content{ai.TextContent("ok")}},
	)

	var endOrder []string
	c := &Context{Tools: []Tool{slow, fast}}
	msgs := Run(context.Background(), []ai.Message{ai.UserMessage("go")}, c,
		LoopConfig{Model: ai.ModelFlash, ToolExecution: ModeParallel},
		stream, func(ev Event) {
			if ev.Type == EventToolEnd {
				endOrder = append(endOrder, ev.ToolName)
			}
		})

	var resultOrder []string
	for _, m := range msgs {
		if m.Role == ai.RoleToolResult {
			resultOrder = append(resultOrder, m.ToolName)
		}
	}

	if len(resultOrder) != 2 || resultOrder[0] != "slow" || resultOrder[1] != "fast" {
		t.Errorf("result messages = %v, want [slow fast] (assistant source order)", resultOrder)
	}
	// The fast tool genuinely finished first, so end-events prove completion
	// ordering is preserved independently.
	if len(endOrder) != 2 || endOrder[0] != "fast" {
		t.Errorf("tool end events = %v, want fast first (completion order)", endOrder)
	}
}

func TestTerminateRequiresEveryResult(t *testing.T) {
	terminating := Tool{
		Name: "stop", Label: "stop", Parameters: ai.Object(nil),
		Execute: func(context.Context, ToolCall, UpdateFunc) (ToolResult, error) {
			r := Text("stopping")
			r.Terminate = true
			return r, nil
		},
	}
	normal := Tool{
		Name: "go", Label: "go", Parameters: ai.Object(nil),
		Execute: func(context.Context, ToolCall, UpdateFunc) (ToolResult, error) {
			return Text("continuing"), nil
		},
	}

	// Mixed batch: one tool cannot unilaterally end a run.
	stream, calls := scriptedStream(
		toolCallMsg(ai.StopToolUse, call("1", "stop", `{}`), call("2", "go", `{}`)),
		ai.Message{Role: ai.RoleAssistant, StopReason: ai.StopEnd, Content: []ai.Content{ai.TextContent("ok")}},
	)
	c := &Context{Tools: []Tool{terminating, normal}}
	Run(context.Background(), []ai.Message{ai.UserMessage("go")}, c,
		LoopConfig{Model: ai.ModelFlash}, stream, func(Event) {})
	if *calls < 2 {
		t.Errorf("mixed batch stopped the run after %d call(s); terminate needs unanimity", *calls)
	}

	// Unanimous batch: the run stops without another provider request.
	stream2, calls2 := scriptedStream(
		toolCallMsg(ai.StopToolUse, call("1", "stop", `{}`)),
		ai.Message{Role: ai.RoleAssistant, StopReason: ai.StopEnd, Content: []ai.Content{ai.TextContent("extra")}},
	)
	c2 := &Context{Tools: []Tool{terminating}}
	Run(context.Background(), []ai.Message{ai.UserMessage("go")}, c2,
		LoopConfig{Model: ai.ModelFlash}, stream2, func(Event) {})
	if *calls2 != 1 {
		t.Errorf("unanimous terminate made %d provider calls, want 1", *calls2)
	}
}

func TestSequentialToolForcesWholeBatchSequential(t *testing.T) {
	var active, maxActive int
	var mu sync.Mutex
	mk := func(name string, mode ExecutionMode) Tool {
		return Tool{
			Name: name, Label: name, Parameters: ai.Object(nil), ExecutionMode: mode,
			Execute: func(context.Context, ToolCall, UpdateFunc) (ToolResult, error) {
				mu.Lock()
				active++
				if active > maxActive {
					maxActive = active
				}
				mu.Unlock()
				time.Sleep(20 * time.Millisecond)
				mu.Lock()
				active--
				mu.Unlock()
				return Text("ok"), nil
			},
		}
	}

	stream, _ := scriptedStream(
		toolCallMsg(ai.StopToolUse, call("1", "par", `{}`), call("2", "seq", `{}`)),
		ai.Message{Role: ai.RoleAssistant, StopReason: ai.StopEnd, Content: []ai.Content{ai.TextContent("ok")}},
	)
	c := &Context{Tools: []Tool{mk("par", ModeParallel), mk("seq", ModeSequential)}}
	Run(context.Background(), []ai.Message{ai.UserMessage("go")}, c,
		LoopConfig{Model: ai.ModelFlash, ToolExecution: ModeParallel}, stream, func(Event) {})

	mu.Lock()
	defer mu.Unlock()
	if maxActive != 1 {
		t.Errorf("max concurrent tools = %d, want 1: one sequential tool forces the whole batch", maxActive)
	}
}

func TestUnknownToolProducesErrorResultAndContinues(t *testing.T) {
	stream, calls := scriptedStream(
		toolCallMsg(ai.StopToolUse, call("1", "nope", `{}`)),
		ai.Message{Role: ai.RoleAssistant, StopReason: ai.StopEnd, Content: []ai.Content{ai.TextContent("recovered")}},
	)
	c := &Context{}
	msgs := Run(context.Background(), []ai.Message{ai.UserMessage("go")}, c,
		LoopConfig{Model: ai.ModelFlash}, stream, func(Event) {})

	if *calls != 2 {
		t.Errorf("provider calls = %d, want 2: an unknown tool is recoverable", *calls)
	}
	var found bool
	for _, m := range msgs {
		if m.Role == ai.RoleToolResult && m.IsError && strings.Contains(m.Text(), "not found") {
			found = true
		}
	}
	if !found {
		t.Error("expected an error tool result naming the missing tool")
	}
}

func TestPanickingToolBecomesErrorResult(t *testing.T) {
	boom := Tool{
		Name: "boom", Label: "boom", Parameters: ai.Object(nil),
		Execute: func(context.Context, ToolCall, UpdateFunc) (ToolResult, error) {
			panic("tool exploded")
		},
	}
	stream, _ := scriptedStream(
		toolCallMsg(ai.StopToolUse, call("1", "boom", `{}`)),
		ai.Message{Role: ai.RoleAssistant, StopReason: ai.StopEnd, Content: []ai.Content{ai.TextContent("ok")}},
	)
	c := &Context{Tools: []Tool{boom}}

	// The assertion is that this returns at all: a panicking tool must not take
	// the process down with it.
	msgs := Run(context.Background(), []ai.Message{ai.UserMessage("go")}, c,
		LoopConfig{Model: ai.ModelFlash}, stream, func(Event) {})

	var found bool
	for _, m := range msgs {
		if m.Role == ai.RoleToolResult && m.IsError && strings.Contains(m.Text(), "panicked") {
			found = true
		}
	}
	if !found {
		t.Error("panicking tool should produce an error result")
	}
}

func TestErrorTurnEndsRunImmediately(t *testing.T) {
	stream, calls := scriptedStream(
		ai.Message{Role: ai.RoleAssistant, StopReason: ai.StopError, ErrorMessage: "upstream died"},
		ai.Message{Role: ai.RoleAssistant, StopReason: ai.StopEnd, Content: []ai.Content{ai.TextContent("never")}},
	)
	c := &Context{}
	var sawAgentEnd bool
	Run(context.Background(), []ai.Message{ai.UserMessage("go")}, c,
		LoopConfig{Model: ai.ModelFlash}, stream, func(ev Event) {
			if ev.Type == EventAgentEnd {
				sawAgentEnd = true
			}
		})
	if *calls != 1 {
		t.Errorf("provider calls = %d, want 1: a failed turn ends the run", *calls)
	}
	if !sawAgentEnd {
		t.Error("a failed run must still emit agent_end so consumers never see a broken stream")
	}
}

func TestSteeringInjectedBeforeNextTurn(t *testing.T) {
	stream, _ := scriptedStream(
		toolCallMsg(ai.StopToolUse, call("1", "noop", `{}`)),
		ai.Message{Role: ai.RoleAssistant, StopReason: ai.StopEnd, Content: []ai.Content{ai.TextContent("ok")}},
	)
	noop := Tool{
		Name: "noop", Label: "noop", Parameters: ai.Object(nil),
		Execute: func(context.Context, ToolCall, UpdateFunc) (ToolResult, error) { return Text("ok"), nil },
	}

	delivered := false
	c := &Context{Tools: []Tool{noop}}
	msgs := Run(context.Background(), []ai.Message{ai.UserMessage("go")}, c,
		LoopConfig{
			Model: ai.ModelFlash,
			GetSteeringMessages: func() []ai.Message {
				if delivered {
					return nil
				}
				delivered = true
				return []ai.Message{ai.UserMessage("actually, stop")}
			},
		}, stream, func(Event) {})

	var found bool
	for _, m := range msgs {
		if m.Role == ai.RoleUser && m.Text() == "actually, stop" {
			found = true
		}
	}
	if !found {
		t.Error("steering message was never injected into the transcript")
	}
}

func TestFollowUpRestartsAfterStop(t *testing.T) {
	stream, calls := scriptedStream(
		ai.Message{Role: ai.RoleAssistant, StopReason: ai.StopEnd, Content: []ai.Content{ai.TextContent("first")}},
		ai.Message{Role: ai.RoleAssistant, StopReason: ai.StopEnd, Content: []ai.Content{ai.TextContent("second")}},
	)
	delivered := false
	c := &Context{}
	Run(context.Background(), []ai.Message{ai.UserMessage("go")}, c,
		LoopConfig{
			Model: ai.ModelFlash,
			GetFollowUpMessages: func() []ai.Message {
				if delivered {
					return nil
				}
				delivered = true
				return []ai.Message{ai.UserMessage("one more thing")}
			},
		}, stream, func(Event) {})

	if *calls != 2 {
		t.Errorf("provider calls = %d, want 2: a follow-up should restart the loop", *calls)
	}
}

func TestPrepareNextTurnSwapsModelAndShouldStopHalts(t *testing.T) {
	stream, calls := scriptedStream(
		toolCallMsg(ai.StopToolUse, call("1", "noop", `{}`)),
		ai.Message{Role: ai.RoleAssistant, StopReason: ai.StopEnd, Content: []ai.Content{ai.TextContent("never")}},
	)
	noop := Tool{
		Name: "noop", Label: "noop", Parameters: ai.Object(nil),
		Execute: func(context.Context, ToolCall, UpdateFunc) (ToolResult, error) { return Text("ok"), nil },
	}

	var swapped string
	c := &Context{Tools: []Tool{noop}}
	Run(context.Background(), []ai.Message{ai.UserMessage("go")}, c,
		LoopConfig{
			Model: ai.ModelFlash,
			PrepareNextTurn: func(tc TurnContext) *TurnUpdate {
				swapped = ai.ModelPro
				return &TurnUpdate{Model: ai.ModelPro}
			},
			ShouldStopAfterTurn: func(TurnContext) bool { return true },
		}, stream, func(Event) {})

	if swapped != ai.ModelPro {
		t.Error("PrepareNextTurn was never called: the between-turns seam is missing")
	}
	if *calls != 1 {
		t.Errorf("provider calls = %d, want 1: ShouldStopAfterTurn should halt before another request", *calls)
	}
}

func TestValidationErrorListsEveryProblemAtOnce(t *testing.T) {
	schema := ai.Object(map[string]*ai.Schema{
		"path":  ai.Str("path"),
		"count": ai.Int("count"),
	}, "path", "count")

	err := ValidateArguments(schema, json.RawMessage(`{"count":"not a number"}`))
	if err == nil {
		t.Fatal("expected a validation error")
	}
	msg := err.Error()
	// Both problems in one message: each rejected call costs a round trip and
	// breaks the prompt cache, so the model must be able to fix everything at
	// once.
	if !strings.Contains(msg, "path") {
		t.Errorf("missing-required problem not reported: %s", msg)
	}
	if !strings.Contains(msg, "count") {
		t.Errorf("wrong-type problem not reported: %s", msg)
	}
}

func TestValidationAcceptsIntegralFloats(t *testing.T) {
	schema := ai.Object(map[string]*ai.Schema{"n": ai.Int("n")}, "n")
	if err := ValidateArguments(schema, json.RawMessage(`{"n":42}`)); err != nil {
		t.Errorf("integral JSON number rejected: %v", err)
	}
	if err := ValidateArguments(schema, json.RawMessage(`{"n":4.5}`)); err == nil {
		t.Error("fractional value accepted for an integer field")
	}
}

func TestHealArgumentsUnwrapsAndAliases(t *testing.T) {
	heal := HealArguments(map[string]string{"file_path": "path"})

	// The whole object delivered as a JSON string.
	got := heal(json.RawMessage(`"{\"path\":\"a.go\"}"`))
	var m map[string]any
	if err := json.Unmarshal(got, &m); err != nil {
		t.Fatalf("unwrap failed: %v (got %s)", err, got)
	}
	if m["path"] != "a.go" {
		t.Errorf("unwrapped to %v, want path=a.go", m)
	}

	// Alias renaming.
	got = heal(json.RawMessage(`{"file_path":"b.go"}`))
	m = nil
	_ = json.Unmarshal(got, &m)
	if m["path"] != "b.go" {
		t.Errorf("alias not renamed: %v", m)
	}
	if _, stillThere := m["file_path"]; stillThere {
		t.Error("alias key should be removed after renaming")
	}

	// A canonical key already present wins over the alias.
	got = heal(json.RawMessage(`{"file_path":"alias.go","path":"real.go"}`))
	m = nil
	_ = json.Unmarshal(got, &m)
	if m["path"] != "real.go" {
		t.Errorf("alias overwrote the canonical key: %v", m)
	}
}

func TestRetryableErrorClassification(t *testing.T) {
	for msg, want := range map[string]bool{
		"provider error (503): unavailable":  true,
		"rate limited (429): slow down":      true,
		"stream ended before completion":     true,
		"authentication failed (401): bad":   false,
		"insufficient balance (402): top up": false,
		"provider error (400): invalid":      false,
		"":                                   false,
	} {
		if got := RetryableError(msg); got != want {
			t.Errorf("RetryableError(%q) = %v, want %v", msg, got, want)
		}
	}
}

func TestAgentRejectsConcurrentPrompt(t *testing.T) {
	release := make(chan struct{})
	blocking := func(_ ai.Context, _ ai.StreamOptions) *ai.Stream {
		st := ai.NewStream()
		go func() {
			<-release
			st.Finish(&ai.Message{Role: ai.RoleAssistant, StopReason: ai.StopEnd,
				Content: []ai.Content{ai.TextContent("done")}})
		}()
		return st
	}

	a := New(&Context{}, LoopConfig{Model: ai.ModelFlash}, blocking)
	a.Retry = SessionRetry{}

	go func() { _, _ = a.Prompt(context.Background(), "first") }()
	// Wait for the run to be in flight.
	for i := 0; i < 100 && !a.Running(); i++ {
		time.Sleep(time.Millisecond)
	}

	if _, err := a.Prompt(context.Background(), "second"); err == nil {
		t.Error("a concurrent Prompt should fail: mid-run input belongs in Steer or FollowUp")
	}
	close(release)
}

func TestAgentSessionRetryReplaysTransientFailure(t *testing.T) {
	var mu sync.Mutex
	n := 0
	flaky := func(_ ai.Context, _ ai.StreamOptions) *ai.Stream {
		mu.Lock()
		n++
		attempt := n
		mu.Unlock()

		st := ai.NewStream()
		go func() {
			if attempt == 1 {
				st.Finish(&ai.Message{Role: ai.RoleAssistant, StopReason: ai.StopError,
					ErrorMessage: "provider error (503): unavailable"})
				return
			}
			st.Finish(&ai.Message{Role: ai.RoleAssistant, StopReason: ai.StopEnd,
				Content: []ai.Content{ai.TextContent("recovered")}})
		}()
		return st
	}

	a := New(&Context{}, LoopConfig{Model: ai.ModelFlash}, flaky)
	a.Retry = SessionRetry{MaxAttempts: 2, BaseDelay: time.Millisecond}

	msgs, err := a.Prompt(context.Background(), "go")
	if err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	mu.Lock()
	attempts := n
	mu.Unlock()
	if attempts != 2 {
		t.Errorf("provider attempts = %d, want 2 (one retry)", attempts)
	}

	last := msgs[len(msgs)-1]
	if last.StopReason != ai.StopEnd || last.Text() != "recovered" {
		t.Errorf("final message = %+v, want the recovered response", last)
	}
}

func TestAgentSessionRetrySkipsFatalErrors(t *testing.T) {
	var mu sync.Mutex
	n := 0
	fatal := func(_ ai.Context, _ ai.StreamOptions) *ai.Stream {
		mu.Lock()
		n++
		mu.Unlock()
		st := ai.NewStream()
		go func() {
			st.Finish(&ai.Message{Role: ai.RoleAssistant, StopReason: ai.StopError,
				ErrorMessage: "insufficient balance (402): top up"})
		}()
		return st
	}

	a := New(&Context{}, LoopConfig{Model: ai.ModelFlash}, fatal)
	a.Retry = SessionRetry{MaxAttempts: 3, BaseDelay: time.Millisecond}

	if _, err := a.Prompt(context.Background(), "go"); err != nil {
		t.Fatalf("Prompt: %v", err)
	}
	mu.Lock()
	defer mu.Unlock()
	if n != 1 {
		t.Errorf("provider attempts = %d, want 1: a billing failure must not be retried", n)
	}
}

func TestEventSequenceIsWellFormed(t *testing.T) {
	stream, _ := scriptedStream(
		ai.Message{Role: ai.RoleAssistant, StopReason: ai.StopEnd, Content: []ai.Content{ai.TextContent("hi")}},
	)
	var seq []EventType
	Run(context.Background(), []ai.Message{ai.UserMessage("go")}, &Context{},
		LoopConfig{Model: ai.ModelFlash}, stream, func(ev Event) {
			seq = append(seq, ev.Type)
		})

	if len(seq) == 0 || seq[0] != EventAgentStart {
		t.Fatalf("first event = %v, want agent_start", seq)
	}
	if seq[len(seq)-1] != EventAgentEnd {
		t.Errorf("last event = %v, want agent_end", seq[len(seq)-1])
	}
	if err := checkPaired(seq, EventTurnStart, EventTurnEnd); err != nil {
		t.Error(err)
	}
}

func checkPaired(seq []EventType, start, end EventType) error {
	depth := 0
	for _, e := range seq {
		switch e {
		case start:
			depth++
		case end:
			depth--
			if depth < 0 {
				return fmt.Errorf("%s without a matching %s", end, start)
			}
		}
	}
	if depth != 0 {
		return fmt.Errorf("%d unclosed %s event(s)", depth, start)
	}
	return nil
}
