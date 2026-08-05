package ai

import (
	"context"
	"encoding/json"
	"os"
	"testing"
	"time"
)

// requireLive gates the tests that hit the real API.
//
// A key alone is NOT enough. Developers keep DEEPSEEK_API_KEY exported for
// ordinary use, and a plain `go test ./...` silently spending money is a bad
// surprise. DEEPSEEK_PI_LIVE=1 is the deliberate opt-in; `make test-live` sets
// it. The tests still compile on every run, so they cannot rot.
func requireLive(t *testing.T) string {
	t.Helper()
	if os.Getenv("DEEPSEEK_PI_LIVE") == "" {
		t.Skip("set DEEPSEEK_PI_LIVE=1 to run tests against the real API (they cost money)")
	}
	key := os.Getenv("DEEPSEEK_API_KEY")
	if key == "" {
		t.Skip("DEEPSEEK_API_KEY not set; skipping live API test")
	}
	return key
}

func liveClient(t *testing.T) *Client {
	t.Helper()
	return NewClient(requireLive(t))
}

func TestLiveStreamText(t *testing.T) {
	c := liveClient(t)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	st := c.Stream(ctx, Context{
		SystemPrompt: "You are terse. Answer with a single word.",
		Messages:     []Message{UserMessage("What is the capital of France?")},
	}, StreamOptions{Model: ModelFlash, MaxTokens: 64})

	var sawStart, sawDelta bool
	for ev := range st.Events() {
		switch ev.Type {
		case EventStart:
			sawStart = true
		case EventTextDelta:
			sawDelta = true
		}
	}
	final := st.Result()

	if final.StopReason == StopError {
		t.Fatalf("stream failed: %s", final.ErrorMessage)
	}
	if !sawStart || !sawDelta {
		t.Errorf("missing events: start=%v delta=%v", sawStart, sawDelta)
	}
	if final.Text() == "" {
		t.Error("empty response text")
	}
	if final.Usage.Input == 0 || final.Usage.Output == 0 {
		t.Errorf("usage not populated: %+v", final.Usage)
	}
	if final.Usage.Cost.Total <= 0 {
		t.Errorf("cost not computed: %+v", final.Usage.Cost)
	}
	t.Logf("text=%q usage=%+v cost=$%.6f", final.Text(), final.Usage, final.Usage.Cost.Total)
}

func TestLiveToolCallRoundTrip(t *testing.T) {
	c := liveClient(t)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	tools := []Tool{{
		Name:        "get_weather",
		Description: "Get the current weather for a city.",
		Parameters:  Object(map[string]*Schema{"city": Str("City name")}, "city"),
	}}

	// Turn 1: expect a tool call.
	st := c.Stream(ctx, Context{
		SystemPrompt: "You are terse. Use tools when they apply.",
		Messages:     []Message{UserMessage("What is the weather in Paris? Use the tool.")},
		Tools:        tools,
	}, StreamOptions{Model: ModelFlash, MaxTokens: 1024, Effort: EffortLow})

	for range st.Events() {
	}
	first := st.Result()
	if first.StopReason == StopError {
		t.Fatalf("turn 1 failed: %s", first.ErrorMessage)
	}
	calls := first.ToolCalls()
	if len(calls) == 0 {
		t.Fatalf("expected a tool call, got stopReason=%s text=%q", first.StopReason, first.Text())
	}
	if first.StopReason != StopToolUse {
		t.Errorf("stopReason = %s, want %s", first.StopReason, StopToolUse)
	}
	var args map[string]any
	if err := json.Unmarshal(calls[0].Arguments, &args); err != nil {
		t.Fatalf("tool arguments not valid JSON: %v (raw=%s)", err, calls[0].Arguments)
	}
	if args["city"] == nil {
		t.Errorf("expected city argument, got %v", args)
	}

	// Turn 2: replay the assistant message (thinking blocks and signature
	// included) plus a tool result. This is the round trip the agent loop
	// performs on every turn; if it breaks, nothing above it works.
	msgs := []Message{
		UserMessage("What is the weather in Paris? Use the tool."),
		*first,
		{
			Role:       RoleToolResult,
			ToolCallID: calls[0].ID,
			ToolName:   calls[0].Name,
			Content:    []Content{TextContent("18C, sunny")},
			Timestamp:  time.Now().UnixMilli(),
		},
	}
	st2 := c.Stream(ctx, Context{
		SystemPrompt: "You are terse. Use tools when they apply.",
		Messages:     msgs,
		Tools:        tools,
	}, StreamOptions{Model: ModelFlash, MaxTokens: 1024, Effort: EffortLow})

	for range st2.Events() {
	}
	second := st2.Result()
	if second.StopReason == StopError {
		t.Fatalf("turn 2 (replay) failed: %s", second.ErrorMessage)
	}
	if second.Text() == "" {
		t.Error("expected a text answer after the tool result")
	}
	t.Logf("answer=%q thinkingBlocks=%d", second.Text(), countThinking(*first))
}

func countThinking(m Message) int {
	n := 0
	for _, c := range m.Content {
		if c.Type == ContentThinking {
			n++
		}
	}
	return n
}

func TestLiveBadKeyIsStreamError(t *testing.T) {
	requireLive(t)
	c := NewClient("sk-obviously-invalid")
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	st := c.Stream(ctx, Context{Messages: []Message{UserMessage("hi")}},
		StreamOptions{Model: ModelFlash, MaxTokens: 16})
	for range st.Events() {
	}
	final := st.Result()

	// The contract: transport failures arrive as a normal final message, never
	// as a panic or a nil result.
	if final.StopReason != StopError {
		t.Fatalf("stopReason = %s, want %s", final.StopReason, StopError)
	}
	if final.ErrorMessage == "" {
		t.Error("expected an error message")
	}
	t.Logf("error surfaced as: %s", final.ErrorMessage)
}
