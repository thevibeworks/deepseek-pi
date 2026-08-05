package ai

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func assistant(stop StopReason, content ...Content) Message {
	return Message{Role: RoleAssistant, StopReason: stop, Content: content}
}

func toolCall(id, name, args string) Content {
	return Content{Type: ContentToolCall, ID: id, Name: name, Arguments: json.RawMessage(args)}
}

func toolResult(id, text string) Message {
	return Message{
		Role: RoleToolResult, ToolCallID: id, ToolName: "t",
		Content: []Content{TextContent(text)},
	}
}

func TestEncodeMessagesMergesConsecutiveToolResults(t *testing.T) {
	// A parallel tool batch produces several tool results in a row. The
	// Messages API requires them in ONE user message; emitting one message per
	// result is the classic way a parallel batch gets rejected.
	msgs := []Message{
		UserMessage("do two things"),
		assistant(StopToolUse, toolCall("a", "t", `{}`), toolCall("b", "t", `{}`)),
		toolResult("a", "first"),
		toolResult("b", "second"),
	}
	got := encodeMessages(msgs)

	if len(got) != 3 {
		t.Fatalf("got %d wire messages, want 3: %+v", len(got), got)
	}
	if got[2].Role != "user" {
		t.Fatalf("results message role = %q, want user", got[2].Role)
	}
	if len(got[2].Content) != 2 {
		t.Fatalf("got %d tool_result blocks in one message, want 2", len(got[2].Content))
	}
	for i, want := range []string{"a", "b"} {
		if got[2].Content[i].ToolUseID != want {
			t.Errorf("block %d tool_use_id = %q, want %q", i, got[2].Content[i].ToolUseID, want)
		}
	}
}

func TestEncodeMessagesDropsCrashedTurns(t *testing.T) {
	// A turn that ended in error must never be replayed: it poisons the
	// conversation and providers reject some of its shapes outright.
	msgs := []Message{
		UserMessage("hello"),
		{
			Role: RoleAssistant, StopReason: StopError, ErrorMessage: "boom",
			Content: []Content{TextContent("partial")},
		},
		UserMessage("try again"),
	}
	got := encodeMessages(msgs)

	if len(got) != 2 {
		t.Fatalf("got %d wire messages, want 2 (crashed turn dropped)", len(got))
	}
	for _, m := range got {
		if m.Role == "assistant" {
			t.Fatal("crashed assistant turn leaked into the payload")
		}
	}
}

func TestEncodeMessagesSynthesizesMissingToolResults(t *testing.T) {
	// Every tool_use must be answered. An unanswered call (crash, restart)
	// gets a synthetic result so the payload stays well-formed.
	msgs := []Message{
		UserMessage("go"),
		assistant(StopToolUse, toolCall("a", "t", `{}`), toolCall("orphan", "t", `{}`)),
		toolResult("a", "done"),
	}
	got := encodeMessages(msgs)

	var found bool
	for _, m := range got {
		for _, b := range m.Content {
			if b.Type == "tool_result" && b.ToolUseID == "orphan" {
				found = true
				if !b.IsError {
					t.Error("synthesized result should be flagged as an error")
				}
			}
		}
	}
	if !found {
		t.Fatal("orphaned tool call got no synthesized result")
	}
}

func TestEncodeMessagesPreservesThinkingSignature(t *testing.T) {
	// DeepSeek ties the thinking signature to the response id and expects it
	// back verbatim on replay.
	msgs := []Message{
		UserMessage("hi"),
		assistant(StopEnd,
			Content{Type: ContentThinking, Thinking: "reasoning", Signature: "sig-123"},
			TextContent("answer"),
		),
	}
	got := encodeMessages(msgs)

	block := got[1].Content[0]
	if block.Type != "thinking" {
		t.Fatalf("first block type = %q, want thinking", block.Type)
	}
	if block.Signature == nil || *block.Signature != "sig-123" {
		t.Errorf("signature = %v, want sig-123", block.Signature)
	}
	if block.Thinking == nil || *block.Thinking != "reasoning" {
		t.Errorf("thinking = %v, want reasoning", block.Thinking)
	}
}

func TestEncodeMessagesKeepsEmptyThinkingField(t *testing.T) {
	// DeepSeek routinely returns a thinking block whose text is empty: only a
	// signature delta arrives. Dropping the empty "thinking" field on replay
	// makes the API reject the whole request with
	//   messages[N].content: missing field `thinking`
	// which kills the run mid-task. Observed live, not hypothetical.
	msgs := []Message{
		UserMessage("hi"),
		assistant(StopToolUse,
			Content{Type: ContentThinking, Thinking: "", Signature: "resp-id"},
			toolCall("a", "t", `{}`),
		),
		toolResult("a", "ok"),
	}
	encoded, err := json.Marshal(encodeMessages(msgs))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !strings.Contains(string(encoded), `"thinking":""`) {
		t.Errorf("empty thinking field was omitted; the API requires it:\n%s", encoded)
	}
	if !strings.Contains(string(encoded), `"signature":"resp-id"`) {
		t.Errorf("signature missing:\n%s", encoded)
	}
}

func TestEncodeMessagesEmptyToolResultGetsPlaceholder(t *testing.T) {
	// The API rejects an empty tool_result content field.
	msgs := []Message{
		UserMessage("go"),
		assistant(StopToolUse, toolCall("a", "t", `{}`)),
		{Role: RoleToolResult, ToolCallID: "a", ToolName: "t"},
	}
	got := encodeMessages(msgs)
	last := got[len(got)-1]
	if last.Content[0].Content == "" {
		t.Error("empty tool result content was sent as empty; the API rejects that")
	}
}

func TestUsageNormalization(t *testing.T) {
	// The endpoint reports input_tokens as the cache MISS and cache_read
	// separately. Usage.Input is the full prompt, so they must be summed.
	// Getting this backwards understates prompt size and overstates hit rate.
	a := newAccumulator(ModelFlash)
	a.applyUsage(wireUsage{InputTokens: 23_000, CacheReadTokens: 111_000, OutputTokens: 500})

	if got, want := a.msg.Usage.Input, 134_000; got != want {
		t.Errorf("Input = %d, want %d (miss + cache read)", got, want)
	}
	if got, want := a.msg.Usage.CacheMiss(), 23_000; got != want {
		t.Errorf("CacheMiss = %d, want %d", got, want)
	}
	if rate := a.msg.Usage.CacheHitRate(); rate < 0.82 || rate > 0.83 {
		t.Errorf("CacheHitRate = %.3f, want ~0.828", rate)
	}
}

func TestPriceUsesCacheRateForCachedTokens(t *testing.T) {
	m := MustLookup(ModelFlash)
	u := Usage{Input: 100_000, CacheRead: 90_000, Output: 1_000}
	c := m.Price(u)

	// 10k miss @ 0.14/M + 90k cached @ 0.0028/M + 1k out @ 0.28/M
	wantInput := 10_000 * 0.14 / 1e6
	wantCache := 90_000 * 0.0028 / 1e6
	wantOutput := 1_000 * 0.28 / 1e6
	if !approx(c.Input, wantInput) || !approx(c.CacheRead, wantCache) || !approx(c.Output, wantOutput) {
		t.Errorf("cost = %+v, want input=%.8f cache=%.8f output=%.8f", c, wantInput, wantCache, wantOutput)
	}
	if !approx(c.Total, wantInput+wantCache+wantOutput) {
		t.Errorf("total = %.8f, want %.8f", c.Total, wantInput+wantCache+wantOutput)
	}
}

func approx(a, b float64) bool {
	d := a - b
	return d < 1e-9 && d > -1e-9
}

func TestBuildRequestEffortAndThinking(t *testing.T) {
	c := &Client{APIKey: "k"}
	req := c.buildRequest(Context{Messages: []Message{UserMessage("hi")}},
		StreamOptions{Model: ModelFlash, Effort: EffortHigh})

	if req.Thinking == nil || req.Thinking.Type != "enabled" {
		t.Errorf("thinking = %+v, want enabled", req.Thinking)
	}
	if req.OutputConfig == nil || req.OutputConfig.Effort != "high" {
		t.Errorf("output_config = %+v, want effort high", req.OutputConfig)
	}

	off := c.buildRequest(Context{Messages: []Message{UserMessage("hi")}},
		StreamOptions{Model: ModelFlash})
	if off.Thinking == nil || off.Thinking.Type != "disabled" {
		t.Errorf("thinking with no effort = %+v, want disabled", off.Thinking)
	}
	if off.OutputConfig != nil {
		t.Error("output_config should be absent when thinking is disabled")
	}
}

func TestBuildRequestClampsMaxTokens(t *testing.T) {
	c := &Client{APIKey: "k"}
	req := c.buildRequest(Context{Messages: []Message{UserMessage("hi")}},
		StreamOptions{Model: ModelFlash, MaxTokens: 99_999_999})
	if req.MaxTokens != MustLookup(ModelFlash).MaxTokens {
		t.Errorf("MaxTokens = %d, want clamp to %d", req.MaxTokens, MustLookup(ModelFlash).MaxTokens)
	}
}

func TestScanSSE(t *testing.T) {
	raw := "event: message_start\n" +
		"data: {\"type\":\"message_start\"}\n" +
		"\n" +
		": keep-alive comment\n" +
		"event: content_block_delta\n" +
		"data: {\"type\":\"content_block_delta\"}\n" +
		"\n"

	var names []string
	err := scanSSE(strings.NewReader(raw), func(ev sseEvent) bool {
		names = append(names, ev.Name)
		return true
	})
	if err != nil {
		t.Fatalf("scanSSE: %v", err)
	}
	want := []string{"message_start", "content_block_delta"}
	if len(names) != len(want) {
		t.Fatalf("got events %v, want %v", names, want)
	}
	for i := range want {
		if names[i] != want[i] {
			t.Errorf("event %d = %q, want %q", i, names[i], want[i])
		}
	}
}

func TestAccumulatorInfersToolUseStopReason(t *testing.T) {
	// A stream that ends without an explicit stop_reason but carries tool calls
	// must not be reported as a normal stop, or the loop ends early and the
	// tools never run.
	a := newAccumulator(ModelFlash)
	a.blocks = []Content{toolCall("x", "t", `{}`)}
	final := a.finalize()
	if final.StopReason != StopToolUse {
		t.Errorf("StopReason = %s, want %s", final.StopReason, StopToolUse)
	}
}

func TestErrorStreamHonoursContract(t *testing.T) {
	st := ErrorStream(ModelFlash, "kaboom")
	for range st.Events() {
	}
	final := st.Result()
	if final == nil {
		t.Fatal("Result returned nil; the contract says never nil")
	}
	if final.StopReason != StopError || final.ErrorMessage != "kaboom" {
		t.Errorf("got %s/%q, want error/kaboom", final.StopReason, final.ErrorMessage)
	}
}

func TestStreamCloseDoesNotPanicProducer(t *testing.T) {
	// A consumer abandoning the stream must not crash a producer mid-push.
	st := NewStream()
	done := make(chan struct{})
	go func() {
		defer close(done)
		for i := 0; i < 1000; i++ {
			if !st.Push(Event{Type: EventTextDelta, Delta: "x"}) {
				break
			}
		}
		st.Finish(&Message{Role: RoleAssistant, StopReason: StopEnd})
	}()

	st.Close()
	<-done
	if st.Result() == nil {
		t.Fatal("Result returned nil after Close")
	}
}

func TestLookupResolvesClaudeAliases(t *testing.T) {
	// The endpoint remaps Claude names server-side; cost must land on the model
	// that actually ran.
	for name, want := range map[string]string{
		"claude-opus-4-6":   ModelPro,
		"claude-sonnet-4-5": ModelFlash,
		"claude-haiku-4-5":  ModelFlash,
		ModelFlash:          ModelFlash,
	} {
		m, ok := Lookup(name)
		if !ok {
			t.Errorf("Lookup(%q) failed", name)
			continue
		}
		if m.ID != want {
			t.Errorf("Lookup(%q).ID = %s, want %s", name, m.ID, want)
		}
	}
	if _, ok := Lookup("gpt-4"); ok {
		t.Error("Lookup should not resolve unrelated model names")
	}
}

func TestRetryPolicy(t *testing.T) {
	p := RetryPolicy{MaxRetries: 3, BaseDelay: time.Millisecond, MaxDelay: time.Second, MaxRetryAfter: 60 * time.Second}

	if _, ok := p.next(0, 400, ""); ok {
		t.Error("400 should not retry")
	}
	if _, ok := p.next(0, 500, ""); !ok {
		t.Error("500 should retry")
	}
	if _, ok := p.next(0, 429, ""); !ok {
		t.Error("429 should retry")
	}
	if _, ok := p.next(3, 500, ""); ok {
		t.Error("should stop at MaxRetries")
	}
	// A server asking for longer than the cap fails fast instead of parking the
	// session, so the user sees it.
	if _, ok := p.next(0, 429, "600"); ok {
		t.Error("a Retry-After beyond the cap should not retry")
	}
	if d, ok := p.next(0, 429, "5"); !ok || d != 5*time.Second {
		t.Errorf("Retry-After 5 gave %v/%v, want 5s/true", d, ok)
	}
}

func TestDefaultTransportRetryIsZero(t *testing.T) {
	// Two retry layers must not multiply. The transport stays quiet; the
	// session layer owns policy because only it can classify a failure.
	if got := DefaultRetryPolicy().MaxRetries; got != 0 {
		t.Errorf("default transport MaxRetries = %d, want 0", got)
	}
}

func TestEncodeMessagesDropsOrphanedToolResults(t *testing.T) {
	// The mirror of the orphaned-call rule, and the API is just as strict:
	// "Each tool_result block must have a corresponding tool_use block in the
	// previous message." A transcript can acquire a stray result by being
	// truncated or rewritten above this layer — compaction is the obvious
	// source — and one stray block fails the ENTIRE request, not just itself.
	msgs := []Message{
		UserMessage("go"),
		// A result whose call is nowhere in the transcript.
		toolResult("ghost", "output from a call that was cut away"),
		assistant(StopToolUse, toolCall("real", "t", `{}`)),
		toolResult("real", "genuine output"),
	}
	got := encodeMessages(msgs)

	for _, m := range got {
		for _, b := range m.Content {
			if b.Type == "tool_result" && b.ToolUseID == "ghost" {
				t.Error("orphaned tool result was sent; the API rejects the whole request")
			}
		}
	}
	// The legitimate pair must survive untouched.
	var foundReal bool
	for _, m := range got {
		for _, b := range m.Content {
			if b.Type == "tool_result" && b.ToolUseID == "real" {
				foundReal = true
			}
		}
	}
	if !foundReal {
		t.Error("dropping the orphan also dropped a valid tool result")
	}
}

func TestEncodeMessagesDropsResultThatPrecedesItsCall(t *testing.T) {
	// Ordering matters, not just presence: a result must FOLLOW its call.
	msgs := []Message{
		UserMessage("go"),
		toolResult("x", "early"),
		assistant(StopToolUse, toolCall("x", "t", `{}`)),
	}
	got := encodeMessages(msgs)
	for _, m := range got {
		for _, b := range m.Content {
			if b.Type == "tool_result" {
				t.Errorf("a result appearing before its call was sent: %+v", b)
			}
		}
	}
}
