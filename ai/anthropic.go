package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Client talks to DeepSeek's Anthropic-compatible Messages endpoint.
//
// Why this protocol and not the OpenAI-style one: v4 emits DSML, an
// Anthropic-shaped invoke/parameter tool grammar, so this is the wire format
// the model is trained toward. It carries tool use, parallel tool calls,
// thinking blocks and cache-read reporting correctly. The OpenAI-compatible
// route brings extra hazards (DSML envelope leakage into visible content,
// reasoning_content replay rules, tool_choice/thinking conflicts) for no
// demonstrated benefit, so deepseek-pi keeps exactly one state machine.
type Client struct {
	APIKey  string
	BaseURL string
	HTTP    *http.Client
	// Retry governs transport-level retries. Zero value means no retries,
	// matching pi's default: the session layer above owns retry policy because
	// only it can see whether a failure is worth replaying a whole turn for.
	Retry RetryPolicy
	// UserAgent identifies the harness in request logs.
	UserAgent string
}

// NewClient builds a client with sane transport defaults.
func NewClient(apiKey string) *Client {
	return &Client{
		APIKey:  apiKey,
		BaseURL: DefaultBaseURL,
		HTTP: &http.Client{
			// Generous: a max-effort pro turn on a large context legitimately
			// runs for minutes. The per-request context governs cancellation.
			Timeout: 30 * time.Minute,
		},
		Retry:     DefaultRetryPolicy(),
		UserAgent: "deepseek-pi",
	}
}

// StreamFunc adapts the client to the StreamFunc contract with a bound context.
func (c *Client) StreamFunc(ctx context.Context) StreamFunc {
	return func(aiCtx Context, opts StreamOptions) *Stream {
		return c.Stream(ctx, aiCtx, opts)
	}
}

// ---------------------------------------------------------------- wire types

type wireRequest struct {
	Model        string        `json:"model"`
	MaxTokens    int           `json:"max_tokens"`
	Stream       bool          `json:"stream"`
	System       []wireBlock   `json:"system,omitempty"`
	Messages     []wireMessage `json:"messages"`
	Tools        []wireTool    `json:"tools,omitempty"`
	Thinking     *wireThinking `json:"thinking,omitempty"`
	OutputConfig *wireEffort   `json:"output_config,omitempty"`
	Temperature  *float64      `json:"temperature,omitempty"`
}

type wireThinking struct {
	Type string `json:"type"` // enabled | disabled
}

type wireEffort struct {
	Effort string `json:"effort"`
}

type wireTool struct {
	Name        string  `json:"name"`
	Description string  `json:"description"`
	InputSchema *Schema `json:"input_schema"`
}

type wireMessage struct {
	Role    string      `json:"role"`
	Content []wireBlock `json:"content"`
}

// wireBlock is every Anthropic content block shape in one struct. omitempty
// keeps each serialized form minimal and, more importantly, byte-stable across
// turns, which is what preserves the prompt cache.
type wireBlock struct {
	Type string `json:"type"`

	Text string `json:"text,omitempty"`

	// Thinking and Signature are pointers so a thinking block can carry an
	// EMPTY string and still serialize the field. DeepSeek routinely returns
	// thinking blocks whose text is empty (only a signature delta arrives), and
	// the API rejects the replay with "missing field `thinking`" if omitempty
	// drops it. A pointer omits only when nil, which is what distinguishes
	// "not a thinking block" from "a thinking block that said nothing".
	Thinking  *string `json:"thinking,omitempty"`
	Signature *string `json:"signature,omitempty"`

	// tool_use
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`

	// tool_result
	ToolUseID string `json:"tool_use_id,omitempty"`
	Content   string `json:"content,omitempty"`
	IsError   bool   `json:"is_error,omitempty"`
}

type wireUsage struct {
	InputTokens         int `json:"input_tokens"`
	OutputTokens        int `json:"output_tokens"`
	CacheReadTokens     int `json:"cache_read_input_tokens"`
	CacheCreationTokens int `json:"cache_creation_input_tokens"`
}

type wireStreamEvent struct {
	Type string `json:"type"`

	Message *struct {
		ID    string    `json:"id"`
		Model string    `json:"model"`
		Usage wireUsage `json:"usage"`
	} `json:"message,omitempty"`

	Index        int        `json:"index"`
	ContentBlock *wireBlock `json:"content_block,omitempty"`

	Delta *struct {
		Type         string          `json:"type"`
		Text         string          `json:"text"`
		Thinking     string          `json:"thinking"`
		Signature    string          `json:"signature"`
		PartialJSON  string          `json:"partial_json"`
		StopReason   string          `json:"stop_reason"`
		StopSequence json.RawMessage `json:"stop_sequence"`
	} `json:"delta,omitempty"`

	Usage *wireUsage `json:"usage,omitempty"`

	Error *struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

// ------------------------------------------------------------------ encoding

// encodeMessages converts a transcript to Anthropic wire messages.
//
// Three rules do real work here:
//
//  1. Crashed turns are scrubbed. An assistant message that ended in error or
//     abort never goes to the provider; replaying it poisons the conversation
//     and, on some turns, is rejected outright. Its tool calls are orphaned, so
//     rule 2 covers them.
//  2. Every tool_use must be answered. An orphaned call (crashed turn, or a
//     result lost to a restart) gets a synthesized "No result provided" result
//     so the payload is always well-formed.
//  3. Consecutive tool results collapse into ONE user message. The Messages API
//     requires that; emitting one user message per result is the single most
//     common way a parallel tool batch gets rejected.
func encodeMessages(msgs []Message) []wireMessage {
	out := make([]wireMessage, 0, len(msgs))

	// Results indexed by tool call id, so an assistant turn can check which of
	// its calls were actually answered.
	answered := make(map[string]bool, len(msgs))
	for _, m := range msgs {
		if m.Role == RoleToolResult {
			answered[m.ToolCallID] = true
		}
	}

	flushResults := func(pending []wireBlock) []wireBlock {
		if len(pending) > 0 {
			out = append(out, wireMessage{Role: "user", Content: pending})
		}
		return nil
	}

	var pendingResults []wireBlock

	for _, m := range msgs {
		switch m.Role {
		case RoleUser:
			pendingResults = flushResults(pendingResults)
			blocks := make([]wireBlock, 0, len(m.Content))
			for _, c := range m.Content {
				switch c.Type {
				case ContentText:
					if c.Text != "" {
						blocks = append(blocks, wireBlock{Type: "text", Text: c.Text})
					}
				case ContentImage:
					// DeepSeek rejects image blocks; degrade to a marker rather
					// than failing the whole request.
					blocks = append(blocks, wireBlock{Type: "text", Text: "[image omitted: not supported by DeepSeek]"})
				}
			}
			if len(blocks) == 0 {
				continue
			}
			out = append(out, wireMessage{Role: "user", Content: blocks})

		case RoleAssistant:
			pendingResults = flushResults(pendingResults)
			// Rule 1: never replay a crashed turn.
			if m.StopReason == StopError || m.StopReason == StopAborted {
				continue
			}
			blocks := make([]wireBlock, 0, len(m.Content))
			var missing []wireBlock
			for _, c := range m.Content {
				switch c.Type {
				case ContentThinking:
					// Replayed verbatim, signature included. DeepSeek ties the
					// signature to the response id and expects it back, and it
					// requires both fields present even when the text is empty.
					thinking, signature := c.Thinking, c.Signature
					blocks = append(blocks, wireBlock{
						Type: "thinking", Thinking: &thinking, Signature: &signature,
					})
				case ContentText:
					if c.Text != "" {
						blocks = append(blocks, wireBlock{Type: "text", Text: c.Text})
					}
				case ContentToolCall:
					input := c.Arguments
					if len(input) == 0 {
						input = json.RawMessage("{}")
					}
					blocks = append(blocks, wireBlock{
						Type: "tool_use", ID: c.ID, Name: c.Name, Input: input,
					})
					// Rule 2: synthesize a result for anything unanswered.
					if !answered[c.ID] {
						missing = append(missing, wireBlock{
							Type:      "tool_result",
							ToolUseID: c.ID,
							Content:   "No result provided",
							IsError:   true,
						})
					}
				}
			}
			if len(blocks) == 0 {
				continue
			}
			out = append(out, wireMessage{Role: "assistant", Content: blocks})
			if len(missing) > 0 {
				out = append(out, wireMessage{Role: "user", Content: missing})
			}

		case RoleToolResult:
			// Rule 3: accumulate; a run of results becomes one user message.
			pendingResults = append(pendingResults, wireBlock{
				Type:      "tool_result",
				ToolUseID: m.ToolCallID,
				Content:   toolResultText(m),
				IsError:   m.IsError,
			})
		}
	}
	flushResults(pendingResults)
	return out
}

// toolResultText flattens tool result content to a string. The endpoint accepts
// a bare string for tool_result content, which keeps the payload smaller and
// byte-stable compared with the block-array form.
func toolResultText(m Message) string {
	var b strings.Builder
	for _, c := range m.Content {
		if c.Type == ContentText {
			b.WriteString(c.Text)
		}
	}
	if b.Len() == 0 {
		// The API rejects an empty tool_result content field.
		return "(no output)"
	}
	return b.String()
}

func (c *Client) buildRequest(aiCtx Context, opts StreamOptions) wireRequest {
	model := MustLookup(opts.Model)
	maxTokens := opts.MaxTokens
	if maxTokens <= 0 || maxTokens > model.MaxTokens {
		maxTokens = model.MaxTokens
	}

	req := wireRequest{
		Model:       opts.Model,
		MaxTokens:   maxTokens,
		Stream:      true,
		Messages:    encodeMessages(aiCtx.Messages),
		Temperature: opts.Temperature,
	}
	if aiCtx.SystemPrompt != "" {
		req.System = []wireBlock{{Type: "text", Text: aiCtx.SystemPrompt}}
	}
	for _, t := range aiCtx.Tools {
		req.Tools = append(req.Tools, wireTool{
			Name: t.Name, Description: t.Description, InputSchema: t.Parameters,
		})
	}
	if model.Reasoning {
		if opts.Effort != EffortOff && model.SupportsEffort(opts.Effort) {
			req.Thinking = &wireThinking{Type: "enabled"}
			req.OutputConfig = &wireEffort{Effort: string(opts.Effort)}
		} else {
			req.Thinking = &wireThinking{Type: "disabled"}
		}
	}
	return req
}

// -------------------------------------------------------------------- stream

// Stream performs one streaming request.
//
// It honours the StreamFunc contract absolutely: it never returns an error and
// never panics out. Every failure becomes a final message with StopError or
// StopAborted, delivered through the stream.
func (c *Client) Stream(ctx context.Context, aiCtx Context, opts StreamOptions) *Stream {
	if c.APIKey == "" {
		return ErrorStream(opts.Model, "no DeepSeek API key: set DEEPSEEK_API_KEY")
	}
	if opts.Model == "" {
		opts.Model = ModelFlash
	}

	body, err := json.Marshal(c.buildRequest(aiCtx, opts))
	if err != nil {
		return ErrorStream(opts.Model, fmt.Sprintf("encoding request: %v", err))
	}

	st := NewStream()
	go c.run(ctx, st, opts, body)
	return st
}

// run owns the stream for its lifetime: it is the only Push/Finish caller.
func (c *Client) run(ctx context.Context, st *Stream, opts StreamOptions, body []byte) {
	acc := newAccumulator(opts.Model)
	defer func() {
		// A panic in the decode path must not take the process down; the loop
		// above needs a well-formed failure instead.
		if r := recover(); r != nil {
			st.Finish(acc.fail(fmt.Sprintf("internal stream error: %v", r)))
		}
	}()

	resp, err := c.do(ctx, body)
	if err != nil {
		if ctx.Err() != nil {
			st.Finish(acc.abort("request cancelled"))
			return
		}
		st.Finish(acc.fail(err.Error()))
		return
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 8<<10))
		st.Finish(acc.fail(describeHTTPError(resp.StatusCode, snippet)))
		return
	}

	scanErr := scanSSE(resp.Body, func(ev sseEvent) bool {
		if ctx.Err() != nil {
			return false
		}
		return acc.handle(ev, st)
	})

	if acc.done {
		st.Finish(acc.finalize())
		return
	}
	switch {
	case ctx.Err() != nil:
		st.Finish(acc.abort("request cancelled"))
	case acc.protocolErr != "":
		st.Finish(acc.fail(acc.protocolErr))
	case scanErr != nil:
		st.Finish(acc.fail(fmt.Sprintf("reading stream: %v", scanErr)))
	default:
		// Clean EOF with no message_stop: upstream truncated the stream.
		st.Finish(acc.fail("stream ended before completion"))
	}
}

func (c *Client) do(ctx context.Context, body []byte) (*http.Response, error) {
	base := c.BaseURL
	if base == "" {
		base = DefaultBaseURL
	}
	url := strings.TrimSuffix(base, "/") + "/anthropic/v1/messages"

	httpClient := c.HTTP
	if httpClient == nil {
		httpClient = http.DefaultClient
	}

	for attempt := 0; ; attempt++ {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
		if err != nil {
			return nil, fmt.Errorf("building request: %w", err)
		}
		// The Anthropic-format endpoint authenticates with x-api-key, not a
		// bearer token; that is the difference that trips up callers porting
		// from the OpenAI-format endpoint.
		req.Header.Set("x-api-key", c.APIKey)
		req.Header.Set("anthropic-version", "2023-06-01")
		req.Header.Set("content-type", "application/json")
		req.Header.Set("accept", "text/event-stream")
		if c.UserAgent != "" {
			req.Header.Set("user-agent", c.UserAgent)
		}

		resp, err := httpClient.Do(req)
		if err != nil {
			if ctx.Err() != nil {
				return nil, err
			}
			if delay, ok := c.Retry.next(attempt, 0, ""); ok {
				if !sleepCtx(ctx, delay) {
					return nil, ctx.Err()
				}
				continue
			}
			return nil, err
		}

		if delay, ok := c.Retry.next(attempt, resp.StatusCode, resp.Header.Get("retry-after")); ok {
			_ = resp.Body.Close()
			if !sleepCtx(ctx, delay) {
				return nil, ctx.Err()
			}
			continue
		}
		return resp, nil
	}
}

func describeHTTPError(status int, body []byte) string {
	text := strings.TrimSpace(string(body))
	// Unwrap the standard {"error":{"message":...}} envelope when present so
	// the user sees the reason instead of a wall of JSON.
	var env struct {
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal(body, &env) == nil && env.Error.Message != "" {
		text = env.Error.Message
	}
	if text == "" {
		text = http.StatusText(status)
	}
	switch status {
	case http.StatusUnauthorized:
		return fmt.Sprintf("authentication failed (401): %s — check DEEPSEEK_API_KEY", text)
	case http.StatusTooManyRequests:
		return fmt.Sprintf("rate limited (429): %s", text)
	case http.StatusPaymentRequired:
		return fmt.Sprintf("insufficient balance (402): %s", text)
	}
	return fmt.Sprintf("provider error (%d): %s", status, text)
}

// --------------------------------------------------------------- accumulator

// accumulator turns the SSE event sequence into our event protocol while
// maintaining the live partial message.
type accumulator struct {
	model string

	msg         Message
	blocks      []Content
	jsonBuf     []strings.Builder // per-index partial tool arguments
	done        bool
	protocolErr string
}

func newAccumulator(model string) *accumulator {
	return &accumulator{
		model: model,
		msg: Message{
			Role:       RoleAssistant,
			Model:      model,
			StopReason: StopPending,
			Timestamp:  time.Now().UnixMilli(),
		},
	}
}

// snapshot returns a copy of the live partial. Consumers hold onto events, so
// handing out the mutable backing slice would let a later delta rewrite an
// event a renderer has not drawn yet.
func (a *accumulator) snapshot() *Message {
	m := a.msg
	m.Content = append([]Content(nil), a.blocks...)
	return &m
}

func (a *accumulator) ensure(idx int) {
	for len(a.blocks) <= idx {
		a.blocks = append(a.blocks, Content{})
	}
	for len(a.jsonBuf) <= idx {
		a.jsonBuf = append(a.jsonBuf, strings.Builder{})
	}
}

func (a *accumulator) handle(ev sseEvent, st *Stream) bool {
	if len(ev.Data) == 0 {
		return true
	}
	var w wireStreamEvent
	if err := json.Unmarshal(ev.Data, &w); err != nil {
		// A frame we cannot parse is not fatal on its own; the stream may
		// still complete. Record it in case nothing else explains a failure.
		a.protocolErr = fmt.Sprintf("malformed stream frame: %v", err)
		return true
	}

	switch w.Type {
	case "message_start":
		if w.Message != nil {
			a.msg.ResponseID = w.Message.ID
			if w.Message.Model != "" {
				a.msg.Model = w.Message.Model
			}
			a.applyUsage(w.Message.Usage)
		}
		return st.Push(Event{Type: EventStart, Partial: a.snapshot()})

	case "content_block_start":
		if w.ContentBlock == nil {
			return true
		}
		a.ensure(w.Index)
		switch w.ContentBlock.Type {
		case "text":
			a.blocks[w.Index] = Content{Type: ContentText, Text: w.ContentBlock.Text}
			return st.Push(Event{Type: EventTextStart, ContentIndex: w.Index, Partial: a.snapshot()})
		case "thinking":
			a.blocks[w.Index] = Content{
				Type:      ContentThinking,
				Thinking:  deref(w.ContentBlock.Thinking),
				Signature: deref(w.ContentBlock.Signature),
			}
			return st.Push(Event{Type: EventThinkingStart, ContentIndex: w.Index, Partial: a.snapshot()})
		case "tool_use":
			a.blocks[w.Index] = Content{
				Type: ContentToolCall, ID: w.ContentBlock.ID, Name: w.ContentBlock.Name,
			}
			return st.Push(Event{Type: EventToolCallStart, ContentIndex: w.Index, Partial: a.snapshot()})
		}
		return true

	case "content_block_delta":
		if w.Delta == nil {
			return true
		}
		a.ensure(w.Index)
		switch w.Delta.Type {
		case "text_delta":
			a.blocks[w.Index].Type = ContentText
			a.blocks[w.Index].Text += w.Delta.Text
			return st.Push(Event{
				Type: EventTextDelta, ContentIndex: w.Index,
				Delta: w.Delta.Text, Partial: a.snapshot(),
			})
		case "thinking_delta":
			a.blocks[w.Index].Type = ContentThinking
			a.blocks[w.Index].Thinking += w.Delta.Thinking
			return st.Push(Event{
				Type: EventThinkingDelta, ContentIndex: w.Index,
				Delta: w.Delta.Thinking, Partial: a.snapshot(),
			})
		case "signature_delta":
			// DeepSeek delivers the thinking signature this way, and it equals
			// the response id. It must survive into the transcript: the next
			// turn replays it verbatim.
			a.blocks[w.Index].Type = ContentThinking
			a.blocks[w.Index].Signature += w.Delta.Signature
			return true
		case "input_json_delta":
			a.jsonBuf[w.Index].WriteString(w.Delta.PartialJSON)
			return st.Push(Event{
				Type: EventToolCallDelta, ContentIndex: w.Index,
				Delta: w.Delta.PartialJSON, Partial: a.snapshot(),
			})
		}
		return true

	case "content_block_stop":
		a.ensure(w.Index)
		switch a.blocks[w.Index].Type {
		case ContentText:
			return st.Push(Event{
				Type: EventTextEnd, ContentIndex: w.Index,
				Content: a.blocks[w.Index].Text, Partial: a.snapshot(),
			})
		case ContentThinking:
			return st.Push(Event{
				Type: EventThinkingEnd, ContentIndex: w.Index,
				Content: a.blocks[w.Index].Thinking, Partial: a.snapshot(),
			})
		case ContentToolCall:
			raw := strings.TrimSpace(a.jsonBuf[w.Index].String())
			if raw == "" {
				raw = "{}"
			}
			// Store whatever arrived. Validation happens in the agent layer so
			// a malformed argument becomes a tool error the model can fix,
			// rather than a stream failure that kills the turn.
			a.blocks[w.Index].Arguments = json.RawMessage(raw)
			tc := a.blocks[w.Index]
			return st.Push(Event{
				Type: EventToolCallEnd, ContentIndex: w.Index,
				ToolCall: &tc, Partial: a.snapshot(),
			})
		}
		return true

	case "message_delta":
		if w.Usage != nil {
			a.applyUsage(*w.Usage)
		}
		if w.Delta != nil && w.Delta.StopReason != "" {
			a.msg.StopReason = mapStopReason(w.Delta.StopReason)
		}
		return true

	case "message_stop":
		a.done = true
		return false // stop scanning; run() finalizes

	case "error":
		if w.Error != nil {
			a.protocolErr = fmt.Sprintf("provider stream error (%s): %s", w.Error.Type, w.Error.Message)
		} else {
			a.protocolErr = "provider stream error"
		}
		return false
	}
	return true
}

// applyUsage normalizes provider counters into our Usage.
//
// The endpoint reports input_tokens as the CACHE-MISS portion and
// cache_read_input_tokens separately. Our Usage.Input is the full prompt, so
// the two are summed here. Getting this backwards silently understates prompt
// size and overstates the cache hit rate.
func (a *accumulator) applyUsage(u wireUsage) {
	if u.InputTokens > 0 || u.CacheReadTokens > 0 {
		a.msg.Usage.Input = u.InputTokens + u.CacheReadTokens
		a.msg.Usage.CacheRead = u.CacheReadTokens
	}
	if u.OutputTokens > 0 {
		a.msg.Usage.Output = u.OutputTokens
	}
	// DeepSeek caches automatically and never bills a cache write; the field is
	// reported as 0 and is deliberately not carried into cost.
	a.msg.Usage.CacheWrite = 0
}

func (a *accumulator) finalize() *Message {
	m := a.snapshot()
	// Drop empty trailing blocks the provider may open and never fill.
	kept := m.Content[:0]
	for _, c := range m.Content {
		if c.Type == "" {
			continue
		}
		if c.Type == ContentText && c.Text == "" {
			continue
		}
		kept = append(kept, c)
	}
	m.Content = kept

	if m.StopReason == StopPending {
		// No stop_reason arrived. Infer from content: a tool call means the
		// model expects execution, anything else is a normal stop.
		m.StopReason = StopEnd
		if len(m.ToolCalls()) > 0 {
			m.StopReason = StopToolUse
		}
	}
	m.Usage.Cost = MustLookup(m.Model).Price(m.Usage)
	return m
}

func (a *accumulator) fail(msg string) *Message {
	m := a.snapshot()
	m.StopReason = StopError
	m.ErrorMessage = msg
	m.Usage.Cost = MustLookup(m.Model).Price(m.Usage)
	return m
}

func (a *accumulator) abort(msg string) *Message {
	m := a.snapshot()
	m.StopReason = StopAborted
	m.ErrorMessage = msg
	m.Usage.Cost = MustLookup(m.Model).Price(m.Usage)
	return m
}

// deref reads an optional wire string.
func deref(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}

func mapStopReason(s string) StopReason {
	switch s {
	case "end_turn", "stop_sequence":
		return StopEnd
	case "tool_use":
		return StopToolUse
	case "max_tokens":
		return StopLength
	default:
		return StopEnd
	}
}
