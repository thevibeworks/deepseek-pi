// Package ai is the DeepSeek wire layer: one message model, one stream
// protocol, one usage/cost accounting, shared by everything above it.
//
// It plays the role @earendil-works/pi-ai plays in the Pi harness, minus the
// multi-provider surface. deepseek-pi targets the DeepSeek v4 series only, so
// there is exactly one protocol state machine to keep correct (the
// Anthropic-compatible Messages API that v4 is trained toward) instead of a
// provider matrix. That single-protocol decision is the whole reason this
// package is a few hundred lines rather than a few thousand.
package ai

import (
	"encoding/json"
	"time"
)

// Role identifies who produced a message.
type Role string

// The roles a transcript message can carry.
const (
	RoleUser       Role = "user"
	RoleAssistant  Role = "assistant"
	RoleToolResult Role = "toolResult"
)

// ContentType discriminates the blocks inside a message.
type ContentType string

// The content block kinds a message can carry.
const (
	ContentText     ContentType = "text"
	ContentThinking ContentType = "thinking"
	ContentToolCall ContentType = "toolCall"
	ContentImage    ContentType = "image"
)

// Content is one block of message content.
//
// This is a flat struct with a Type discriminator rather than a sum type.
// Sessions are persisted as JSONL and replayed byte-identically to preserve
// the prompt cache; a flat struct round-trips through encoding/json without
// custom marshallers, which is what makes that guarantee cheap to keep.
type Content struct {
	Type ContentType `json:"type"`

	// Text blocks.
	Text string `json:"text,omitempty"`

	// Thinking blocks. DeepSeek returns a signature equal to the response id;
	// it must be replayed verbatim on the next turn or the model loses the
	// thread of its own reasoning.
	Thinking  string `json:"thinking,omitempty"`
	Signature string `json:"signature,omitempty"`

	// Tool call blocks.
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Arguments json.RawMessage `json:"arguments,omitempty"`

	// Image blocks.
	Data     string `json:"data,omitempty"`
	MimeType string `json:"mimeType,omitempty"`
}

// TextContent builds a text block.
func TextContent(text string) Content {
	return Content{Type: ContentText, Text: text}
}

// StopReason is why the assistant stopped generating.
type StopReason string

const (
	// StopPending is the zero value: the message is still streaming.
	StopPending StopReason = "pending"
	// StopEnd is a normal completion.
	StopEnd StopReason = "stop"
	// StopLength means the output token limit truncated the response. Any tool
	// calls in such a message may carry silently incomplete arguments.
	StopLength StopReason = "length"
	// StopToolUse means the model wants tools executed before continuing.
	StopToolUse StopReason = "toolUse"
	// StopError is a transport, protocol or upstream failure.
	StopError StopReason = "error"
	// StopAborted means the caller cancelled the request.
	StopAborted StopReason = "aborted"
)

// Usage is token accounting for one request, taken from the provider only.
//
// DeepSeek caches automatically and reports reads but never writes, so
// CacheWrite exists for interface symmetry and is always zero here. Input is
// the full prompt including cached tokens: Input == CacheRead + billed miss.
type Usage struct {
	Input     int `json:"input"`
	Output    int `json:"output"`
	CacheRead int `json:"cacheRead"`
	// CacheWrite is always 0 on DeepSeek. Kept so cost math reads the same as
	// on providers that bill cache writes.
	CacheWrite int `json:"cacheWrite"`
	Reasoning  int `json:"reasoning,omitempty"`

	// Cost is computed from our own pricing table, never from a provider or
	// harness cost field. Retargeted Claude harnesses misprice DeepSeek by
	// ~13x because they apply Claude rate cards to DeepSeek tokens.
	Cost Cost `json:"cost"`
}

// CacheMiss is the part of the prompt that was billed at the full input rate.
func (u Usage) CacheMiss() int {
	miss := u.Input - u.CacheRead
	if miss < 0 {
		return 0
	}
	return miss
}

// CacheHitRate is the fraction of the prompt served from cache, in [0,1].
func (u Usage) CacheHitRate() float64 {
	if u.Input <= 0 {
		return 0
	}
	return float64(u.CacheRead) / float64(u.Input)
}

// Total is every token the request touched.
func (u Usage) Total() int { return u.Input + u.Output }

// Add accumulates another usage record into this one.
func (u *Usage) Add(other Usage) {
	u.Input += other.Input
	u.Output += other.Output
	u.CacheRead += other.CacheRead
	u.CacheWrite += other.CacheWrite
	u.Reasoning += other.Reasoning
	u.Cost.Add(other.Cost)
}

// Message is one entry in a transcript.
//
// Pi models user/assistant/toolResult as three types; Go gets one struct with
// a Role discriminator so the transcript is a plain []Message that persists
// and replays without type switches at every layer.
type Message struct {
	Role      Role      `json:"role"`
	Content   []Content `json:"content,omitempty"`
	Timestamp int64     `json:"timestamp"`

	// Assistant fields.
	Model        string     `json:"model,omitempty"`
	StopReason   StopReason `json:"stopReason,omitempty"`
	ErrorMessage string     `json:"errorMessage,omitempty"`
	Usage        Usage      `json:"usage,omitzero"`
	ResponseID   string     `json:"responseId,omitempty"`

	// Tool result fields.
	ToolCallID string `json:"toolCallId,omitempty"`
	ToolName   string `json:"toolName,omitempty"`
	IsError    bool   `json:"isError,omitempty"`
}

// UserMessage builds a plain text user turn.
func UserMessage(text string) Message {
	return Message{
		Role:      RoleUser,
		Content:   []Content{TextContent(text)},
		Timestamp: time.Now().UnixMilli(),
	}
}

// ToolCalls returns the tool call blocks of an assistant message, in the order
// the model emitted them. That order is the contract for appending results.
func (m Message) ToolCalls() []Content {
	var calls []Content
	for _, c := range m.Content {
		if c.Type == ContentToolCall {
			calls = append(calls, c)
		}
	}
	return calls
}

// Text concatenates every text block, which is what a caller wants when it
// asks "what did the model actually say".
func (m Message) Text() string {
	var out string
	for _, c := range m.Content {
		if c.Type == ContentText {
			out += c.Text
		}
	}
	return out
}

// Schema is a minimal JSON Schema for tool parameters.
//
// Deliberately not a general JSON Schema implementation: tool schemas are part
// of the cached prompt prefix, so they must serialize deterministically. A
// small explicit struct with ordered required-fields does that; a generic
// map[string]any does not (Go randomizes map iteration, but encoding/json
// sorts keys, so the risk is in nested any values, which this type forbids).
type Schema struct {
	Type        string             `json:"type"`
	Description string             `json:"description,omitempty"`
	Properties  map[string]*Schema `json:"properties,omitempty"`
	Required    []string           `json:"required,omitempty"`
	Items       *Schema            `json:"items,omitempty"`
	Enum        []string           `json:"enum,omitempty"`
}

// Object builds an object schema. Required order is preserved as given.
func Object(props map[string]*Schema, required ...string) *Schema {
	return &Schema{Type: "object", Properties: props, Required: required}
}

// Str builds a string schema.
func Str(desc string) *Schema { return &Schema{Type: "string", Description: desc} }

// Int builds an integer schema.
func Int(desc string) *Schema { return &Schema{Type: "integer", Description: desc} }

// Bool builds a boolean schema.
func Bool(desc string) *Schema { return &Schema{Type: "boolean", Description: desc} }

// Array builds an array schema.
func Array(desc string, items *Schema) *Schema {
	return &Schema{Type: "array", Description: desc, Items: items}
}

// Tool is a callable exposed to the model. This is the wire-level half; the
// executable half lives in package agent.
type Tool struct {
	Name        string  `json:"name"`
	Description string  `json:"description"`
	Parameters  *Schema `json:"parameters"`
}

// Context is everything sent to the model for one request.
type Context struct {
	SystemPrompt string
	Messages     []Message
	Tools        []Tool
}

// EventType discriminates streaming events.
type EventType string

// The streaming event kinds, in the order a well-formed response emits them.
const (
	EventStart         EventType = "start"
	EventTextStart     EventType = "text_start"
	EventTextDelta     EventType = "text_delta"
	EventTextEnd       EventType = "text_end"
	EventThinkingStart EventType = "thinking_start"
	EventThinkingDelta EventType = "thinking_delta"
	EventThinkingEnd   EventType = "thinking_end"
	EventToolCallStart EventType = "toolcall_start"
	EventToolCallDelta EventType = "toolcall_delta"
	EventToolCallEnd   EventType = "toolcall_end"
	EventDone          EventType = "done"
	EventError         EventType = "error"
)

// Event is one item in an assistant response stream.
//
// Partial always points at the live partial message, so a consumer can render
// from Partial alone and ignore the deltas if it wants.
type Event struct {
	Type         EventType
	ContentIndex int
	Delta        string
	Content      string
	ToolCall     *Content
	Partial      *Message
	// Message is set on Done and Error and is the final assistant message.
	Message *Message
	Reason  StopReason
}

// StreamFunc is the provider call boundary.
//
// Contract, copied verbatim from pi because it is what keeps the loop simple:
// a StreamFunc must NEVER return an error for a request, model or runtime
// failure. Failures are encoded in the returned stream as a final message with
// StopError or StopAborted plus ErrorMessage. The loop therefore has exactly
// one failure path to handle instead of two.
type StreamFunc func(ctx Context, opts StreamOptions) *Stream

// StreamOptions are the per-request knobs.
type StreamOptions struct {
	Model     string
	MaxTokens int
	// Effort is the DeepSeek reasoning level. Empty disables thinking.
	Effort Effort
	// Temperature is applied only when non-nil; v4 defaults are good.
	Temperature *float64
	// SessionID is echoed to the provider for cache routing where supported.
	SessionID string
}

// Effort is DeepSeek's reasoning level vocabulary.
type Effort string

// The reasoning levels DeepSeek v4 accepts. EffortOff disables thinking.
const (
	EffortOff   Effort = ""
	EffortLow   Effort = "low"
	EffortHigh  Effort = "high"
	EffortXHigh Effort = "xhigh"
	EffortMax   Effort = "max"
)

// Valid reports whether e is a level DeepSeek v4 accepts.
func (e Effort) Valid() bool {
	switch e {
	case EffortOff, EffortLow, EffortHigh, EffortXHigh, EffortMax:
		return true
	}
	return false
}
