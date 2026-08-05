// Package agent is the runtime: the loop that turns a prompt into assistant
// turns and tool executions, plus the state around it.
//
// It plays the role @earendil-works/pi-agent-core plays in the Pi harness. The
// loop contracts are ported deliberately and in full, because they are the part
// of a coding agent that is expensive to rediscover:
//
//   - A StreamFunc never fails; failures arrive as a final message.
//   - A truncated response fails its whole tool batch rather than executing
//     possibly-incomplete arguments.
//   - Tool end-events fire in completion order; tool RESULT MESSAGES are
//     appended in the model's source order.
//   - Everything that happens "between turns" goes through one named seam.
//   - A crashed turn never poisons the next provider payload.
package agent

import (
	"context"
	"encoding/json"

	"github.com/thevibeworks/deepseek-pi/ai"
)

// ExecutionMode controls how a tool batch runs.
type ExecutionMode string

const (
	// ModeParallel runs the batch concurrently. This is the default.
	ModeParallel ExecutionMode = "parallel"
	// ModeSequential runs the batch one call at a time. A single tool that
	// declares this forces the ENTIRE batch sequential, because the reason a
	// tool needs it (shared mutable state) is not confined to that tool.
	ModeSequential ExecutionMode = "sequential"
)

// ToolCall is one call requested by the model.
type ToolCall struct {
	ID        string
	Name      string
	Arguments json.RawMessage
}

// ToolResult is what a tool produces.
type ToolResult struct {
	// Content goes back to the model.
	Content []ai.Content
	// Details is structured data for renderers and logs. It never reaches the
	// model, so it can be as verbose as a UI needs.
	Details any
	// Terminate hints that the run should stop after this batch. The loop only
	// honours it when EVERY result in the batch sets it, so one tool cannot
	// unilaterally end a run it does not have the whole picture for.
	Terminate bool
}

// Text builds a plain text tool result.
func Text(s string) ToolResult {
	return ToolResult{Content: []ai.Content{ai.TextContent(s)}}
}

// Textf builds a text tool result from a format string.
func Textf(format string, args ...any) ToolResult {
	return Text(sprintf(format, args...))
}

// UpdateFunc lets a long-running tool stream partial output to the UI.
// Calls made after Execute returns are ignored.
type UpdateFunc func(ToolResult)

// Tool is an executable capability offered to the model.
//
// PromptSnippet and PromptGuidelines are why this is a struct rather than a
// bare function: a tool's documentation lives WITH the tool, and the system
// prompt is assembled deterministically from these fields. That is the
// mechanism that keeps prompt and tool set from drifting apart as tools are
// added, and it is much cheaper to adopt now than to retrofit later.
type Tool struct {
	Name        string
	Label       string
	Description string
	Parameters  *ai.Schema

	// PromptSnippet is this tool's line in the system prompt's tool list.
	// Empty means the tool is hidden from the prompt (still callable).
	PromptSnippet string
	// PromptGuidelines are merged, deduplicated, into the Guidelines section.
	PromptGuidelines string

	// ExecutionMode defaults to the loop's mode when empty.
	ExecutionMode ExecutionMode

	// PrepareArguments heals raw model arguments before validation: key
	// aliases, a JSON object delivered as a JSON string, and similar. Coercing
	// beats rejecting, because every rejected call is wasted tokens AND a
	// cache-unfriendly retry turn.
	PrepareArguments func(raw json.RawMessage) json.RawMessage

	// Execute runs the call. Return an error to fail it; the loop converts the
	// error into an error tool result the model can read and react to. Tools
	// should NOT encode failure inside a successful result.
	Execute func(ctx context.Context, call ToolCall, onUpdate UpdateFunc) (ToolResult, error)
}

// Context is the state one loop run works against.
type Context struct {
	SystemPrompt string
	Messages     []ai.Message
	Tools        []Tool
}

// FindTool looks up a tool by name.
func (c *Context) FindTool(name string) (Tool, bool) {
	for _, t := range c.Tools {
		if t.Name == name {
			return t, true
		}
	}
	return Tool{}, false
}

// wireTools projects the executable tools onto the wire schema.
func (c *Context) wireTools() []ai.Tool {
	out := make([]ai.Tool, 0, len(c.Tools))
	for _, t := range c.Tools {
		out = append(out, ai.Tool{
			Name: t.Name, Description: t.Description, Parameters: t.Parameters,
		})
	}
	return out
}

// EventType discriminates loop events.
type EventType string

// The loop event kinds. Agent and turn events bracket a run; message and tool
// events report what happened inside it.
const (
	EventAgentStart EventType = "agent_start"
	EventAgentEnd   EventType = "agent_end"
	EventTurnStart  EventType = "turn_start"
	EventTurnEnd    EventType = "turn_end"

	EventMessageStart  EventType = "message_start"
	EventMessageUpdate EventType = "message_update"
	EventMessageEnd    EventType = "message_end"

	EventToolStart  EventType = "tool_execution_start"
	EventToolUpdate EventType = "tool_execution_update"
	EventToolEnd    EventType = "tool_execution_end"
)

// Event is one observable step of a run.
//
// The vocabulary is intentionally close to ACP's session/update shapes
// (agent_message_chunk, tool_call, tool_call_update, thought_chunk) so a thin
// protocol adapter can project these without the internal model bending to fit
// a wire format.
type Event struct {
	Type EventType

	// Message is set on message_* and turn_end.
	Message *ai.Message
	// StreamEvent carries the underlying token delta on message_update.
	StreamEvent *ai.Event
	// ToolResults is set on turn_end.
	ToolResults []ai.Message
	// Messages is set on agent_end: everything this run produced.
	Messages []ai.Message

	// Tool execution fields.
	ToolCallID string
	ToolName   string
	Args       json.RawMessage
	Result     *ToolResult
	IsError    bool
}

// Sink receives events. It runs inline with the loop, so a slow sink slows the
// run; renderers should hand off rather than block.
type Sink func(Event)

// TurnContext is the state handed to the between-turns hooks.
type TurnContext struct {
	// Message is the assistant message that completed the turn.
	Message *ai.Message
	// ToolResults are the results appended for that message's tool batch.
	ToolResults []ai.Message
	// Context is the live context, with the turn already appended.
	Context *Context
	// NewMessages is everything this run has produced so far.
	NewMessages []ai.Message
}

// TurnUpdate replaces runtime state before the next provider request.
type TurnUpdate struct {
	Context *Context
	Model   string
	Effort  *ai.Effort
}

// BeforeToolResult is returned by the BeforeToolCall hook.
type BeforeToolResult struct {
	Block  bool
	Reason string
}

// AfterToolResult overrides parts of an executed tool result, field by field.
// Omitted (nil / false) fields keep their executed values; there is no deep
// merge.
type AfterToolResult struct {
	Content   []ai.Content
	Details   any
	IsError   *bool
	Terminate *bool
}

// LoopConfig is everything the loop needs beyond the context.
type LoopConfig struct {
	Model     string
	Effort    ai.Effort
	MaxTokens int
	SessionID string

	// ToolExecution defaults to ModeParallel.
	ToolExecution ExecutionMode

	// ConvertToLLM projects the transcript onto provider messages. Use it to
	// drop UI-only entries and to render synthetic messages from pure
	// functions of stored fields, so the same transcript always serializes to
	// the same bytes. Nil means pass through unchanged.
	//
	// Contract: must not panic. Return a safe fallback instead.
	ConvertToLLM func([]ai.Message) []ai.Message

	// TransformContext runs before ConvertToLLM and works at transcript level:
	// context-window management, reclaim, compaction. Nil means no transform.
	//
	// Contract: must not panic. Return the input unchanged on any doubt.
	TransformContext func(context.Context, []ai.Message) []ai.Message

	// PrepareNextTurn is the between-turns seam. Everything that acts between
	// turns — autocompaction, budget envelopes, model arming, steering
	// injection — belongs here rather than inlined in the loop body. Return
	// nil to keep the current state.
	PrepareNextTurn func(TurnContext) *TurnUpdate

	// ShouldStopAfterTurn requests a graceful stop after the current turn,
	// before any further provider request. Tool execution for the finished
	// turn still completes normally.
	ShouldStopAfterTurn func(TurnContext) bool

	// GetSteeringMessages returns messages to inject mid-run, polled once at
	// loop start and after each turn. This is how a user redirects a working
	// agent without cancelling it.
	GetSteeringMessages func() []ai.Message

	// GetFollowUpMessages returns messages to process after the agent would
	// otherwise stop. This is how queued input keeps a run going.
	GetFollowUpMessages func() []ai.Message

	// BeforeToolCall can block a call. It is the permission seam.
	BeforeToolCall func(context.Context, ToolCall, *Context) BeforeToolResult

	// AfterToolCall can rewrite a result before it is recorded.
	AfterToolCall func(context.Context, ToolCall, ToolResult, bool, *Context) *AfterToolResult
}

func (c LoopConfig) mode() ExecutionMode {
	if c.ToolExecution == ModeSequential {
		return ModeSequential
	}
	return ModeParallel
}
