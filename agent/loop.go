package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/thevibeworks/deepseek-pi/ai"
)

func sprintf(format string, args ...any) string { return fmt.Sprintf(format, args...) }

// Run executes the loop with new prompt messages appended to the context.
//
// It returns the messages this run produced, in transcript order. It does not
// return an error: a failed run ends with an assistant message carrying
// StopError, which is a fact about the conversation rather than a Go error.
func Run(ctx context.Context, prompts []ai.Message, c *Context, cfg LoopConfig, stream ai.StreamFunc, emit Sink) []ai.Message {
	newMessages := append([]ai.Message(nil), prompts...)
	c.Messages = append(c.Messages, prompts...)

	emit(Event{Type: EventAgentStart})
	emit(Event{Type: EventTurnStart})
	for i := range prompts {
		emit(Event{Type: EventMessageStart, Message: &prompts[i]})
		emit(Event{Type: EventMessageEnd, Message: &prompts[i]})
	}

	return runLoop(ctx, c, &newMessages, cfg, stream, emit)
}

// Continue resumes the loop from the current context without adding a message.
//
// This is the retry primitive: after a failed turn is popped from the context,
// Continue replays from the tail. The last message must convert to a user or
// tool-result message, otherwise the provider rejects the payload.
func Continue(ctx context.Context, c *Context, cfg LoopConfig, stream ai.StreamFunc, emit Sink) []ai.Message {
	if len(c.Messages) == 0 {
		return nil
	}
	if c.Messages[len(c.Messages)-1].Role == ai.RoleAssistant {
		return nil
	}

	var newMessages []ai.Message
	emit(Event{Type: EventAgentStart})
	emit(Event{Type: EventTurnStart})
	return runLoop(ctx, c, &newMessages, cfg, stream, emit)
}

// runLoop is the shared body.
//
// Two nested loops with distinct jobs: the INNER loop keeps going while the
// model wants tools or steering has arrived; the OUTER loop restarts it when a
// follow-up message shows up after the agent would otherwise have stopped.
func runLoop(ctx context.Context, c *Context, newMessages *[]ai.Message, cfg LoopConfig, stream ai.StreamFunc, emit Sink) []ai.Message {
	firstTurn := true

	// Poll steering once at the start: the user may have typed while the
	// previous turn was still streaming.
	var pending []ai.Message
	if cfg.GetSteeringMessages != nil {
		pending = cfg.GetSteeringMessages()
	}

	for {
		hasMoreToolCalls := true

		for hasMoreToolCalls || len(pending) > 0 {
			if firstTurn {
				firstTurn = false
			} else {
				emit(Event{Type: EventTurnStart})
			}

			// Inject steering before the next assistant response.
			if len(pending) > 0 {
				for i := range pending {
					emit(Event{Type: EventMessageStart, Message: &pending[i]})
					emit(Event{Type: EventMessageEnd, Message: &pending[i]})
					c.Messages = append(c.Messages, pending[i])
					*newMessages = append(*newMessages, pending[i])
				}
				pending = nil
			}

			msg := streamAssistant(ctx, c, cfg, stream, emit)
			*newMessages = append(*newMessages, *msg)

			// A failed or cancelled turn ends the run. The message stays in the
			// transcript as a record; the provider payload encoder drops it.
			if msg.StopReason == ai.StopError || msg.StopReason == ai.StopAborted {
				emit(Event{Type: EventTurnEnd, Message: msg})
				emit(Event{Type: EventAgentEnd, Messages: *newMessages})
				return *newMessages
			}

			calls := msg.ToolCalls()
			var toolResults []ai.Message
			hasMoreToolCalls = false

			if len(calls) > 0 {
				var batch executedBatch
				if msg.StopReason == ai.StopLength {
					// The output token limit cut the response off, so every tool
					// call in it may carry truncated arguments. Streamed
					// arguments are finalized with a best-effort salvage parse,
					// so a truncated call can still parse AND validate while
					// being silently incomplete. None are safe to run.
					batch = failTruncatedBatch(calls, emit)
				} else {
					batch = executeBatch(ctx, c, cfg, calls, emit)
				}
				toolResults = batch.results
				hasMoreToolCalls = !batch.terminate

				c.Messages = append(c.Messages, toolResults...)
				*newMessages = append(*newMessages, toolResults...)
			}

			emit(Event{Type: EventTurnEnd, Message: msg, ToolResults: toolResults})

			// The between-turns seam. Everything that acts between turns goes
			// through here instead of being inlined above.
			turn := TurnContext{
				Message: msg, ToolResults: toolResults,
				Context: c, NewMessages: *newMessages,
			}
			if cfg.PrepareNextTurn != nil {
				if up := cfg.PrepareNextTurn(turn); up != nil {
					if up.Context != nil {
						c = up.Context
						turn.Context = c
					}
					if up.Model != "" {
						cfg.Model = up.Model
					}
					if up.Effort != nil {
						cfg.Effort = *up.Effort
					}
				}
			}

			if cfg.ShouldStopAfterTurn != nil && cfg.ShouldStopAfterTurn(turn) {
				emit(Event{Type: EventAgentEnd, Messages: *newMessages})
				return *newMessages
			}

			if ctx.Err() != nil {
				emit(Event{Type: EventAgentEnd, Messages: *newMessages})
				return *newMessages
			}

			if cfg.GetSteeringMessages != nil {
				pending = cfg.GetSteeringMessages()
			}
		}

		// The agent would stop here. Anything queued for after the run?
		if cfg.GetFollowUpMessages != nil {
			if followUps := cfg.GetFollowUpMessages(); len(followUps) > 0 {
				pending = followUps
				continue
			}
		}
		break
	}

	emit(Event{Type: EventAgentEnd, Messages: *newMessages})
	return *newMessages
}

// streamAssistant runs one provider request and keeps the live partial message
// at the tail of the context while it streams.
//
// The partial is REPLACED in place on every delta rather than appended, so the
// context always holds exactly one entry for the in-flight turn. A renderer
// reading the context mid-stream therefore sees current state, and a transform
// that runs mid-turn cannot double-count the streaming message.
func streamAssistant(ctx context.Context, c *Context, cfg LoopConfig, stream ai.StreamFunc, emit Sink) *ai.Message {
	msgs := c.Messages
	if cfg.TransformContext != nil {
		msgs = cfg.TransformContext(ctx, msgs)
	}
	if cfg.ConvertToLLM != nil {
		msgs = cfg.ConvertToLLM(msgs)
	}

	st := stream(ai.Context{
		SystemPrompt: c.SystemPrompt,
		Messages:     msgs,
		Tools:        c.wireTools(),
	}, ai.StreamOptions{
		Model:     cfg.Model,
		MaxTokens: cfg.MaxTokens,
		Effort:    cfg.Effort,
		SessionID: cfg.SessionID,
	})

	addedPartial := false
	for ev := range st.Events() {
		switch ev.Type {
		case ai.EventStart:
			c.Messages = append(c.Messages, *ev.Partial)
			addedPartial = true
			emit(Event{Type: EventMessageStart, Message: ev.Partial})
		case ai.EventDone, ai.EventError:
			// Final message handled below, after the channel drains.
		default:
			if addedPartial && ev.Partial != nil {
				c.Messages[len(c.Messages)-1] = *ev.Partial
				e := ev
				emit(Event{Type: EventMessageUpdate, Message: ev.Partial, StreamEvent: &e})
			}
		}
	}

	final := st.Result()
	if addedPartial {
		c.Messages[len(c.Messages)-1] = *final
	} else {
		c.Messages = append(c.Messages, *final)
		emit(Event{Type: EventMessageStart, Message: final})
	}
	emit(Event{Type: EventMessageEnd, Message: final})
	return final
}

// executedBatch is the outcome of one tool batch.
type executedBatch struct {
	results   []ai.Message
	terminate bool
}

// failTruncatedBatch fails every call of a length-truncated message.
//
// terminate is false on purpose: the model should re-issue the calls with
// complete arguments, so the run continues.
func failTruncatedBatch(calls []ai.Content, emit Sink) executedBatch {
	results := make([]ai.Message, 0, len(calls))
	for _, call := range calls {
		emit(Event{
			Type: EventToolStart, ToolCallID: call.ID,
			ToolName: call.Name, Args: call.Arguments,
		})
		res := Text(fmt.Sprintf(
			"Tool call %q was not executed: the response hit the output token limit, "+
				"so its arguments may be truncated. Re-issue the tool call with complete arguments.",
			call.Name))
		emit(Event{
			Type: EventToolEnd, ToolCallID: call.ID, ToolName: call.Name,
			Result: &res, IsError: true,
		})
		msg := resultMessage(call, res, true)
		emit(Event{Type: EventMessageStart, Message: &msg})
		emit(Event{Type: EventMessageEnd, Message: &msg})
		results = append(results, msg)
	}
	return executedBatch{results: results, terminate: false}
}

// finalized is one call's settled outcome.
type finalized struct {
	call    ai.Content
	result  ToolResult
	isError bool
}

// executeBatch runs a tool batch and enforces the two ordering rules.
//
//   - tool_execution_end fires in COMPLETION order, so a UI can show whichever
//     tool finished first.
//   - tool result MESSAGES are appended in the model's SOURCE order, so the
//     transcript matches the assistant message that requested them. Providers
//     reject a mismatch, and a transcript reordered by timing is not
//     reproducible.
func executeBatch(ctx context.Context, c *Context, cfg LoopConfig, calls []ai.Content, emit Sink) executedBatch {
	mode := cfg.mode()
	// One sequential tool forces the whole batch sequential: the shared state
	// it is protecting is not confined to itself.
	for _, call := range calls {
		if t, ok := c.FindTool(call.Name); ok && t.ExecutionMode == ModeSequential {
			mode = ModeSequential
			break
		}
	}

	settled := make([]finalized, len(calls))
	if mode == ModeSequential {
		for i, call := range calls {
			emit(Event{Type: EventToolStart, ToolCallID: call.ID, ToolName: call.Name, Args: call.Arguments})
			settled[i] = runOne(ctx, c, cfg, call, emit)
			emitToolEnd(settled[i], emit)
			if ctx.Err() != nil {
				// Fill the rest so every call still gets a result: the provider
				// requires every tool_use to be answered.
				for j := i + 1; j < len(calls); j++ {
					settled[j] = finalized{
						call: calls[j], result: Text("Operation aborted"), isError: true,
					}
					emit(Event{
						Type: EventToolStart, ToolCallID: calls[j].ID,
						ToolName: calls[j].Name, Args: calls[j].Arguments,
					})
					emitToolEnd(settled[j], emit)
				}
				break
			}
		}
	} else {
		// Start events fire in source order before anything runs, so a UI can
		// draw the whole batch immediately.
		for _, call := range calls {
			emit(Event{Type: EventToolStart, ToolCallID: call.ID, ToolName: call.Name, Args: call.Arguments})
		}
		var wg sync.WaitGroup
		var mu sync.Mutex // serializes emit; sinks are not required to be safe
		for i, call := range calls {
			wg.Add(1)
			go func() {
				defer wg.Done()
				f := runOne(ctx, c, cfg, call, func(ev Event) {
					mu.Lock()
					defer mu.Unlock()
					emit(ev)
				})
				mu.Lock()
				settled[i] = f
				emitToolEnd(f, emit) // completion order
				mu.Unlock()
			}()
		}
		wg.Wait()
	}

	// Source order for the messages.
	results := make([]ai.Message, 0, len(settled))
	terminate := len(settled) > 0
	for _, f := range settled {
		msg := resultMessage(f.call, f.result, f.isError)
		emit(Event{Type: EventMessageStart, Message: &msg})
		emit(Event{Type: EventMessageEnd, Message: &msg})
		results = append(results, msg)
		if !f.result.Terminate {
			terminate = false
		}
	}
	return executedBatch{results: results, terminate: terminate}
}

// runOne prepares, permission-checks, executes and finalizes a single call.
// It never panics out: a panicking tool becomes an error result.
func runOne(ctx context.Context, c *Context, cfg LoopConfig, call ai.Content, emit Sink) (f finalized) {
	tc := ToolCall{ID: call.ID, Name: call.Name, Arguments: call.Arguments}
	f = finalized{call: call}

	defer func() {
		if r := recover(); r != nil {
			f = finalized{
				call:    call,
				result:  Text(fmt.Sprintf("Tool %q panicked: %v", call.Name, r)),
				isError: true,
			}
		}
	}()

	tool, ok := c.FindTool(call.Name)
	if !ok {
		f.result, f.isError = Text(fmt.Sprintf("Tool %q not found", call.Name)), true
		return f
	}

	args := tc.Arguments
	if len(args) == 0 {
		args = json.RawMessage("{}")
	}
	if tool.PrepareArguments != nil {
		args = tool.PrepareArguments(args)
	}
	if err := ValidateArguments(tool.Parameters, args); err != nil {
		// One structured error listing everything wrong, so the model fixes it
		// in a single retry instead of discovering problems one at a time.
		f.result, f.isError = Text(err.Error()), true
		return f
	}
	tc.Arguments = args

	if cfg.BeforeToolCall != nil {
		if b := cfg.BeforeToolCall(ctx, tc, c); b.Block {
			reason := b.Reason
			if reason == "" {
				reason = "Tool execution was blocked"
			}
			f.result, f.isError = Text(reason), true
			return f
		}
	}
	if ctx.Err() != nil {
		f.result, f.isError = Text("Operation aborted"), true
		return f
	}

	var updateMu sync.Mutex
	accepting := true
	res, err := tool.Execute(ctx, tc, func(partial ToolResult) {
		updateMu.Lock()
		defer updateMu.Unlock()
		if !accepting {
			return
		}
		p := partial
		emit(Event{
			Type: EventToolUpdate, ToolCallID: call.ID,
			ToolName: call.Name, Args: args, Result: &p,
		})
	})
	updateMu.Lock()
	accepting = false
	updateMu.Unlock()

	if err != nil {
		f.result, f.isError = Text(err.Error()), true
	} else {
		f.result, f.isError = res, false
	}

	if cfg.AfterToolCall != nil {
		if over := cfg.AfterToolCall(ctx, tc, f.result, f.isError, c); over != nil {
			if over.Content != nil {
				f.result.Content = over.Content
			}
			if over.Details != nil {
				f.result.Details = over.Details
			}
			if over.IsError != nil {
				f.isError = *over.IsError
			}
			if over.Terminate != nil {
				f.result.Terminate = *over.Terminate
			}
		}
	}
	return f
}

func emitToolEnd(f finalized, emit Sink) {
	res := f.result
	emit(Event{
		Type: EventToolEnd, ToolCallID: f.call.ID, ToolName: f.call.Name,
		Result: &res, IsError: f.isError,
	})
}

func resultMessage(call ai.Content, res ToolResult, isError bool) ai.Message {
	content := res.Content
	if len(content) == 0 {
		content = []ai.Content{ai.TextContent("(no output)")}
	}
	return ai.Message{
		Role:       ai.RoleToolResult,
		ToolCallID: call.ID,
		ToolName:   call.Name,
		Content:    content,
		IsError:    isError,
		Timestamp:  time.Now().UnixMilli(),
	}
}
