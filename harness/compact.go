package harness

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync"

	"github.com/thevibeworks/deepseek-pi/agent"
	"github.com/thevibeworks/deepseek-pi/ai"
)

// EstimateTokens sizes a transcript using two tiers.
//
// Provider-reported usage is the anchor: the last assistant message's Input is
// the exact prompt size at that point, and its Output is what it added. Only
// messages appended AFTER that anchor are estimated, at 4 bytes per token.
//
// The point of anchoring is that estimation error cannot accumulate. A pure
// len/4 estimate drifts further from the truth on every turn, and drift in the
// wrong direction means either compacting early forever or discovering the real
// size when the provider rejects the request.
func EstimateTokens(msgs []ai.Message) int {
	anchor, anchorIdx := 0, -1
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == ai.RoleAssistant && msgs[i].Usage.Input > 0 {
			anchor = msgs[i].Usage.Input + msgs[i].Usage.Output
			anchorIdx = i
			break
		}
	}
	total := anchor
	for _, m := range msgs[anchorIdx+1:] {
		total += estimateMessage(m)
	}
	return total
}

// estimateMessage approximates one message. Four bytes per token is the usual
// rule for English-plus-code; no tokenizer dependency is worth carrying for a
// number that only decides when to start a summary.
func estimateMessage(m ai.Message) int {
	bytes := 0
	for _, c := range m.Content {
		bytes += len(c.Text) + len(c.Thinking) + len(c.Arguments)
	}
	// Per-message envelope: role, ids, block framing.
	return bytes/4 + 8
}

// Compactor keeps a session inside the context window.
//
// It runs from the between-turns seam, which is the only place it can: mid-turn
// the transcript has an assistant message whose tool calls are still unanswered,
// and rewriting there would orphan them.
type Compactor struct {
	// Model bounds the context. Budget math uses its real usable window, not
	// the advertised one.
	Model ai.Model
	// Stream performs the summarization request. Nil disables LLM summaries and
	// leaves the deterministic fallback.
	Stream ai.StreamFunc
	// SystemPrompt is the parent's prompt, reused so the summarization request
	// hits the same cached prefix instead of paying full input rate.
	SystemPrompt string

	// Trigger is the fraction of the usable window that starts compaction.
	Trigger float64
	// RetainTail is roughly how many tokens of recent transcript survive.
	RetainTail int

	// OnEvent reports what happened, for UI and session records.
	OnEvent func(CompactionEvent)

	mu sync.Mutex
	// lastSummary lets a later compaction UPDATE the previous summary rather
	// than summarizing a transcript that already begins with one. Restarting
	// each time makes the oldest history progressively lossier.
	lastSummary string
}

// CompactionEvent describes one compaction.
type CompactionEvent struct {
	Reason      string
	BeforeToken int
	AfterToken  int
	Reclaimed   int
	Summarized  int
	UsedLLM     bool
	Err         error
}

// DefaultTrigger compacts at 75% of the usable window.
//
// Not higher: compaction itself needs headroom to run, since the summarization
// request sends the transcript being summarized. Leaving it to 95% means
// discovering there is no room to make room.
const DefaultTrigger = 0.75

// DefaultRetainTail keeps roughly this many tokens of recent work.
const DefaultRetainTail = 40_000

// NewCompactor builds a compactor with the standard budget.
func NewCompactor(model ai.Model, stream ai.StreamFunc, systemPrompt string) *Compactor {
	return &Compactor{
		Model: model, Stream: stream, SystemPrompt: systemPrompt,
		Trigger: DefaultTrigger, RetainTail: DefaultRetainTail,
	}
}

// Threshold is the token count at which compaction starts.
func (c *Compactor) Threshold() int {
	trigger := c.Trigger
	if trigger <= 0 || trigger > 1 {
		trigger = DefaultTrigger
	}
	return int(float64(c.Model.ContextWindow) * trigger)
}

// SyncSummary re-points the running summary at whatever the transcript now
// holds. Anything that cuts history has to call it.
//
// The summary is carried forward so a later compaction can UPDATE it instead of
// re-summarizing a transcript that already begins with one. That is only valid
// while the summary is still in the transcript. Rewind past it and the carried
// text describes turns the user deliberately discarded — the next compaction
// would fold them straight back in, which is the opposite of what they asked
// for. Compaction itself does not need this: it only ever appends a summary it
// just produced.
func (c *Compactor) SyncSummary(msgs []ai.Message) {
	found := ""
	for _, m := range msgs {
		if m.Role != ai.RoleUser {
			continue
		}
		if text := m.Text(); strings.HasPrefix(text, summaryPreamble) {
			found = strings.TrimPrefix(text, summaryPreamble)
		}
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.lastSummary = found
}

// PrepareNextTurn is the agent.LoopConfig hook.
//
// It returns a replacement context when the transcript was rewritten, and nil
// when nothing was needed. It never returns an error: failing to compact means
// continuing with the transcript intact, which is worse later but correct now.
func (c *Compactor) PrepareNextTurn(tc agent.TurnContext) *agent.TurnUpdate {
	if EstimateTokens(tc.Context.Messages) < c.Threshold() {
		return nil
	}
	return c.Compact(tc.Context, "context threshold reached")
}

// Compact rewrites a context regardless of the threshold. This is what
// /compact calls: a user who can see the session filling up should not have to
// wait for an automatic trigger.
func (c *Compactor) Compact(actx *agent.Context, reason string) *agent.TurnUpdate {
	before := EstimateTokens(actx.Messages)
	ev := CompactionEvent{BeforeToken: before, Reason: reason}

	// Reclaim first: it is free. Zero model calls, and it often clears enough
	// that no summary is needed at all.
	msgs, reclaimed := Reclaim(actx.Messages)
	ev.Reclaimed = reclaimed

	if EstimateTokens(msgs) >= c.Threshold() {
		summarized, used, err := c.summarize(context.Background(), msgs)
		ev.UsedLLM, ev.Err = used, err
		if summarized != nil {
			ev.Summarized = len(msgs) - len(summarized)
			msgs = summarized
		}
	}

	if len(msgs) == len(actx.Messages) && reclaimed == 0 {
		return nil // nothing changed; do not churn the context
	}

	ev.AfterToken = EstimateTokens(msgs)
	if c.OnEvent != nil {
		c.OnEvent(ev)
	}

	next := *actx
	next.Messages = msgs
	return &agent.TurnUpdate{Context: &next}
}

// Reclaim drops transcript weight without a model call.
//
// The only thing it removes is a file read that a LATER read of the same file
// supersedes. That is safe in a way that dropping arbitrary history is not: the
// newer result is the current truth, and if the file changed in between, the
// newer read is precisely what the model should be looking at.
//
// The stub left behind matters. Deleting the message outright would make the
// transcript claim the read never happened, and the model would not understand
// why it is missing.
func Reclaim(msgs []ai.Message) ([]ai.Message, int) {
	// A tool RESULT does not carry its arguments, so the file it refers to has
	// to come from the matching tool CALL, keyed by id.
	pathOf := readPathsByCallID(msgs)

	// Last read result per path wins.
	lastRead := map[string]int{}
	for i, m := range msgs {
		if m.Role != ai.RoleToolResult || m.ToolName != "read" || m.IsError {
			continue
		}
		if path := pathOf[m.ToolCallID]; path != "" {
			lastRead[path] = i
		}
	}

	out := make([]ai.Message, len(msgs))
	copy(out, msgs)
	reclaimed := 0
	for i, m := range out {
		if m.Role != ai.RoleToolResult || m.ToolName != "read" || m.IsError {
			continue
		}
		path := pathOf[m.ToolCallID]
		if path == "" || lastRead[path] == i {
			continue
		}
		saved := 0
		for _, cblock := range m.Content {
			saved += len(cblock.Text)
		}
		if saved < 500 {
			continue // not worth the churn
		}
		out[i].Content = []ai.Content{ai.TextContent(fmt.Sprintf(
			"[Earlier read of %s omitted to save context; this file is read again later in the conversation. "+
				"Read it again if you need it now.]", path))}
		reclaimed += saved
	}
	return out, reclaimed
}

// readPathsByCallID maps each read tool call id to the path it requested.
//
// Only whole-file reads are eligible: a paged read (offset or limit) is not
// superseded by a later read of a different window, so treating them as
// interchangeable would discard content the model still needs.
func readPathsByCallID(msgs []ai.Message) map[string]string {
	out := map[string]string{}
	for _, m := range msgs {
		if m.Role != ai.RoleAssistant {
			continue
		}
		for _, c := range m.Content {
			if c.Type != ai.ContentToolCall || c.Name != "read" {
				continue
			}
			var args struct {
				Path   string `json:"path"`
				Offset int    `json:"offset"`
				Limit  int    `json:"limit"`
			}
			if json.Unmarshal(c.Arguments, &args) != nil {
				continue
			}
			if args.Path == "" || args.Offset > 0 || args.Limit > 0 {
				continue
			}
			out[c.ID] = args.Path
		}
	}
	return out
}

// summarize replaces the head of the transcript with a summary message.
//
// The cut point is chosen so no tool result is left without its tool call. A
// dangling tool_result is not a degraded transcript, it is a rejected request:
// the provider refuses a tool_result whose tool_use id it never saw.
func (c *Compactor) summarize(ctx context.Context, msgs []ai.Message) ([]ai.Message, bool, error) {
	cut := findCutPoint(msgs, c.RetainTail)
	if cut <= 0 {
		return nil, false, nil // nothing safe to summarize
	}
	head, tail := msgs[:cut], msgs[cut:]

	c.mu.Lock()
	previous := c.lastSummary
	c.mu.Unlock()

	summary, usedLLM, err := c.renderSummary(ctx, head, previous)
	if summary == "" {
		return nil, usedLLM, err
	}

	c.mu.Lock()
	c.lastSummary = summary
	c.mu.Unlock()

	// The summary enters as an ordinary user message. Everything mid-session
	// arrives as an appended message rather than as prompt mutation, so the
	// system prompt stays byte-identical and only this one break in the
	// transcript costs a cache miss.
	out := make([]ai.Message, 0, len(tail)+1)
	out = append(out, ai.UserMessage(summaryPreamble+summary))
	out = append(out, clearStaleUsage(tail)...)
	return out, usedLLM, err
}

const summaryPreamble = "Earlier conversation was summarized to stay within the context window. " +
	"This summary replaces it; treat it as established fact.\n\n"

// renderSummary asks the model for a summary, falling back to a deterministic
// one when that is impossible.
//
// The fallback is not a nicety. Compaction runs precisely when the context is
// nearly full, which is also when a request is most likely to fail, and a
// compactor that can fail leaves a session that can no longer make progress.
func (c *Compactor) renderSummary(ctx context.Context, head []ai.Message, previous string) (string, bool, error) {
	if c.Stream == nil {
		return DeterministicSummary(head, previous), false, nil
	}

	instruction := summaryInstruction
	if previous != "" {
		// Update the previous summary instead of starting over: re-summarizing
		// a summary loses a little more of the oldest history every time.
		instruction = fmt.Sprintf(
			"%s\n\nA previous summary of even earlier work follows. Fold it into your "+
				"answer so nothing already established is lost:\n\n%s", summaryInstruction, previous)
	}

	// Reuse the parent system prompt so this request hits the same cached
	// prefix rather than paying full input rate for a fresh one.
	req := ai.Context{
		SystemPrompt: c.SystemPrompt,
		Messages:     append(append([]ai.Message{}, head...), ai.UserMessage(instruction)),
	}
	st := c.Stream(req, ai.StreamOptions{
		Model:     c.Model.ID,
		MaxTokens: 4096,
	})
	for range st.Events() {
	}
	final := st.Result()

	if final.StopReason == ai.StopError || strings.TrimSpace(final.Text()) == "" {
		err := fmt.Errorf("summarization failed: %s", final.ErrorMessage)
		return DeterministicSummary(head, previous), false, err
	}
	return final.Text(), true, nil
}

const summaryInstruction = `Summarize the conversation so far so that another engineer could pick up the work with no other context.

Cover, in this order and only where they apply:
1. What the user asked for, including any constraints or preferences they stated.
2. What has been done: files created or changed and what changed in them, commands run and their outcomes.
3. What was learned about the codebase that is not obvious from reading it.
4. What is still outstanding, and the immediate next step.

Be specific: name files, functions and commands. Do not include the contents of files. Do not speculate about what might be true. Write plain prose under short headings, no preamble.`

// DeterministicSummary reconstructs a summary from the transcript alone.
//
// It records only facts already present in the messages, so it is always
// available and can never be wrong in the way a model summary can. It is worse
// than a model summary at explaining intent and better at not inventing it.
func DeterministicSummary(msgs []ai.Message, previous string) string {
	var b strings.Builder
	if previous != "" {
		b.WriteString(previous)
		b.WriteString("\n\n")
	}
	b.WriteString("## Recovered summary\n\n")
	b.WriteString("(Generated without a model call, so it lists facts rather than intent.)\n")

	var requests []string
	filesTouched := map[string]bool{}
	var commands []string
	errors := 0

	for _, m := range msgs {
		switch m.Role {
		case ai.RoleUser:
			if text := strings.TrimSpace(m.Text()); text != "" && !strings.HasPrefix(text, summaryPreamble) {
				requests = append(requests, firstLine(text, 200))
			}
		case ai.RoleToolResult:
			if m.IsError {
				errors++
			}
			switch m.ToolName {
			case "edit", "write":
				if p := pathFromResultText(m.Text()); p != "" {
					filesTouched[p] = true
				}
			case "bash":
				if len(commands) < 20 {
					commands = append(commands, firstLine(m.Text(), 100))
				}
			}
		}
	}

	if len(requests) > 0 {
		b.WriteString("\n### What the user asked for\n\n")
		for _, r := range requests {
			fmt.Fprintf(&b, "- %s\n", r)
		}
	}
	if len(filesTouched) > 0 {
		paths := make([]string, 0, len(filesTouched))
		for p := range filesTouched {
			paths = append(paths, p)
		}
		sort.Strings(paths) // deterministic: map order is not
		b.WriteString("\n### Files changed\n\n")
		for _, p := range paths {
			fmt.Fprintf(&b, "- %s\n", p)
		}
	}
	if len(commands) > 0 {
		b.WriteString("\n### Commands run (most recent output lines)\n\n")
		for _, c := range commands {
			fmt.Fprintf(&b, "- %s\n", c)
		}
	}
	if errors > 0 {
		fmt.Fprintf(&b, "\n%d tool call(s) failed during the summarized period.\n", errors)
	}
	return b.String()
}

// pathFromResultText pulls a path out of an edit/write result line such as
// "Applied 2 edit(s) to internal/x.go (3 line(s) changed)".
func pathFromResultText(text string) string {
	line := firstLine(text, 300)
	for _, marker := range []string{" to ", "Created ", "Overwrote "} {
		if i := strings.Index(line, marker); i >= 0 {
			rest := strings.TrimSpace(line[i+len(marker):])
			if j := strings.IndexAny(rest, " ("); j > 0 {
				return rest[:j]
			}
			return rest
		}
	}
	return ""
}

// clearStaleUsage drops provider usage from messages that survive compaction.
//
// This is not bookkeeping, it is what makes compaction terminate. Usage.Input
// on a retained assistant message describes a prompt that INCLUDED the head we
// just dropped, and EstimateTokens anchors on the newest such figure. Leave it
// and the estimate never falls, so the very next turn compacts again, and every
// turn after that, each one burning a summarization call while the transcript
// stops shrinking.
//
// Resume has the identical hazard and the identical fix; see zeroUsage.
func clearStaleUsage(msgs []ai.Message) []ai.Message {
	out := make([]ai.Message, len(msgs))
	copy(out, msgs)
	for i := range out {
		out[i].Usage = ai.Usage{}
	}
	return out
}

// findCutPoint picks where to split the transcript.
//
// It walks backwards accumulating the tail until the retain budget is spent,
// then keeps walking to the next index that starts a clean segment. A clean cut
// is one where messages[cut] is a real user message, which guarantees the
// retained tail contains no tool result whose tool call was left behind.
func findCutPoint(msgs []ai.Message, retainTokens int) int {
	if len(msgs) == 0 {
		return 0
	}
	budget := 0
	cut := len(msgs)
	for i := len(msgs) - 1; i >= 0; i-- {
		budget += estimateMessage(msgs[i])
		if budget >= retainTokens {
			cut = i
			break
		}
		cut = i
	}

	// Move to a clean boundary. Cutting anywhere else can leave a tool result
	// whose tool_use id no longer appears, and the provider rejects that
	// outright rather than degrading.
	//
	// Forward first, since that reclaims the most. If nothing clean lies ahead
	// — a long tail of tool traffic with no user turn after it — fall BACK to
	// the last clean boundary before the cut. Retaining more than the budget is
	// a weaker compaction; retaining an orphan is a broken request.
	for i := cut; i < len(msgs); i++ {
		if isCleanBoundary(msgs, i) {
			return i
		}
	}
	for i := cut - 1; i > 0; i-- {
		if isCleanBoundary(msgs, i) {
			return i
		}
	}
	return 0 // no safe cut anywhere; skip compaction rather than corrupt the transcript
}

// isCleanBoundary reports whether the transcript can be split before index i.
func isCleanBoundary(msgs []ai.Message, i int) bool {
	if i <= 0 || i >= len(msgs) {
		return false
	}
	// A real user turn, not a tool result.
	return msgs[i].Role == ai.RoleUser
}
