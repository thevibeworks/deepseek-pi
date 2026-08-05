package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"

	"github.com/thevibeworks/deepseek-pi/agent"
	"github.com/thevibeworks/deepseek-pi/ai"
	"github.com/thevibeworks/deepseek-pi/tools"
)

// style holds the ANSI codes, emptied when colour is off.
type style struct {
	dim, bold, red, green, yellow, cyan, reset string
}

func newStyle(enabled bool) style {
	if !enabled {
		return style{}
	}
	return style{
		dim: "\x1b[2m", bold: "\x1b[1m", red: "\x1b[31m", green: "\x1b[32m",
		yellow: "\x1b[33m", cyan: "\x1b[36m", reset: "\x1b[0m",
	}
}

// colorEnabled follows the conventions users already expect: NO_COLOR wins,
// then an explicit FORCE_COLOR, then whether stdout is a terminal.
func colorEnabled(w *os.File) bool {
	if os.Getenv("NO_COLOR") != "" {
		return false
	}
	if os.Getenv("FORCE_COLOR") != "" {
		return true
	}
	info, err := w.Stat()
	if err != nil {
		return false
	}
	return info.Mode()&os.ModeCharDevice != 0
}

// renderer draws loop events to a writer.
//
// It is deliberately a state machine over events rather than a redraw loop:
// output is append-only, so a piped transcript reads exactly like what was on
// screen, and there is no cursor arithmetic to get wrong on resize.
type renderer struct {
	mu sync.Mutex
	w  io.Writer
	s  style

	// Quiet suppresses thinking and tool chatter, leaving only final text.
	Quiet bool
	// ShowThinking prints reasoning as it streams.
	ShowThinking bool

	inText     bool
	inThinking bool
}

func newRenderer(w io.Writer, s style) *renderer {
	return &renderer{w: w, s: s}
}

// printf and print write to the terminal. A failed write to stdout is not
// something the agent can act on, so the error is dropped deliberately here
// rather than threaded through every render path.
func (r *renderer) printf(format string, args ...any) {
	_, _ = fmt.Fprintf(r.w, format, args...)
}

func (r *renderer) print(s string) {
	_, _ = fmt.Fprint(r.w, s)
}

func (r *renderer) Handle(ev agent.Event) {
	r.mu.Lock()
	defer r.mu.Unlock()

	switch ev.Type {
	case agent.EventMessageUpdate:
		r.handleStream(ev)

	case agent.EventMessageEnd:
		if ev.Message == nil {
			return
		}
		r.closeBlocks()
		if ev.Message.Role == ai.RoleAssistant {
			if ev.Message.StopReason == ai.StopError {
				r.printf("\n%serror:%s %s\n", r.s.red, r.s.reset, ev.Message.ErrorMessage)
			}
			if ev.Message.StopReason == ai.StopAborted {
				r.printf("\n%scancelled%s\n", r.s.yellow, r.s.reset)
			}
		}

	case agent.EventToolStart:
		if r.Quiet {
			return
		}
		r.closeBlocks()
		r.printf("%s%s%s %s\n", r.s.cyan, ev.ToolName, r.s.reset, summarizeArgs(ev.ToolName, ev.Args))

	case agent.EventToolEnd:
		if r.Quiet {
			return
		}
		mark, colour := "ok", r.s.green
		if ev.IsError {
			mark, colour = "error", r.s.red
		}
		r.printf("  %s%s%s %s\n", colour, mark, r.s.reset, r.s.dim+summarizeResult(ev.Result)+r.s.reset)
	}
}

func (r *renderer) handleStream(ev agent.Event) {
	if ev.StreamEvent == nil {
		return
	}
	switch ev.StreamEvent.Type {
	case ai.EventThinkingDelta:
		if !r.ShowThinking || r.Quiet {
			return
		}
		if !r.inThinking {
			r.closeBlocks()
			r.printf("%sthinking: ", r.s.dim)
			r.inThinking = true
		}
		r.print(ev.StreamEvent.Delta)
	case ai.EventTextDelta:
		if !r.inText {
			r.closeBlocks()
			r.inText = true
		}
		r.print(ev.StreamEvent.Delta)
	}
}

func (r *renderer) closeBlocks() {
	if r.inThinking {
		r.printf("%s\n", r.s.reset)
		r.inThinking = false
	}
	if r.inText {
		r.print("\n")
		r.inText = false
	}
}

// Finish closes any open block. Call once a run settles.
func (r *renderer) Finish() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.closeBlocks()
}

// summarizeArgs renders a one-line preview of a tool call.
//
// The point is that a user watching the run can tell what the agent is doing
// without reading JSON. Unknown tools fall back to compact JSON rather than
// printing nothing.
func summarizeArgs(name string, raw json.RawMessage) string {
	var m map[string]any
	if json.Unmarshal(raw, &m) != nil {
		return truncateInline(string(raw), 100)
	}
	str := func(k string) string {
		if v, ok := m[k].(string); ok {
			return v
		}
		return ""
	}
	switch name {
	case "read":
		s := str("path")
		if off, ok := m["offset"].(float64); ok && off > 0 {
			s += fmt.Sprintf(" (from line %d)", int(off))
		}
		return s
	case "bash":
		return truncateInline(strings.ReplaceAll(str("command"), "\n", " "), 100)
	case "write":
		return str("path")
	case "edit":
		n := 0
		if edits, ok := m["edits"].([]any); ok {
			n = len(edits)
		}
		return fmt.Sprintf("%s (%d edit(s))", str("path"), n)
	}
	compact, err := json.Marshal(m)
	if err != nil {
		return ""
	}
	return truncateInline(string(compact), 100)
}

func summarizeResult(res *agent.ToolResult) string {
	if res == nil {
		return ""
	}
	var text string
	for _, c := range res.Content {
		if c.Type == ai.ContentText {
			text += c.Text
		}
	}
	text = strings.TrimSpace(text)
	if text == "" {
		return ""
	}
	lines := strings.Split(text, "\n")
	first := truncateInline(lines[0], 100)
	if len(lines) > 1 {
		return fmt.Sprintf("%s (+%d more lines)", first, len(lines)-1)
	}
	return first
}

func truncateInline(s string, max int) string {
	s = strings.TrimSpace(s)
	if len(s) <= max {
		return s
	}
	return s[:max] + "…"
}

// formatUsage renders the accounting line.
//
// Cache hit rate is shown alongside cost because it is the number that explains
// the cost: a run that broke its prefix cache costs many times one that did not,
// and without the hit rate the user has no way to tell which happened.
func formatUsage(model ai.Model, u ai.Usage, s style) string {
	if u.Total() == 0 {
		return ""
	}
	saved := model.CacheSavings(u)
	return fmt.Sprintf(
		"%s%d in / %d out · cache %d (%.0f%%) · $%.4f · saved $%.4f%s",
		s.dim, u.Input, u.Output, u.CacheRead, u.CacheHitRate()*100,
		u.Cost.Total, saved, s.reset)
}

// formatBytes is re-exported for status output.
func formatBytes(n int) string { return tools.FormatSize(n) }
