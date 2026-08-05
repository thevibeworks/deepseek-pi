package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"

	"github.com/thevibeworks/deepseek-pi/agent"
)

// approver asks the user to confirm a tool call.
//
// It holds a mutex for the whole prompt because tool batches run in PARALLEL:
// without it, two concurrent calls would interleave their questions and read
// each other's answers off the same stdin. Serializing means the second tool
// waits, which is correct — a person can only answer one question at a time.
type approver struct {
	mu    sync.Mutex
	in    *input
	out   *os.File
	style style
}

func newApprover(in *input, out *os.File, s style) *approver {
	return &approver{in: in, out: out, style: s}
}

// printf writes the prompt. A failed write to the terminal is not something
// the caller can act on, and dropping the error here keeps the prompt logic
// readable rather than threading an error through every line.
func (a *approver) printf(format string, args ...any) {
	_, _ = fmt.Fprintf(a.out, format, args...)
}

// Ask presents the call and returns whether to run it and whether to stop
// asking for that tool. A closed stdin or an unreadable answer is a refusal:
// silence must never be read as consent.
func (a *approver) Ask(ctx context.Context, call agent.ToolCall, reason string) (allow, remember bool) {
	a.mu.Lock()
	defer a.mu.Unlock()

	if ctx.Err() != nil {
		return false, false
	}

	s := a.style
	a.printf("\n%s%s wants to run:%s\n", s.yellow, call.Name, s.reset)
	for _, line := range describeCall(call) {
		a.printf("  %s\n", line)
	}
	if reason != "" {
		a.printf("  %sneeds approval: %s%s\n", s.dim, reason, s.reset)
	}
	a.printf("%s[y] once  [a] always for %s  [n] no%s %s>%s ",
		s.dim, call.Name, s.reset, s.yellow, s.reset)

	line, ok := a.in.Line()
	if !ok {
		a.printf("\n")
		return false, false
	}
	answer := strings.ToLower(strings.TrimSpace(line))
	// Close the prompt line ourselves: when stdin is a pipe the terminal never
	// echoes a newline, and the next output would run onto the question.
	a.printf("\n")

	switch answer {
	case "y", "yes":
		return true, false
	case "a", "always":
		return true, true
	default:
		return false, false
	}
}

// describeCall renders the call for a human deciding whether to permit it.
//
// This is the last thing a user sees before arbitrary code runs, so it shows
// the ACTUAL command and the real paths rather than a summary. A preview that
// hides the dangerous part of a command is worse than no preview.
func describeCall(call agent.ToolCall) []string {
	args := decodeArgs(call.Arguments)
	str := func(k string) string {
		v, _ := args[k].(string)
		return v
	}

	switch call.Name {
	case "bash":
		lines := strings.Split(strings.TrimRight(str("command"), "\n"), "\n")
		if cwd := str("cwd"); cwd != "" {
			lines = append(lines, "(in "+cwd+")")
		}
		return lines
	case "write":
		content := str("content")
		return []string{fmt.Sprintf("%s (%d bytes, overwrites any existing file)",
			str("path"), len(content))}
	case "edit":
		edits, _ := args["edits"].([]any)
		out := []string{fmt.Sprintf("%s (%d change(s))", str("path"), len(edits))}
		for i, e := range edits {
			m, ok := e.(map[string]any)
			if !ok {
				continue
			}
			oldText, _ := m["oldText"].(string)
			newText, _ := m["newText"].(string)
			out = append(out,
				fmt.Sprintf("  %d. %s -> %s", i+1,
					truncateInline(firstLineOf(oldText), 60),
					truncateInline(firstLineOf(newText), 60)))
			if i >= 4 && len(edits) > 5 {
				out = append(out, fmt.Sprintf("  ... %d more", len(edits)-5))
				break
			}
		}
		return out
	}
	return []string{summarizeArgs(call.Name, call.Arguments)}
}

func firstLineOf(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i] + " ..."
	}
	return s
}

// decodeArgs parses tool arguments for display, returning an empty map on
// anything malformed so a broken payload cannot break the prompt.
func decodeArgs(raw json.RawMessage) map[string]any {
	var m map[string]any
	if json.Unmarshal(raw, &m) != nil {
		return map[string]any{}
	}
	return m
}
