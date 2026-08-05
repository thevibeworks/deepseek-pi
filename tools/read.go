package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"unicode/utf8"

	"github.com/thevibeworks/deepseek-pi/agent"
	"github.com/thevibeworks/deepseek-pi/ai"
)

// ReadArgs are the read tool's parameters.
type ReadArgs struct {
	Path   string `json:"path"`
	Offset int    `json:"offset"`
	Limit  int    `json:"limit"`
}

// Read builds the file-reading tool.
//
// Line numbers are emitted because the edit tool matches on text and the model
// needs to see structure to pick a unique anchor. Offset is 1-indexed to match
// what every editor and error message in the world reports.
func Read(w *Workspace) agent.Tool {
	return agent.Tool{
		Name:  "read",
		Label: "Read",
		Description: fmt.Sprintf(
			"Read a file from the workspace. Returns numbered lines.\n"+
				"Bounded at %d lines or %s per call, whichever comes first; "+
				"use offset and limit to page through a larger file.",
			DefaultMaxLines, FormatSize(DefaultMaxBytes)),
		Parameters: ai.Object(map[string]*ai.Schema{
			"path":   ai.Str("Path to the file, relative to the workspace root or absolute."),
			"offset": ai.Int("1-indexed line to start at. Omit to start at the beginning."),
			"limit":  ai.Int("Maximum number of lines to return."),
		}, "path"),
		PromptSnippet: "read(path, offset?, limit?) — read a file as numbered lines",
		PromptGuidelines: "Read a file before editing it. Never guess at a file's contents. " +
			"When a read reports more lines remain, page with offset rather than re-reading from the start.",
		ExecutionMode: agent.ModeParallel,
		PrepareArguments: agent.HealArguments(map[string]string{
			"file_path": "path", "filename": "path", "filepath": "path", "file": "path",
			"start_line": "offset", "line": "offset", "max_lines": "limit",
		}),
		Execute: func(_ context.Context, call agent.ToolCall, _ agent.UpdateFunc) (agent.ToolResult, error) {
			var args ReadArgs
			if err := json.Unmarshal(call.Arguments, &args); err != nil {
				return agent.ToolResult{}, fmt.Errorf("invalid arguments: %w", err)
			}
			abs, err := w.Resolve(args.Path)
			if err != nil {
				return agent.ToolResult{}, err
			}

			info, err := os.Stat(abs)
			if err != nil {
				if os.IsNotExist(err) {
					return agent.ToolResult{}, fmt.Errorf("file not found: %s", w.Rel(abs))
				}
				return agent.ToolResult{}, err
			}
			if info.IsDir() {
				return agent.ToolResult{}, fmt.Errorf(
					"%s is a directory, not a file; use bash with ls to list it", w.Rel(abs))
			}

			raw, err := os.ReadFile(abs)
			if err != nil {
				return agent.ToolResult{}, err
			}
			if isBinary(raw) {
				return agent.ToolResult{}, fmt.Errorf(
					"%s looks like a binary file (%s); reading it would waste context, "+
						"so use bash with a tool that understands the format",
					w.Rel(abs), FormatSize(len(raw)))
			}

			content := string(raw)
			lines := splitLines(content)
			total := len(lines)

			start := args.Offset
			if start <= 0 {
				start = 1
			}
			if start > total {
				return agent.Text(fmt.Sprintf(
					"[%s has %d lines; offset %d is past the end]", w.Rel(abs), total, start)), nil
			}
			end := total
			if args.Limit > 0 && start-1+args.Limit < end {
				end = start - 1 + args.Limit
			}
			window := lines[start-1 : end]

			numbered := numberLines(window, start)
			tr := TruncateHead(numbered, Limits{})

			var b strings.Builder
			b.WriteString(tr.Content)

			shown := tr.OutputLines
			lastShown := start + shown - 1
			switch {
			case tr.Truncated:
				b.WriteString(tr.Notice(fmt.Sprintf(
					"Continue with offset=%d.", lastShown+1)))
			case end < total:
				fmt.Fprintf(&b, "\n\n[Showing lines %d-%d of %d. Continue with offset=%d.]",
					start, end, total, end+1)
			}

			return agent.ToolResult{
				Content: []ai.Content{ai.TextContent(b.String())},
				Details: map[string]any{
					"path": w.Rel(abs), "totalLines": total,
					"shownFrom": start, "shownTo": lastShown, "bytes": len(raw),
				},
			}, nil
		},
	}
}

// numberLines renders lines with right-aligned 1-indexed numbers.
func numberLines(lines []string, start int) string {
	width := len(fmt.Sprint(start + len(lines) - 1))
	var b strings.Builder
	for i, line := range lines {
		if i > 0 {
			b.WriteByte('\n')
		}
		// Long lines are clipped so one minified file cannot eat the budget.
		text, _ := ClipLine(line, 2000)
		fmt.Fprintf(&b, "%*d\t%s", width, start+i, text)
	}
	return b.String()
}

// isBinary reports whether data looks like a binary file.
//
// A NUL byte in the first sniff window is the reliable signal; invalid UTF-8 on
// its own is not, because legitimate source files in other encodings exist.
func isBinary(data []byte) bool {
	const sniff = 8000
	window := data
	if len(window) > sniff {
		window = window[:sniff]
	}
	for _, b := range window {
		if b == 0 {
			return true
		}
	}
	// A high proportion of invalid UTF-8 alongside no NULs still suggests
	// binary; require a clear majority so odd encodings survive.
	if len(window) > 0 && !utf8.Valid(window) {
		invalid := 0
		for i := 0; i < len(window); {
			r, size := utf8.DecodeRune(window[i:])
			if r == utf8.RuneError && size == 1 {
				invalid++
			}
			i += size
		}
		return invalid*10 > len(window)
	}
	return false
}
