// Package tools implements the coding capabilities the model calls.
//
// Every tool here obeys two rules that exist because context is the scarce
// resource, not compute:
//
//  1. Output is BOUNDED by two independent limits (lines and bytes), whichever
//     is hit first, and never cut mid-line except in one documented tail case.
//  2. When output is too big it SPILLS to disk and the model is told the path,
//     rather than being silently truncated. Truncation loses information the
//     model does not know it is missing; a spill lets it go get the rest.
package tools

import (
	"fmt"
	"strings"
	"unicode/utf8"
)

// Limits shared by every tool. They are part of each tool's description text,
// so the model knows the ceiling before it calls rather than after.
const (
	DefaultMaxLines = 2000
	DefaultMaxBytes = 50 * 1024
	// MaxLineLength bounds a single line in search-style output.
	MaxLineLength = 500
)

// Truncation records what a bounding pass did. Tools turn this into the
// actionable notice appended to their output.
type Truncation struct {
	Content     string
	Truncated   bool
	TruncatedBy string // "lines", "bytes", or ""
	TotalLines  int
	TotalBytes  int
	OutputLines int
	OutputBytes int
	// FirstLineTooLong marks head truncation where even line 1 exceeded the
	// byte limit, so nothing could be returned.
	FirstLineTooLong bool
	// LastLinePartial marks the one case where a partial line is returned:
	// tail truncation of a single line longer than the byte budget.
	LastLinePartial bool
	MaxLines        int
	MaxBytes        int
}

// Limits are per-call bounds. Zero fields fall back to the defaults.
type Limits struct {
	MaxLines int
	MaxBytes int
}

func (l Limits) resolve() (int, int) {
	maxLines, maxBytes := l.MaxLines, l.MaxBytes
	if maxLines <= 0 {
		maxLines = DefaultMaxLines
	}
	if maxBytes <= 0 {
		maxBytes = DefaultMaxBytes
	}
	return maxLines, maxBytes
}

// splitLines splits content into lines without a trailing empty element, so a
// file ending in a newline is not counted as having one extra blank line.
func splitLines(content string) []string {
	if content == "" {
		return nil
	}
	lines := strings.Split(content, "\n")
	if strings.HasSuffix(content, "\n") {
		lines = lines[:len(lines)-1]
	}
	return lines
}

// TruncateHead keeps the first lines that fit. Used for file reads, where the
// beginning is what orients the reader.
func TruncateHead(content string, limits Limits) Truncation {
	maxLines, maxBytes := limits.resolve()
	lines := splitLines(content)
	res := Truncation{
		TotalLines: len(lines), TotalBytes: len(content),
		MaxLines: maxLines, MaxBytes: maxBytes,
	}

	if len(lines) <= maxLines && len(content) <= maxBytes {
		res.Content = content
		res.OutputLines, res.OutputBytes = len(lines), len(content)
		return res
	}

	res.Truncated = true
	if len(lines) > 0 && len(lines[0]) > maxBytes {
		// Nothing can be returned without splitting a line.
		res.TruncatedBy = "bytes"
		res.FirstLineTooLong = true
		return res
	}

	var kept []string
	total := 0
	res.TruncatedBy = "lines"
	for i := 0; i < len(lines) && i < maxLines; i++ {
		size := len(lines[i])
		if i > 0 {
			size++ // newline
		}
		if total+size > maxBytes {
			res.TruncatedBy = "bytes"
			break
		}
		kept = append(kept, lines[i])
		total += size
	}
	res.Content = strings.Join(kept, "\n")
	res.OutputLines, res.OutputBytes = len(kept), len(res.Content)
	return res
}

// TruncateTail keeps the last lines that fit. Used for command output, where
// the end holds the error and the result.
func TruncateTail(content string, limits Limits) Truncation {
	maxLines, maxBytes := limits.resolve()
	lines := splitLines(content)
	res := Truncation{
		TotalLines: len(lines), TotalBytes: len(content),
		MaxLines: maxLines, MaxBytes: maxBytes,
	}

	if len(lines) <= maxLines && len(content) <= maxBytes {
		res.Content = content
		res.OutputLines, res.OutputBytes = len(lines), len(content)
		return res
	}

	res.Truncated = true
	res.TruncatedBy = "lines"
	var kept []string
	total := 0
	for i := len(lines) - 1; i >= 0 && len(kept) < maxLines; i-- {
		size := len(lines[i])
		if len(kept) > 0 {
			size++ // newline
		}
		if total+size > maxBytes {
			res.TruncatedBy = "bytes"
			// The one partial-line case: a single line longer than the whole
			// byte budget. Keep its end, because that is where the error is.
			if len(kept) == 0 {
				kept = append(kept, tailBytes(lines[i], maxBytes))
				res.LastLinePartial = true
			}
			break
		}
		kept = append([]string{lines[i]}, kept...)
		total += size
	}
	res.Content = strings.Join(kept, "\n")
	res.OutputLines, res.OutputBytes = len(kept), len(res.Content)
	return res
}

// tailBytes returns the last maxBytes of s, snapped to a rune boundary so the
// result is never invalid UTF-8.
func tailBytes(s string, maxBytes int) string {
	if maxBytes <= 0 || len(s) <= maxBytes {
		if len(s) <= maxBytes {
			return s
		}
		return ""
	}
	start := len(s) - maxBytes
	for start < len(s) && !utf8.RuneStart(s[start]) {
		start++
	}
	return s[start:]
}

// Notice renders the actionable line appended to truncated output.
//
// "Actionable" is the point: it names the limit that fired and tells the model
// exactly how to get the rest. A bare "[truncated]" makes a model guess, and it
// usually guesses by re-running the same call.
func (t Truncation) Notice(retryHint string) string {
	if !t.Truncated {
		return ""
	}
	var b strings.Builder
	b.WriteString("\n\n[")
	if t.FirstLineTooLong {
		fmt.Fprintf(&b, "Output not shown: the first line alone exceeds the %s limit", FormatSize(t.MaxBytes))
	} else {
		fmt.Fprintf(&b, "Showing %d of %d lines (%s of %s)",
			t.OutputLines, t.TotalLines, FormatSize(t.OutputBytes), FormatSize(t.TotalBytes))
		if t.TruncatedBy == "lines" {
			fmt.Fprintf(&b, ", truncated at the %d-line limit", t.MaxLines)
		} else {
			fmt.Fprintf(&b, ", truncated at the %s limit", FormatSize(t.MaxBytes))
		}
	}
	if retryHint != "" {
		b.WriteString(". ")
		b.WriteString(retryHint)
	}
	b.WriteString("]")
	return b.String()
}

// FormatSize renders a byte count for humans and models alike.
func FormatSize(n int) string {
	switch {
	case n < 1024:
		return fmt.Sprintf("%dB", n)
	case n < 1024*1024:
		return fmt.Sprintf("%.1fKB", float64(n)/1024)
	default:
		return fmt.Sprintf("%.1fMB", float64(n)/(1024*1024))
	}
}

// ClipLine bounds one line, marking it so the model knows it was cut.
func ClipLine(line string, maxChars int) (string, bool) {
	if maxChars <= 0 {
		maxChars = MaxLineLength
	}
	if utf8.RuneCountInString(line) <= maxChars {
		return line, false
	}
	runes := []rune(line)
	return string(runes[:maxChars]) + "... [truncated]", true
}
