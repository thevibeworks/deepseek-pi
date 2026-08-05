package main

import (
	"bufio"
	"os"
	"strings"
)

// Bracketed paste. A terminal in this mode wraps pasted text in ESC[200~ and
// ESC[201~, which is the only way a line-oriented REPL can tell "the user typed
// a line" from "the user dropped in a twenty-line stack trace".
//
// Without it every pasted line is its own prompt: a twenty-line paste fires
// twenty API calls, nineteen of them on fragments with no question attached.
const (
	pasteOn    = "\x1b[?2004h"
	pasteOff   = "\x1b[?2004l"
	pasteStart = "\x1b[200~"
	pasteEnd   = "\x1b[201~"
)

// input reads prompts, joining a bracketed paste into one.
//
// One reader for the whole process: the REPL and the approval prompt read the
// same stdin, and two readers would each buffer ahead and swallow the other's
// input.
type input struct {
	sc *bufio.Scanner
	// bracketed records whether the terminal was actually put into the mode, so
	// Close only sends the restore sequence when it sent the enable.
	bracketed bool
	out       *os.File
	in        *os.File
	// restore undoes the ECHOCTL change. Nil when there was nothing to change.
	restore func()
}

func newInput(in, out *os.File) *input {
	sc := bufio.NewScanner(in)
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
	return &input{sc: sc, in: in, out: out}
}

// Bracket turns on bracketed paste when both ends are a terminal.
//
// Both, not either: the sequence is written to stdout, so a redirected stdout
// would take the escape codes as content, and a piped stdin will never contain
// the markers anyway.
func (i *input) Bracket() {
	if i.bracketed || !isTerminal(i.in) || !isTerminal(i.out) {
		return
	}
	_, _ = i.out.WriteString(pasteOn)
	i.bracketed = true
	// Hide the markers the terminal is now going to send back at us. Failing
	// is fine: the paste still works, it just looks worse.
	if restore, ok := quietEcho(i.in); ok {
		i.restore = restore
	}
}

// Close restores the terminal. Leaving either setting changed would hand it to
// whatever runs next in that shell.
func (i *input) Close() {
	if i.restore != nil {
		i.restore()
		i.restore = nil
	}
	if !i.bracketed {
		return
	}
	_, _ = i.out.WriteString(pasteOff)
	i.bracketed = false
}

// Err reports a read failure, matching bufio.Scanner: io.EOF is nil.
func (i *input) Err() error { return i.sc.Err() }

// Line reads one answer for a prompt that expects a single line, discarding
// paste markers. The approval prompt uses it: "y" pasted rather than typed is
// still "y".
func (i *input) Line() (string, bool) {
	if !i.sc.Scan() {
		return "", false
	}
	text := i.sc.Text()
	text = strings.ReplaceAll(text, pasteStart, "")
	return strings.ReplaceAll(text, pasteEnd, ""), true
}

// Prompt reads one prompt. A bracketed paste spanning many lines comes back as
// a single multi-line string, so it costs one turn instead of one per line.
func (i *input) Prompt() (string, bool) {
	if !i.sc.Scan() {
		return "", false
	}
	line := i.sc.Text()

	head, rest, found := strings.Cut(line, pasteStart)
	if !found {
		return line, true
	}

	// Everything from here to the end marker is pasted, newlines included.
	var b strings.Builder
	b.WriteString(head)
	for {
		if body, tail, done := strings.Cut(rest, pasteEnd); done {
			b.WriteString(body)
			// Text typed after the paste, before Enter, belongs to this prompt.
			b.WriteString(tail)
			return b.String(), true
		}
		b.WriteString(rest)
		b.WriteString("\n")

		if !i.sc.Scan() {
			// EOF inside a paste: return what arrived rather than dropping it.
			// The bytes are real input, and the terminal is not going to send
			// the closing marker now.
			return strings.TrimSuffix(b.String(), "\n"), b.Len() > 0
		}
		rest = i.sc.Text()
	}
}

func isTerminal(f *os.File) bool {
	info, err := f.Stat()
	if err != nil {
		return false
	}
	return info.Mode()&os.ModeCharDevice != 0
}
