package main

import (
	"os"
	"strings"
	"testing"
)

// feed builds a reader over canned terminal input. os.Stdout stands in for the
// terminal: nothing is written to it because a pipe is never bracketed.
func feed(t *testing.T, text string) *input {
	t.Helper()
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	go func() {
		_, _ = w.WriteString(text)
		_ = w.Close()
	}()
	t.Cleanup(func() { _ = r.Close() })
	return newInput(r, os.Stdout)
}

func prompts(t *testing.T, in *input) []string {
	t.Helper()
	var got []string
	for {
		line, ok := in.Prompt()
		if !ok {
			return got
		}
		got = append(got, line)
	}
}

func TestTypedLinesAreOnePromptEach(t *testing.T) {
	in := feed(t, "first\nsecond\n")
	got := prompts(t, in)
	if len(got) != 2 || got[0] != "first" || got[1] != "second" {
		t.Errorf("got %q, want two separate prompts", got)
	}
}

// TestAPastedBlockIsOnePrompt is the whole point. Without bracketing, a
// twenty-line stack trace is twenty prompts and twenty API calls, nineteen of
// them fragments with no question attached.
func TestAPastedBlockIsOnePrompt(t *testing.T) {
	in := feed(t, pasteStart+"line one\nline two\nline three"+pasteEnd+"\n")

	got := prompts(t, in)
	if len(got) != 1 {
		t.Fatalf("a paste became %d prompts: %q", len(got), got)
	}
	if got[0] != "line one\nline two\nline three" {
		t.Errorf("got %q, want the three lines joined verbatim", got[0])
	}
}

func TestASingleLinePasteLosesItsMarkers(t *testing.T) {
	in := feed(t, pasteStart+"just one line"+pasteEnd+"\n")
	got := prompts(t, in)
	if len(got) != 1 || got[0] != "just one line" {
		t.Errorf("got %q, want the bare line", got)
	}
}

func TestTextAroundAPasteStaysWithIt(t *testing.T) {
	// Typing before and after a paste, then hitting Enter, is one prompt.
	in := feed(t, "explain this: "+pasteStart+"panic: nil\nstack"+pasteEnd+" please\n")
	got := prompts(t, in)
	if len(got) != 1 {
		t.Fatalf("got %d prompts: %q", len(got), got)
	}
	if got[0] != "explain this: panic: nil\nstack please" {
		t.Errorf("got %q; the typed text around the paste was lost or reordered", got[0])
	}
}

func TestTypingResumesAfterAPaste(t *testing.T) {
	in := feed(t, pasteStart+"a\nb"+pasteEnd+"\nnext question\n")
	got := prompts(t, in)
	if len(got) != 2 {
		t.Fatalf("got %d prompts: %q", len(got), got)
	}
	if got[1] != "next question" {
		t.Errorf("the line after a paste came back as %q", got[1])
	}
}

// TestAnUnterminatedPasteKeepsWhatArrived covers a truncated paste — a closed
// pipe, or a terminal that never sends the end marker. The bytes are real
// input; dropping them silently would be worse than a partial prompt.
func TestAnUnterminatedPasteKeepsWhatArrived(t *testing.T) {
	in := feed(t, pasteStart+"half a\npaste\n")
	got := prompts(t, in)
	if len(got) != 1 {
		t.Fatalf("got %d prompts: %q", len(got), got)
	}
	if got[0] != "half a\npaste" {
		t.Errorf("got %q, want everything that arrived", got[0])
	}
}

func TestLineStripsMarkersForTheApprovalPrompt(t *testing.T) {
	// "y" pasted rather than typed is still "y".
	in := feed(t, pasteStart+"y"+pasteEnd+"\n")
	got, ok := in.Line()
	if !ok || strings.TrimSpace(got) != "y" {
		t.Errorf("Line() = %q, %v; want a bare y", got, ok)
	}
}

func TestAPipeIsNeverBracketed(t *testing.T) {
	// Writing terminal modes into a redirected stream would put escape codes
	// in the output as content.
	in := feed(t, "hello\n")
	in.Bracket()
	if in.bracketed {
		t.Error("bracketed paste was enabled for a pipe")
	}
}
