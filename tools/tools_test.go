package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/thevibeworks/deepseek-pi/agent"
)

func testWorkspace(t *testing.T) *Workspace {
	t.Helper()
	w, err := NewWorkspace(t.TempDir())
	if err != nil {
		t.Fatalf("NewWorkspace: %v", err)
	}
	return w
}

func writeFile(t *testing.T, w *Workspace, name, content string) string {
	t.Helper()
	p := filepath.Join(w.Root, name)
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func runTool(t *testing.T, tool agent.Tool, args string) (agent.ToolResult, error) {
	t.Helper()
	raw := json.RawMessage(args)
	if tool.PrepareArguments != nil {
		raw = tool.PrepareArguments(raw)
	}
	if err := agent.ValidateArguments(tool.Parameters, raw); err != nil {
		return agent.ToolResult{}, err
	}
	return tool.Execute(context.Background(), agent.ToolCall{ID: "t1", Name: tool.Name, Arguments: raw}, nil)
}

func text(r agent.ToolResult) string {
	var b strings.Builder
	for _, c := range r.Content {
		b.WriteString(c.Text)
	}
	return b.String()
}

// ------------------------------------------------------------------ truncate

func TestTruncateHeadNeverSplitsLines(t *testing.T) {
	content := strings.Repeat("hello world\n", 100)
	res := TruncateHead(content, Limits{MaxBytes: 50})
	if !res.Truncated {
		t.Fatal("expected truncation")
	}
	for _, line := range strings.Split(res.Content, "\n") {
		if line != "hello world" {
			t.Fatalf("partial line returned: %q", line)
		}
	}
}

func TestTruncateTailKeepsTheEnd(t *testing.T) {
	// Command output is truncated from the tail because the error and the
	// result live at the end.
	var b strings.Builder
	for i := range 100 {
		fmt.Fprintf(&b, "line %d\n", i)
	}
	res := TruncateTail(b.String(), Limits{MaxLines: 3})
	if !strings.Contains(res.Content, "line 99") {
		t.Errorf("tail truncation lost the end: %q", res.Content)
	}
	if strings.Contains(res.Content, "line 0\n") {
		t.Errorf("tail truncation kept the start: %q", res.Content)
	}
	if res.OutputLines != 3 {
		t.Errorf("kept %d lines, want 3", res.OutputLines)
	}
}

func TestTruncateTailSingleHugeLineKeepsEnd(t *testing.T) {
	// The one documented partial-line case.
	line := strings.Repeat("a", 500) + "ERROR_AT_END"
	res := TruncateTail(line, Limits{MaxBytes: 20})
	if !res.LastLinePartial {
		t.Error("expected LastLinePartial for an over-budget single line")
	}
	if !strings.Contains(res.Content, "ERROR_AT_END") {
		t.Errorf("kept the wrong end of the line: %q", res.Content)
	}
}

func TestTruncateHeadFirstLineTooLong(t *testing.T) {
	res := TruncateHead(strings.Repeat("x", 200), Limits{MaxBytes: 50})
	if !res.FirstLineTooLong {
		t.Error("expected FirstLineTooLong")
	}
	if res.Content != "" {
		t.Errorf("content = %q, want empty", res.Content)
	}
	if !strings.Contains(res.Notice(""), "first line") {
		t.Errorf("notice should explain the cause: %q", res.Notice(""))
	}
}

func TestTruncationNoticeIsActionable(t *testing.T) {
	res := TruncateHead(strings.Repeat("line\n", 100), Limits{MaxLines: 10})
	notice := res.Notice("Continue with offset=11.")
	for _, want := range []string{"10 of 100", "Continue with offset=11"} {
		if !strings.Contains(notice, want) {
			t.Errorf("notice %q missing %q", notice, want)
		}
	}
}

func TestTailBytesSnapsToRuneBoundary(t *testing.T) {
	// Cutting mid-rune would emit invalid UTF-8 into the transcript.
	s := strings.Repeat("日", 100)
	out := tailBytes(s, 10)
	if !utf8ValidString(out) {
		t.Errorf("tailBytes produced invalid UTF-8: %q", out)
	}
}

func utf8ValidString(s string) bool {
	for _, r := range s {
		if r == 0xFFFD {
			return false
		}
	}
	return true
}

// ---------------------------------------------------------------------- read

func TestReadNumbersLinesAndPages(t *testing.T) {
	w := testWorkspace(t)
	writeFile(t, w, "a.txt", "one\ntwo\nthree\nfour\n")

	res, err := runTool(t, Read(w), `{"path":"a.txt"}`)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	out := text(res)
	if !strings.Contains(out, "1\tone") || !strings.Contains(out, "4\tfour") {
		t.Errorf("expected numbered lines, got:\n%s", out)
	}

	res, err = runTool(t, Read(w), `{"path":"a.txt","offset":2,"limit":2}`)
	if err != nil {
		t.Fatalf("paged read: %v", err)
	}
	out = text(res)
	if strings.Contains(out, "one") || !strings.Contains(out, "2\ttwo") {
		t.Errorf("paging window wrong:\n%s", out)
	}
	if !strings.Contains(out, "offset=4") {
		t.Errorf("expected an actionable continuation hint, got:\n%s", out)
	}
}

func TestReadHealsCommonArgumentAliases(t *testing.T) {
	w := testWorkspace(t)
	writeFile(t, w, "a.txt", "hello\n")
	// file_path is the alias models reach for most often; rejecting it costs a
	// round trip and a cache break for no information gain.
	if _, err := runTool(t, Read(w), `{"file_path":"a.txt"}`); err != nil {
		t.Errorf("file_path alias not healed: %v", err)
	}
}

func TestReadRejectsEscapeFromWorkspace(t *testing.T) {
	w := testWorkspace(t)
	if _, err := runTool(t, Read(w), `{"path":"../../../etc/passwd"}`); err == nil {
		t.Error("path escape was allowed")
	}
}

func TestReadRejectsBinary(t *testing.T) {
	w := testWorkspace(t)
	writeFile(t, w, "bin", "abc\x00\x01\x02def")
	if _, err := runTool(t, Read(w), `{"path":"bin"}`); err == nil {
		t.Error("binary file was read into context")
	}
}

func TestReadDirectoryGivesActionableError(t *testing.T) {
	w := testWorkspace(t)
	if err := os.Mkdir(filepath.Join(w.Root, "sub"), 0o755); err != nil {
		t.Fatal(err)
	}
	_, err := runTool(t, Read(w), `{"path":"sub"}`)
	if err == nil || !strings.Contains(err.Error(), "bash") {
		t.Errorf("directory error should point at a working alternative, got %v", err)
	}
}

// ---------------------------------------------------------------------- edit

func TestEditExactMatch(t *testing.T) {
	w := testWorkspace(t)
	p := writeFile(t, w, "a.go", "package main\n\nfunc main() {}\n")

	if _, err := runTool(t, Edit(w), `{"path":"a.go","edits":[{"oldText":"func main() {}","newText":"func main() { println(1) }"}]}`); err != nil {
		t.Fatalf("edit: %v", err)
	}
	got, _ := os.ReadFile(p)
	if !strings.Contains(string(got), "println(1)") {
		t.Errorf("edit not applied: %s", got)
	}
}

func TestEditFuzzyMatchPreservesUntouchedBytes(t *testing.T) {
	// The model retypes a smart quote as ASCII. Fuzzy matching should rescue
	// that edit WITHOUT normalizing the rest of the file, which would produce
	// an enormous unreviewable diff.
	w := testWorkspace(t)
	original := "a := “keep this”\nb := “and this”\nc := plain\n"
	p := writeFile(t, w, "a.txt", original)

	_, err := runTool(t, Edit(w), `{"path":"a.txt","edits":[{"oldText":"c := plain","newText":"c := changed"}]}`)
	if err != nil {
		t.Fatalf("edit: %v", err)
	}
	got, _ := os.ReadFile(p)
	if !strings.Contains(string(got), "c := changed") {
		t.Errorf("edit not applied: %q", got)
	}
	if !strings.Contains(string(got), "“keep this”") {
		t.Errorf("untouched smart quotes were normalized away: %q", got)
	}
}

func TestEditAmbiguousMatchIsRejected(t *testing.T) {
	// Guessing which occurrence was meant is how an agent corrupts a file while
	// reporting success.
	w := testWorkspace(t)
	writeFile(t, w, "a.txt", "x = 1\nx = 1\n")

	_, err := runTool(t, Edit(w), `{"path":"a.txt","edits":[{"oldText":"x = 1","newText":"x = 2"}]}`)
	if err == nil {
		t.Fatal("ambiguous edit was applied")
	}
	if !strings.Contains(err.Error(), "2 occurrences") || !strings.Contains(err.Error(), "unique") {
		t.Errorf("error should name the count and the fix, got: %v", err)
	}
}

func TestEditMultipleEditsMatchOriginalContent(t *testing.T) {
	w := testWorkspace(t)
	p := writeFile(t, w, "a.txt", "alpha\nbeta\ngamma\n")

	_, err := runTool(t, Edit(w),
		`{"path":"a.txt","edits":[{"oldText":"alpha","newText":"ALPHA"},{"oldText":"gamma","newText":"GAMMA"}]}`)
	if err != nil {
		t.Fatalf("multi-edit: %v", err)
	}
	got, _ := os.ReadFile(p)
	if string(got) != "ALPHA\nbeta\nGAMMA\n" {
		t.Errorf("got %q, want ALPHA/beta/GAMMA", got)
	}
}

func TestEditOverlappingEditsRejected(t *testing.T) {
	w := testWorkspace(t)
	writeFile(t, w, "a.txt", "hello world\n")
	_, err := runTool(t, Edit(w),
		`{"path":"a.txt","edits":[{"oldText":"hello world","newText":"x"},{"oldText":"hello","newText":"y"}]}`)
	if err == nil || !strings.Contains(err.Error(), "overlap") {
		t.Errorf("overlapping edits should be rejected, got %v", err)
	}
}

func TestEditPreservesCRLFAndBOM(t *testing.T) {
	w := testWorkspace(t)
	p := writeFile(t, w, "a.txt", "\uFEFFone\r\ntwo\r\n")

	if _, err := runTool(t, Edit(w), `{"path":"a.txt","edits":[{"oldText":"two","newText":"三"}]}`); err != nil {
		t.Fatalf("edit: %v", err)
	}
	got, _ := os.ReadFile(p)
	s := string(got)
	if !strings.HasPrefix(s, "\uFEFF") {
		t.Error("BOM was dropped")
	}
	if !strings.Contains(s, "\r\n") {
		t.Errorf("CRLF endings were not restored: %q", s)
	}
}

func TestEditHealsLegacySingleEditForm(t *testing.T) {
	w := testWorkspace(t)
	p := writeFile(t, w, "a.txt", "old\n")
	// Models trained on other harnesses emit this shape constantly.
	if _, err := runTool(t, Edit(w), `{"path":"a.txt","oldText":"old","newText":"new"}`); err != nil {
		t.Fatalf("legacy single-edit form not healed: %v", err)
	}
	got, _ := os.ReadFile(p)
	if strings.TrimSpace(string(got)) != "new" {
		t.Errorf("got %q, want new", got)
	}
}

func TestEditNoOpIsRejected(t *testing.T) {
	w := testWorkspace(t)
	writeFile(t, w, "a.txt", "same\n")
	_, err := runTool(t, Edit(w), `{"path":"a.txt","edits":[{"oldText":"same","newText":"same"}]}`)
	if err == nil {
		t.Error("a no-op edit should be reported, not silently succeed")
	}
}

func TestEditMissingFilePointsAtWrite(t *testing.T) {
	w := testWorkspace(t)
	_, err := runTool(t, Edit(w), `{"path":"nope.txt","edits":[{"oldText":"a","newText":"b"}]}`)
	if err == nil || !strings.Contains(err.Error(), "write") {
		t.Errorf("error should point at the working alternative, got %v", err)
	}
}

// --------------------------------------------------------------------- write

func TestWriteCreatesParentDirs(t *testing.T) {
	w := testWorkspace(t)
	if _, err := runTool(t, Write(w), `{"path":"a/b/c.txt","content":"hi"}`); err != nil {
		t.Fatalf("write: %v", err)
	}
	got, err := os.ReadFile(filepath.Join(w.Root, "a/b/c.txt"))
	if err != nil || string(got) != "hi" {
		t.Errorf("got %q, %v", got, err)
	}
}

func TestWritePreservesFileMode(t *testing.T) {
	w := testWorkspace(t)
	p := writeFile(t, w, "run.sh", "old")
	if err := os.Chmod(p, 0o755); err != nil {
		t.Fatal(err)
	}
	if _, err := runTool(t, Write(w), `{"path":"run.sh","content":"new"}`); err != nil {
		t.Fatalf("write: %v", err)
	}
	info, err := os.Stat(p)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0o755 {
		t.Errorf("mode = %v, want 0755: an overwrite must not strip the executable bit", info.Mode().Perm())
	}
}

// ---------------------------------------------------------------------- bash

func TestBashCapturesOutputAndExitCode(t *testing.T) {
	w := testWorkspace(t)
	res, err := runTool(t, Bash(w, nil), `{"command":"echo hello"}`)
	if err != nil {
		t.Fatalf("bash: %v", err)
	}
	if !strings.Contains(text(res), "hello") {
		t.Errorf("output = %q", text(res))
	}
}

func TestBashNonZeroExitIsAnError(t *testing.T) {
	// The model reliably reacts to the error flag; it routinely misses an exit
	// code buried in prose.
	w := testWorkspace(t)
	_, err := runTool(t, Bash(w, nil), `{"command":"echo to-stderr >&2; exit 3"}`)
	if err == nil {
		t.Fatal("non-zero exit should be an error result")
	}
	if !strings.Contains(err.Error(), "exit 3") || !strings.Contains(err.Error(), "to-stderr") {
		t.Errorf("error should carry the code and the output, got %v", err)
	}
}

func TestBashTimeoutKillsProcessTree(t *testing.T) {
	w := testWorkspace(t)
	_, err := runTool(t, Bash(w, nil), `{"command":"sleep 30","timeout":1}`)
	if err == nil || !strings.Contains(err.Error(), "timed out") {
		t.Errorf("expected a timeout error, got %v", err)
	}
}

func TestBashStdinIsClosed(t *testing.T) {
	// A command waiting on stdin would otherwise burn the whole timeout.
	w := testWorkspace(t)
	res, err := runTool(t, Bash(w, nil), `{"command":"cat; echo done"}`)
	if err != nil {
		t.Fatalf("bash blocked on stdin: %v", err)
	}
	if !strings.Contains(text(res), "done") {
		t.Errorf("output = %q", text(res))
	}
}

func TestBashLargeOutputSpillsToDisk(t *testing.T) {
	// "Spill, don't truncate" must be true for the tool that produces most of
	// the oversized output.
	w := testWorkspace(t)
	res, err := runTool(t, Bash(w, nil), `{"command":"seq 1 200000"}`)
	if err != nil {
		t.Fatalf("bash: %v", err)
	}
	out := text(res)
	if !strings.Contains(out, "Full output saved to") {
		t.Errorf("expected a spill path in the notice, got tail:\n%s", out[max(0, len(out)-300):])
	}
	details, _ := res.Details.(map[string]any)
	path, _ := details["fullOutputPath"].(string)
	if path == "" {
		t.Fatal("no fullOutputPath recorded")
	}
	defer func() { _ = os.Remove(path) }()

	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("spill file missing: %v", err)
	}
	if info.Size() < DefaultMaxBytes {
		t.Errorf("spill file is %d bytes, smaller than the in-context limit", info.Size())
	}
	// The tail is what the model sees, and the end is where the answer is.
	if !strings.Contains(out, "200000") {
		t.Error("spilled output lost the tail")
	}
}

func TestBashRunsInWorkspace(t *testing.T) {
	w := testWorkspace(t)
	writeFile(t, w, "marker.txt", "x")
	res, err := runTool(t, Bash(w, nil), `{"command":"ls"}`)
	if err != nil {
		t.Fatalf("bash: %v", err)
	}
	if !strings.Contains(text(res), "marker.txt") {
		t.Errorf("command did not run in the workspace: %q", text(res))
	}
}

// ----------------------------------------------------------------- workspace

func TestWorkspaceFileLockSerializesSamePath(t *testing.T) {
	// Tool batches run in parallel by default; two edits to one file must not
	// interleave, or the second silently discards the first.
	w := testWorkspace(t)
	p := writeFile(t, w, "a.txt", "")

	var mu sync.Mutex
	var concurrent, peak int
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = w.WithFileLock(p, func() error {
				mu.Lock()
				concurrent++
				if concurrent > peak {
					peak = concurrent
				}
				mu.Unlock()
				mu.Lock()
				concurrent--
				mu.Unlock()
				return nil
			})
		}()
	}
	wg.Wait()
	if peak > 1 {
		t.Errorf("peak concurrency on one file = %d, want 1", peak)
	}
}

func TestWorkspaceResolveAllowsOutsideWhenEnabled(t *testing.T) {
	w := testWorkspace(t)
	if _, err := w.Resolve("/etc/hosts"); err == nil {
		t.Error("outside path allowed while scoped")
	}
	w.AllowOutside = true
	if _, err := w.Resolve("/etc/hosts"); err != nil {
		t.Errorf("outside path rejected with AllowOutside: %v", err)
	}
}
