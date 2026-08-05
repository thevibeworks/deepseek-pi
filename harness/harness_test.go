package harness

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/thevibeworks/deepseek-pi/agent"
	"github.com/thevibeworks/deepseek-pi/ai"
	"github.com/thevibeworks/deepseek-pi/tools"
)

func demoTools(t *testing.T) []agent.Tool {
	t.Helper()
	w, err := tools.NewWorkspace(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	return []agent.Tool{tools.Read(w), tools.Bash(w, nil), tools.Edit(w), tools.Write(w)}
}

// TestPrefixStability is the guard test the whole cost model rests on.
//
// DeepSeek's automatic prefix cache re-bills the ENTIRE prompt the moment one
// byte before the change point differs. Cached input is 0.0028/M against
// 0.14/M for a miss — a 50x swing — so a prompt that drifts between turns
// quietly multiplies the cost of every long session.
//
// Byte-comparing two builds from identical inputs is what catches the usual
// culprits: a timestamp, a counter, a map iterated without sorting.
func TestPrefixStabilityAcrossTurns(t *testing.T) {
	toolSet := demoTools(t)
	cfg := PromptConfig{
		Cwd:   "/workspace",
		Tools: toolSet,
		ProjectInstructions: []InstructionFile{
			{Path: "/workspace/AGENTS.md", Content: "Use tabs."},
		},
		Skills: []Skill{
			{Name: "zebra", Description: "last alphabetically", Path: "/s/zebra/SKILL.md"},
			{Name: "alpha", Description: "first alphabetically", Path: "/s/alpha/SKILL.md"},
		},
	}

	first := BuildSystemPrompt(cfg)
	time.Sleep(10 * time.Millisecond) // any time-based content would drift here
	second := BuildSystemPrompt(cfg)

	if first != second {
		t.Fatalf("system prompt is not byte-stable across turns.\nfirst:\n%s\n\nsecond:\n%s", first, second)
	}

	// Rebuilding from a freshly constructed config must also match: this
	// catches state accumulating inside the builder.
	third := BuildSystemPrompt(PromptConfig{
		Cwd:   "/workspace",
		Tools: toolSet,
		ProjectInstructions: []InstructionFile{
			{Path: "/workspace/AGENTS.md", Content: "Use tabs."},
		},
		Skills: []Skill{
			{Name: "alpha", Description: "first alphabetically", Path: "/s/alpha/SKILL.md"},
			{Name: "zebra", Description: "last alphabetically", Path: "/s/zebra/SKILL.md"},
		},
	})
	if first != third {
		t.Error("prompt changed when skills were supplied in a different order; sorting is not deterministic")
	}
}

func TestPromptCarriesNoTimeVaryingContent(t *testing.T) {
	// A date in the system prompt breaks every cached session at midnight. It
	// looks harmless and costs a full prefix re-bill per turn for a day.
	got := BuildSystemPrompt(PromptConfig{Cwd: "/w", Tools: demoTools(t)})
	now := time.Now()
	for _, banned := range []string{
		now.Format("2006-01-02"),
		now.Format("15:04"),
		now.Format("January"),
		now.Format("Monday"),
	} {
		if strings.Contains(got, banned) {
			t.Errorf("system prompt contains time-varying content %q", banned)
		}
	}
}

func TestCwdIsTheLastLine(t *testing.T) {
	// The only per-machine value goes last so two machines share the longest
	// possible common prefix.
	got := BuildSystemPrompt(PromptConfig{Cwd: "/my/workspace", Tools: demoTools(t)})
	lines := strings.Split(strings.TrimRight(got, "\n"), "\n")
	last := lines[len(lines)-1]
	if last != "Current working directory: /my/workspace" {
		t.Errorf("last line = %q, want the cwd line", last)
	}
}

func TestToolDocsComeFromTheTools(t *testing.T) {
	// Tool documentation lives with the tool. A hand-maintained list here would
	// drift the first time someone adds a tool in a hurry.
	toolSet := demoTools(t)
	got := BuildSystemPrompt(PromptConfig{Cwd: "/w", Tools: toolSet})

	for _, tool := range toolSet {
		if tool.PromptSnippet == "" {
			continue
		}
		if !strings.Contains(got, tool.PromptSnippet) {
			t.Errorf("prompt is missing the snippet for %s: %q", tool.Name, tool.PromptSnippet)
		}
	}

	// A tool added later shows up with no change to this package.
	extra := agent.Tool{
		Name: "fetch", Label: "Fetch", PromptSnippet: "fetch(url) — retrieve a URL",
		PromptGuidelines: "Prefer local files over the network.",
	}
	withExtra := BuildSystemPrompt(PromptConfig{Cwd: "/w", Tools: append(toolSet, extra)})
	if !strings.Contains(withExtra, "fetch(url)") {
		t.Error("a newly registered tool did not appear in the prompt")
	}
	if !strings.Contains(withExtra, "Prefer local files over the network.") {
		t.Error("a newly registered tool's guidelines did not appear")
	}
}

func TestGuidelinesAreDeduplicated(t *testing.T) {
	// Several tools legitimately say "read before you write"; the model should
	// see it once.
	dup := "Read a file before editing it."
	toolSet := []agent.Tool{
		{Name: "a", PromptSnippet: "a()", PromptGuidelines: dup},
		{Name: "b", PromptSnippet: "b()", PromptGuidelines: dup},
	}
	got := BuildSystemPrompt(PromptConfig{Cwd: "/w", Tools: toolSet})
	if n := strings.Count(got, dup); n != 1 {
		t.Errorf("guideline appears %d times, want 1", n)
	}
}

func TestSkillIndexOmitsBodies(t *testing.T) {
	// Progressive disclosure: the index costs a line per skill in the cached
	// prefix; the body is paid for only on the turn the model reads it.
	got := FormatSkills([]Skill{
		{Name: "deploy", Description: "How to deploy", Path: "/s/deploy/SKILL.md"},
	})
	for _, want := range []string{"deploy", "How to deploy", "/s/deploy/SKILL.md"} {
		if !strings.Contains(got, want) {
			t.Errorf("skill index missing %q", want)
		}
	}
	if strings.Contains(got, "<body") || len(got) > 800 {
		t.Errorf("skill index looks like it carries bodies:\n%s", got)
	}
}

func TestHiddenSkillsAreExcluded(t *testing.T) {
	got := FormatSkills([]Skill{
		{Name: "visible", Description: "shown", Path: "/a"},
		{Name: "secret", Description: "hidden", Path: "/b", Hidden: true},
	})
	if strings.Contains(got, "secret") {
		t.Error("hidden skill leaked into the prompt")
	}
	if !strings.Contains(got, "visible") {
		t.Error("visible skill missing")
	}
}

func TestDiscoverInstructionsWalksRootDown(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	root := t.TempDir()
	// Mark the repo boundary so the walk stops here.
	if err := os.Mkdir(filepath.Join(root, ".git"), 0o755); err != nil {
		t.Fatal(err)
	}
	sub := filepath.Join(root, "pkg", "inner")
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	write := func(dir, name, body string) {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write(root, "AGENTS.md", "root rules")
	write(sub, "AGENTS.md", "inner rules")

	got := DiscoverInstructions(sub)

	var order []string
	for _, f := range got {
		order = append(order, f.Content)
	}
	// Most specific last, so it reads as a refinement of the general rules.
	if len(order) < 2 || order[len(order)-1] != "inner rules" {
		t.Fatalf("instruction order = %v, want root rules before inner rules", order)
	}
	rootIdx, innerIdx := -1, -1
	for i, c := range order {
		switch c {
		case "root rules":
			rootIdx = i
		case "inner rules":
			innerIdx = i
		}
	}
	if rootIdx == -1 || innerIdx == -1 || rootIdx > innerIdx {
		t.Errorf("root should come before inner, got %v", order)
	}
}

func TestDiscoverInstructionsFirstMatchPerDirectory(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	root := t.TempDir()
	if err := os.Mkdir(filepath.Join(root, ".git"), 0o755); err != nil {
		t.Fatal(err)
	}
	// DEEPSEEK.md outranks AGENTS.md in the same directory.
	for name, body := range map[string]string{
		"DEEPSEEK.md": "preferred",
		"AGENTS.md":   "fallback",
	} {
		if err := os.WriteFile(filepath.Join(root, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	got := DiscoverInstructions(root)
	var bodies []string
	for _, f := range got {
		bodies = append(bodies, f.Content)
	}
	joined := strings.Join(bodies, "|")
	if !strings.Contains(joined, "preferred") {
		t.Errorf("expected the preferred filename to win, got %v", bodies)
	}
	if strings.Contains(joined, "fallback") {
		t.Errorf("both files loaded from one directory: %v", bodies)
	}
}

func TestDiscoverSkills(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	root := t.TempDir()
	dir := filepath.Join(root, ".agents", "skills", "deploy")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	body := `---
name: deploy
description: Ship the service safely
---

Long instructions the prompt must not carry.
`
	if err := os.WriteFile(filepath.Join(dir, "SKILL.md"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}

	got := DiscoverSkills(root)
	if len(got) != 1 {
		t.Fatalf("found %d skills, want 1: %+v", len(got), got)
	}
	if got[0].Name != "deploy" || got[0].Description != "Ship the service safely" {
		t.Errorf("parsed skill = %+v", got[0])
	}
}

func TestSkillWithoutDescriptionIsSkipped(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	root := t.TempDir()
	dir := filepath.Join(root, ".agents", "skills", "nodesc")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	// Name and description are the entire basis for the model's decision to
	// read a skill; guessing one would be worse than skipping.
	if err := os.WriteFile(filepath.Join(dir, "SKILL.md"), []byte("---\nname: nodesc\n---\nbody\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := DiscoverSkills(root); len(got) != 0 {
		t.Errorf("found %d skills, want 0", len(got))
	}
}

func TestSessionRoundTrip(t *testing.T) {
	t.Setenv("DEEPSEEK_PI_HOME", t.TempDir())
	cwd := t.TempDir()

	s := NewSession("abc123", cwd, ai.ModelFlash)
	// Nothing on disk until the first append: an abandoned session should not
	// litter the listing.
	if _, err := os.Stat(s.Path()); !os.IsNotExist(err) {
		t.Error("session file created before any message was written")
	}

	msgs := []ai.Message{
		ai.UserMessage("hello"),
		{
			Role: ai.RoleAssistant, StopReason: ai.StopEnd, Model: ai.ModelFlash,
			Content: []ai.Content{ai.TextContent("hi there")},
			Usage:   ai.Usage{Input: 100, Output: 20, CacheRead: 80, Cost: ai.Cost{Total: 0.001}},
		},
	}
	for _, m := range msgs {
		if err := s.Append(m); err != nil {
			t.Fatalf("Append: %v", err)
		}
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	if s.Usage.Input != 100 || s.Usage.Output != 20 {
		t.Errorf("session usage = %+v, want input 100 / output 20", s.Usage)
	}

	loaded, info, err := LoadSession(s.Path())
	if err != nil {
		t.Fatalf("LoadSession: %v", err)
	}
	if info.ID != "abc123" || info.Model != ai.ModelFlash {
		t.Errorf("header = %+v", info)
	}
	if len(loaded) != 2 {
		t.Fatalf("loaded %d messages, want 2", len(loaded))
	}
	if loaded[0].Text() != "hello" || loaded[1].Text() != "hi there" {
		t.Errorf("round trip lost content: %+v", loaded)
	}
	if loaded[1].Content[0].Type != ai.ContentText {
		t.Errorf("content type not preserved: %+v", loaded[1].Content[0])
	}
}

func TestResumeZeroesUsageOnPreservedMessages(t *testing.T) {
	// Tokens on a resumed transcript were billed in the earlier run. Counting
	// them again makes a resumed session look twice as expensive and trips any
	// budget check immediately.
	msgs := []ai.Message{
		{Role: ai.RoleAssistant, Usage: ai.Usage{Input: 500_000, Output: 1000}},
	}
	got := zeroUsage(msgs)
	if got[0].Usage.Input != 0 || got[0].Usage.Output != 0 {
		t.Errorf("usage survived resume: %+v", got[0].Usage)
	}
	if msgs[0].Usage.Input != 500_000 {
		t.Error("zeroUsage mutated its input")
	}
}

func TestListSessionsNewestFirst(t *testing.T) {
	t.Setenv("DEEPSEEK_PI_HOME", t.TempDir())
	cwd := t.TempDir()

	for _, id := range []string{"old", "new"} {
		s := NewSession(id, cwd, ai.ModelFlash)
		if err := s.Append(ai.UserMessage("prompt " + id)); err != nil {
			t.Fatal(err)
		}
		if err := s.Close(); err != nil {
			t.Fatal(err)
		}
		time.Sleep(10 * time.Millisecond)
	}

	list, err := ListSessions(cwd)
	if err != nil {
		t.Fatalf("ListSessions: %v", err)
	}
	if len(list) != 2 {
		t.Fatalf("got %d sessions, want 2", len(list))
	}
	if list[0].ID != "new" {
		t.Errorf("first session = %s, want the newest", list[0].ID)
	}
	if list[0].Preview != "prompt new" {
		t.Errorf("preview = %q, want the first user message", list[0].Preview)
	}
}

func TestSkillBlockScalarDescription(t *testing.T) {
	// Real skills in the wild write long descriptions as YAML block scalars.
	// Reading only the marker put a literal ">-" in the prompt where the
	// description belongs, which is what the model routes on.
	t.Setenv("HOME", t.TempDir())
	root := t.TempDir()
	dir := filepath.Join(root, ".agents", "skills", "folded")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	body := "---\nname: folded\ndescription: >-\n  Use this when the task spans\n  several lines of description.\n---\n\nBody.\n"
	if err := os.WriteFile(filepath.Join(dir, "SKILL.md"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}

	got := DiscoverSkills(root)
	if len(got) != 1 {
		t.Fatalf("found %d skills, want 1", len(got))
	}
	want := "Use this when the task spans several lines of description."
	if got[0].Description != want {
		t.Errorf("description = %q, want %q", got[0].Description, want)
	}
}
