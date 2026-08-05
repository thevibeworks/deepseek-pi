// Package harness assembles a working coding agent out of the runtime and the
// tools: system prompt, project context, skills, and session persistence.
//
// It plays the role @earendil-works/pi-coding-agent plays in the Pi harness.
package harness

import (
	"fmt"
	"sort"
	"strings"

	"github.com/thevibeworks/deepseek-pi/agent"
)

// rolePrompt is the fixed opening. It is short on purpose: v4 follows explicit
// operational rules better than it follows personality description, and every
// line here is paid for on every cached turn forever.
const rolePrompt = `You are deepseek-pi, a coding agent running in a terminal on the user's machine.

You work by using tools to inspect and change a real workspace. Read before you
write, verify before you claim, and prefer the smallest change that fully solves
the problem. When you finish, say what you did and what you verified — not what
you intend to do next.`

// alwaysOnGuidelines apply regardless of the active tool set.
var alwaysOnGuidelines = []string{
	"Be concise. Answer in plain prose without headers or bullet lists unless the user asks for structure.",
	"Never claim something works unless you ran it and saw it work. If you could not verify, say so explicitly.",
	"When a command or edit fails, read the error and fix the cause. Do not retry the identical call unchanged.",
}

// PromptConfig is everything that goes into the system prompt.
type PromptConfig struct {
	// Cwd is the workspace root shown to the model.
	Cwd string
	// Tools contribute their own documentation.
	Tools []agent.Tool
	// ProjectInstructions are AGENTS.md-style files, outermost first.
	ProjectInstructions []InstructionFile
	// Skills are the discovered skill index entries.
	Skills []Skill
	// Append is extra text appended verbatim after the built-in sections.
	Append string
}

// BuildSystemPrompt assembles the system prompt deterministically.
//
// Two properties matter more than the wording of any section:
//
//  1. The output is a pure function of its inputs. Same inputs, same bytes,
//     every turn. DeepSeek's automatic prefix cache pays for the whole prompt
//     again the moment one byte changes, so the prompt carries NO date, NO
//     time, no counters and nothing else that varies on its own.
//  2. Tool documentation comes from the tools themselves (PromptSnippet and
//     PromptGuidelines), not from a hand-maintained list here. A list here
//     drifts from the tool set the first time someone adds a tool in a hurry;
//     assembly from the tools cannot.
//
// The cwd line goes LAST because it is the only per-machine value, so two
// machines share the longest possible common prefix.
func BuildSystemPrompt(cfg PromptConfig) string {
	var b strings.Builder

	b.WriteString(rolePrompt)

	// Available tools, in the order registered so the prompt matches the tool
	// array the provider sees.
	var snippets []string
	for _, t := range cfg.Tools {
		if t.PromptSnippet != "" {
			snippets = append(snippets, t.PromptSnippet)
		}
	}
	if len(snippets) > 0 {
		b.WriteString("\n\n## Available tools\n\n")
		for _, s := range snippets {
			b.WriteString("- ")
			b.WriteString(s)
			b.WriteString("\n")
		}
	}

	// Guidelines: per-tool text merged and deduplicated, then the always-on
	// rules. Dedup matters because several tools legitimately say "read before
	// you write" and the model should see it once.
	guidelines := dedupe(collectGuidelines(cfg.Tools))
	guidelines = append(guidelines, alwaysOnGuidelines...)
	if len(guidelines) > 0 {
		b.WriteString("\n## Guidelines\n\n")
		for _, g := range guidelines {
			b.WriteString("- ")
			b.WriteString(g)
			b.WriteString("\n")
		}
	}

	if len(cfg.Skills) > 0 {
		b.WriteString("\n")
		b.WriteString(FormatSkills(cfg.Skills))
		b.WriteString("\n")
	}

	if len(cfg.ProjectInstructions) > 0 {
		b.WriteString("\n<project_context>\n")
		for _, f := range cfg.ProjectInstructions {
			fmt.Fprintf(&b, "<project_instructions path=%q>\n", f.Path)
			b.WriteString(strings.TrimRight(f.Content, "\n"))
			b.WriteString("\n</project_instructions>\n")
		}
		b.WriteString("</project_context>\n")
	}

	if strings.TrimSpace(cfg.Append) != "" {
		b.WriteString("\n")
		b.WriteString(strings.TrimRight(cfg.Append, "\n"))
		b.WriteString("\n")
	}

	// Last line, always. Nothing time-varying may follow it.
	fmt.Fprintf(&b, "\nCurrent working directory: %s", cfg.Cwd)
	return b.String()
}

func collectGuidelines(tools []agent.Tool) []string {
	var out []string
	for _, t := range tools {
		if t.PromptGuidelines == "" {
			continue
		}
		// A tool may contribute several sentences; split so dedup works at the
		// sentence level rather than the blob level.
		out = append(out, splitSentences(t.PromptGuidelines)...)
	}
	return out
}

// splitSentences breaks guideline text on sentence boundaries, keeping the
// terminator. Deliberately simple: guidelines are authored text, not arbitrary
// prose, so full NLP sentence splitting would be effort spent on nothing.
func splitSentences(s string) []string {
	var out []string
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] != '.' && s[i] != '!' && s[i] != '?' {
			continue
		}
		// Only split when followed by a space or end of string, so "AGENTS.md"
		// and "e.g." survive intact.
		if i+1 < len(s) && s[i+1] != ' ' {
			continue
		}
		part := strings.TrimSpace(s[start : i+1])
		if part != "" {
			out = append(out, part)
		}
		start = i + 1
	}
	if rest := strings.TrimSpace(s[start:]); rest != "" {
		out = append(out, rest)
	}
	return out
}

func dedupe(items []string) []string {
	seen := make(map[string]bool, len(items))
	out := make([]string, 0, len(items))
	for _, s := range items {
		key := strings.ToLower(strings.TrimSpace(s))
		if key == "" || seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, s)
	}
	return out
}

// FormatSkills renders the skill index.
//
// Only name, description and location — never the body. That is the whole
// point of progressive disclosure: the index costs a line per skill in the
// cached prefix, and the model pays for a skill's body only on the turn it
// actually reads it.
func FormatSkills(skills []Skill) string {
	visible := make([]Skill, 0, len(skills))
	for _, s := range skills {
		if !s.Hidden {
			visible = append(visible, s)
		}
	}
	if len(visible) == 0 {
		return ""
	}
	// Deterministic order: the prompt must be byte-identical across runs, and
	// directory iteration order is not a promise any filesystem makes.
	sort.Slice(visible, func(i, j int) bool { return visible[i].Name < visible[j].Name })

	var b strings.Builder
	b.WriteString("## Skills\n\n")
	b.WriteString("These skills carry detailed instructions for specific tasks. ")
	b.WriteString("When a task matches a description, read the skill file with the read tool before starting. ")
	b.WriteString("Resolve relative paths inside a skill against that skill's own directory.\n\n")
	b.WriteString("<available_skills>\n")
	for _, s := range visible {
		b.WriteString("  <skill>\n")
		fmt.Fprintf(&b, "    <name>%s</name>\n", escapeXML(s.Name))
		fmt.Fprintf(&b, "    <description>%s</description>\n", escapeXML(s.Description))
		fmt.Fprintf(&b, "    <location>%s</location>\n", escapeXML(s.Path))
		b.WriteString("  </skill>\n")
	}
	b.WriteString("</available_skills>")
	return b.String()
}

func escapeXML(s string) string {
	r := strings.NewReplacer(
		"&", "&amp;", "<", "&lt;", ">", "&gt;", `"`, "&quot;", "'", "&apos;",
	)
	return r.Replace(s)
}
