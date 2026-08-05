// Command deepseek-pi is a coding agent for the DeepSeek v4 series.
package main

import (
	"bufio"
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/thevibeworks/deepseek-pi/agent"
	"github.com/thevibeworks/deepseek-pi/ai"
	"github.com/thevibeworks/deepseek-pi/harness"
)

func main() {
	if err := run(); err != nil {
		if errors.Is(err, context.Canceled) {
			os.Exit(130)
		}
		fmt.Fprintf(os.Stderr, "deepseek-pi: %v\n", err)
		os.Exit(1)
	}
}

type flags struct {
	prompt      string
	model       string
	effort      string
	cwd         string
	resume      string
	continueRun bool
	maxTokens   int
	yolo        bool
	plan        bool
	noSkills    bool
	noSubagents bool
	allow       listFlag
	quiet       bool
	thinking    bool
	showPrompt  bool
	listModels  bool
	sessions    bool
	version     bool
}

func parseFlags() *flags {
	f := &flags{}
	flag.StringVar(&f.prompt, "p", "", "run one prompt and exit (headless)")
	flag.StringVar(&f.model, "m", "", "model: flash (default) or pro")
	flag.StringVar(&f.model, "model", "", "model: flash (default) or pro")
	flag.StringVar(&f.effort, "e", "", "reasoning effort: low, high, xhigh, max (default: off)")
	flag.StringVar(&f.effort, "effort", "", "reasoning effort: low, high, xhigh, max (default: off)")
	flag.StringVar(&f.cwd, "C", "", "workspace directory (default: current)")
	flag.StringVar(&f.resume, "r", "", "resume a session by file path")
	flag.BoolVar(&f.continueRun, "c", false, "continue the most recent session in this workspace")
	flag.IntVar(&f.maxTokens, "max-tokens", 0, "cap output tokens per turn")
	flag.BoolVar(&f.yolo, "yolo", false, "run every tool without asking (sandboxes and CI)")
	flag.BoolVar(&f.plan, "plan", false, "read-only: investigate and propose, never modify")
	flag.BoolVar(&f.noSkills, "no-skills", false, "skip skill discovery, for a minimal prompt prefix")
	flag.BoolVar(&f.noSubagents, "no-subagents", false, "remove the task tool, for a flat one-agent shape")
	flag.Var(&f.allow, "allow", "tool to pre-approve; repeatable or comma-separated (edit,write)")
	flag.BoolVar(&f.quiet, "q", false, "print only the final answer")
	flag.BoolVar(&f.thinking, "show-thinking", false, "stream reasoning as it arrives")
	flag.BoolVar(&f.showPrompt, "show-prompt", false, "print the assembled system prompt and exit")
	flag.BoolVar(&f.listModels, "models", false, "list models with real limits and pricing")
	flag.BoolVar(&f.sessions, "sessions", false, "list stored sessions for this workspace")
	flag.BoolVar(&f.version, "version", false, "print the version and exit")

	flag.Usage = usage
	flag.Parse()

	// Any remaining args are a prompt, so `deepseek-pi fix the build` works.
	if f.prompt == "" && flag.NArg() > 0 {
		f.prompt = strings.Join(flag.Args(), " ")
	}
	return f
}

func usage() {
	fmt.Fprintf(os.Stderr, `%s %s — a coding agent for DeepSeek v4

Usage:
  deepseek-pi                      start an interactive session
  deepseek-pi -p "fix the build"   run one prompt and exit
  deepseek-pi "fix the build"      same, positional form
  deepseek-pi -c                   continue the most recent session here

Permissions:
  Reading and read-only shell always run. Anything that can change the
  machine asks first in an interactive session. A headless run (-p) has
  nobody to ask, so it refuses instead — pass -yolo or -allow to opt in.

Flags:
`, harness.Name, harness.Version)
	flag.PrintDefaults()
	fmt.Fprintf(os.Stderr, `
Environment:
  DEEPSEEK_API_KEY   required
  DEEPSEEK_PI_HOME   session storage root (default ~/.deepseek-pi)
  NO_COLOR           disable colour

In an interactive session, /help lists the slash commands.
`)
}

func run() error {
	f := parseFlags()

	switch {
	case f.version:
		fmt.Printf("%s %s\n", harness.Name, harness.Version)
		return nil
	case f.listModels:
		return printModels()
	case f.sessions:
		return printSessions(f.cwd)
	}

	// Ctrl-C cancels the in-flight run rather than killing the process, so a
	// long tool call can be interrupted without losing the session.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	resume := f.resume
	if f.continueRun && resume == "" {
		latest, err := latestSession(f.cwd)
		if err != nil {
			return err
		}
		if latest == "" {
			return fmt.Errorf("no previous session in this workspace")
		}
		resume = latest
	}

	mode, err := resolveMode(f)
	if err != nil {
		return err
	}

	s := newStyle(colorEnabled(os.Stdout))

	// One scanner for the whole process: the REPL and the approval prompt read
	// the same stdin, and two scanners would each buffer ahead and swallow the
	// other's input.
	stdin := bufio.NewScanner(os.Stdin)
	stdin.Buffer(make([]byte, 0, 64*1024), 1<<20)

	// Headless runs get NO approver. A prompt nobody can answer must not become
	// silent permission, so -p denies what it would otherwise ask about and
	// says how to opt in.
	var ask func(context.Context, agent.ToolCall, string) (bool, bool)
	if f.prompt == "" {
		ask = newApprover(stdin, os.Stdout, s).Ask
	}

	h, err := harness.New(ctx, harness.Options{
		Cwd:         f.cwd,
		Model:       normalizeModel(f.model),
		Effort:      ai.Effort(f.effort),
		MaxTokens:   f.maxTokens,
		Resume:      resume,
		NoSkills:    f.noSkills,
		NoSubagents: f.noSubagents,
		Mode:        mode,
		AllowTools:  f.allow,
		Approve:     ask,
	})
	if err != nil {
		return err
	}
	defer func() { _ = h.Close() }()

	if f.showPrompt {
		fmt.Println(h.SystemPrompt())
		return nil
	}

	r := newRenderer(os.Stdout, s)
	r.Quiet = f.quiet
	r.ShowThinking = f.thinking
	h.Agent.Subscribe(r.Handle)

	// Compaction is not a silent event. It changes what the model can see, so
	// a user watching a long session should be told when their earlier
	// conversation stopped being verbatim.
	record := h.Compactor.OnEvent
	h.Compactor.OnEvent = func(ev harness.CompactionEvent) {
		if record != nil {
			record(ev)
		}
		if !f.quiet {
			fmt.Fprintln(os.Stderr, formatCompaction(ev, s))
		}
	}

	if f.prompt != "" {
		return runOnce(ctx, h, r, f.prompt, s)
	}
	return runInteractive(ctx, h, r, s, stdin)
}

// resolveMode turns the mode flags into one posture, rejecting combinations
// that contradict each other rather than silently picking a winner.
func resolveMode(f *flags) (harness.Mode, error) {
	if f.yolo && f.plan {
		return "", fmt.Errorf("-yolo and -plan are opposites; pick one")
	}
	switch {
	case f.yolo:
		return harness.ModeYolo, nil
	case f.plan:
		return harness.ModePlan, nil
	}
	return harness.ModeDefault, nil
}

// listFlag accumulates a repeatable flag that also accepts comma-separated
// values, so --allow edit --allow write and --allow edit,write both work. A
// user guessing either form should not have to discover which one we chose.
type listFlag []string

func (l *listFlag) String() string { return strings.Join(*l, ",") }

func (l *listFlag) Set(v string) error {
	for _, part := range strings.Split(v, ",") {
		if part = strings.TrimSpace(part); part != "" {
			*l = append(*l, part)
		}
	}
	return nil
}

func normalizeModel(m string) string {
	switch strings.ToLower(strings.TrimSpace(m)) {
	case "", "flash":
		return ai.ModelFlash
	case "pro":
		return ai.ModelPro
	default:
		return m
	}
}

func runOnce(ctx context.Context, h *harness.Harness, r *renderer, prompt string, s style) error {
	_, err := h.Agent.Prompt(ctx, prompt)
	r.Finish()
	if err != nil {
		return err
	}
	if !r.Quiet {
		if line := formatUsage(h.Model, h.Session.Usage, s); line != "" {
			fmt.Fprintln(os.Stderr, line)
		}
	}
	return ctx.Err()
}

func runInteractive(ctx context.Context, h *harness.Harness, r *renderer, s style, in *bufio.Scanner) error {
	fmt.Printf("%s%s %s%s — %s, %s mode, workspace %s\n",
		s.bold, harness.Name, harness.Version, s.reset, h.Model.Name,
		h.Policy.Mode, h.Workspace.Root)
	if n := len(h.Skills); n > 0 {
		fmt.Printf("%s%d skill(s) available%s\n", s.dim, n, s.reset)
	}
	fmt.Printf("%sType a prompt, or /help for commands. Ctrl-D to exit.%s\n\n", s.dim, s.reset)

	for {
		fmt.Printf("%s>%s ", s.green, s.reset)
		if !in.Scan() {
			fmt.Println()
			return in.Err()
		}
		line := strings.TrimSpace(in.Text())
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "/") {
			done, err := handleCommand(h, r, line, s)
			if err != nil {
				fmt.Fprintf(os.Stderr, "%serror:%s %v\n", s.red, s.reset, err)
			}
			if done {
				return nil
			}
			continue
		}

		// A fresh cancel scope per turn: Ctrl-C should end the current run and
		// return to the prompt, not tear down the session.
		turnCtx, cancel := signal.NotifyContext(ctx, os.Interrupt)
		_, err := h.Agent.Prompt(turnCtx, line)
		r.Finish()
		cancel()

		if err != nil {
			fmt.Fprintf(os.Stderr, "%serror:%s %v\n", s.red, s.reset, err)
		}
		if line := formatUsage(h.Model, h.Session.Usage, s); line != "" {
			fmt.Println(line)
		}
		fmt.Println()

		if ctx.Err() != nil {
			return nil
		}
	}
}

// handleCommand runs a slash command. It reports whether the session should end.
func handleCommand(h *harness.Harness, r *renderer, line string, s style) (bool, error) {
	cmd, arg, _ := strings.Cut(strings.TrimPrefix(line, "/"), " ")
	arg = strings.TrimSpace(arg)

	switch cmd {
	case "help", "?":
		fmt.Print(`Commands:
  /mode [default|plan|yolo]
                       show or change what tools may run without asking
  /model [flash|pro]   show or change the model for the next turn
  /effort [off|low|high|xhigh|max]
                       show or change the reasoning level
  /cache               why the prompt cache missed, and what it cost
  /compact             summarize earlier turns now, freeing context
  /status              session accounting and configuration
  /prompt              print the assembled system prompt
  /session             path to this session's transcript
  /thinking            toggle streaming of reasoning
  /clear               forget the conversation, keep the session file
  /exit                leave
`)
	case "exit", "quit", "q":
		return true, nil

	case "mode":
		if arg == "" {
			fmt.Printf("%s\n", h.Policy.Mode)
			return false, nil
		}
		m, err := harness.ParseMode(arg)
		if err != nil {
			return false, err
		}
		// Only the permission changes; the tool set and therefore the cached
		// prompt prefix stay byte-identical, so switching modes mid-session
		// costs nothing.
		h.Policy.Mode = m
		fmt.Printf("mode: %s\n", m)

	case "model":
		if arg == "" {
			fmt.Printf("%s\n", h.Agent.Config().Model)
			return false, nil
		}
		id := normalizeModel(arg)
		m, ok := ai.Lookup(id)
		if !ok {
			return false, fmt.Errorf("unknown model %q", arg)
		}
		// A transcript is pinned to one model by design: nothing verifies that
		// thinking blocks generated by one model replay correctly under
		// another. Switching starts a fresh context rather than pretending.
		if len(h.Agent.Context().Messages) > 0 {
			h.Agent.Context().Messages = nil
			fmt.Printf("%sconversation cleared: a transcript is pinned to one model%s\n", s.dim, s.reset)
		}
		h.Agent.SetModel(m.ID)
		h.Model = m
		if err := h.Session.RecordModelChange(m.ID); err != nil {
			return false, err
		}
		fmt.Printf("model: %s\n", m.Name)

	case "effort":
		if arg == "" {
			e := h.Agent.Config().Effort
			if e == "" {
				e = "off"
			}
			fmt.Printf("%s\n", e)
			return false, nil
		}
		if arg == "off" {
			h.Agent.SetEffort(ai.EffortOff)
			fmt.Println("effort: off")
			return false, nil
		}
		e := ai.Effort(arg)
		if !e.Valid() || !h.Model.SupportsEffort(e) {
			return false, fmt.Errorf("unknown effort %q (low, high, xhigh, max, off)", arg)
		}
		h.Agent.SetEffort(e)
		fmt.Printf("effort: %s\n", e)

	case "cache":
		fmt.Print(h.Cache.Report())

	case "compact":
		actx := h.Agent.Context()
		before := harness.EstimateTokens(actx.Messages)
		update := h.Compactor.Compact(actx, "requested with /compact")
		if update == nil {
			fmt.Printf("%snothing to compact (%d tokens, %d messages)%s\n",
				s.dim, before, len(actx.Messages), s.reset)
			return false, nil
		}
		actx.Messages = update.Context.Messages

	case "status":
		printStatus(h, s)

	case "prompt":
		fmt.Println(h.SystemPrompt())

	case "session":
		fmt.Println(h.Session.Path())

	case "thinking":
		r.mu.Lock()
		r.ShowThinking = !r.ShowThinking
		state := "off"
		if r.ShowThinking {
			state = "on"
		}
		r.mu.Unlock()
		fmt.Printf("thinking display: %s\n", state)

	case "clear":
		h.Agent.Context().Messages = nil
		fmt.Println("conversation cleared")

	default:
		return false, fmt.Errorf("unknown command /%s (try /help)", cmd)
	}
	return false, nil
}

func printStatus(h *harness.Harness, s style) {
	cfg := h.Agent.Config()
	u := h.Session.Usage
	effort := string(cfg.Effort)
	if effort == "" {
		effort = "off"
	}

	fmt.Printf("%smodel%s        %s (%s)\n", s.bold, s.reset, h.Model.Name, h.Model.ID)
	fmt.Printf("%seffort%s       %s\n", s.bold, s.reset, effort)
	fmt.Printf("%smode%s         %s\n", s.bold, s.reset, h.Policy.Mode)
	fmt.Printf("%sworkspace%s    %s\n", s.bold, s.reset, h.Workspace.Root)
	fmt.Printf("%ssession%s      %s\n", s.bold, s.reset, h.Session.Path())
	fmt.Printf("%smessages%s     %d\n", s.bold, s.reset, len(h.Agent.Context().Messages))
	fmt.Printf("%sprompt%s       %s (system)\n", s.bold, s.reset, formatBytes(len(h.SystemPrompt())))

	// Context usage is measured against the REAL window, not the advertised
	// one. Budgeting against 1M when 616k is usable is how a harness hits hard
	// truncation while reporting 62% utilisation.
	live := harness.EstimateTokens(h.Agent.Context().Messages)
	pct := float64(live) / float64(h.Model.ContextWindow) * 100
	fmt.Printf("%scontext%s      %d / %d tokens (%.1f%% of the usable window; %d advertised)\n",
		s.bold, s.reset, live, h.Model.ContextWindow, pct, h.Model.AdvertisedWindow)
	fmt.Printf("%scompact at%s   %d tokens\n", s.bold, s.reset, h.Compactor.Threshold())
	fmt.Printf("%stokens%s       %d in / %d out · cache read %d (%.0f%%)\n",
		s.bold, s.reset, u.Input, u.Output, u.CacheRead, u.CacheHitRate()*100)
	fmt.Printf("%scost%s         $%.4f (cache saved $%.4f)\n",
		s.bold, s.reset, u.Cost.Total, h.Model.CacheSavings(u))
	if children := h.Children(); len(children) > 0 {
		var childCost float64
		var childTokens int
		for _, c := range children {
			childCost += c.Usage.Cost.Total
			childTokens += c.Usage.Input + c.Usage.Output
		}
		// Sub-agent cost is reported separately because it does NOT appear in
		// the session usage above: children have their own transcripts, and
		// folding them in would make the parent's context look enormous.
		fmt.Printf("%ssub-agents%s   %d run · %d tokens · $%.4f (separate transcripts)\n",
			s.bold, s.reset, len(children), childTokens, childCost)
	}
	if n := len(h.Cache.Breaks); n > 0 {
		fmt.Printf("%scache breaks%s %d, %d tokens re-billed (/cache for why)\n",
			s.bold, s.reset, n, h.Cache.WastedTokens)
	}

	if len(h.Instructions) > 0 {
		fmt.Printf("%sinstructions%s\n", s.bold, s.reset)
		for _, f := range h.Instructions {
			fmt.Printf("  %s\n", f.Path)
		}
	}
	if len(h.Skills) > 0 {
		// Show what the index costs. Skills are progressive-disclosure, so only
		// this index sits in the prefix — but with a large personal collection
		// it can still be most of the prompt, and that should not be a mystery.
		index := harness.FormatSkills(h.Skills)
		fmt.Printf("%sskills%s       %d (index %s, %.0f%% of the prompt)\n",
			s.bold, s.reset, len(h.Skills), formatBytes(len(index)),
			float64(len(index))/float64(len(h.SystemPrompt()))*100)
	}
}

func printModels() error {
	fmt.Printf("%-20s %-22s %10s %10s %12s %12s %12s\n",
		"ID", "NAME", "CONTEXT", "OUTPUT", "$/M CACHED", "$/M INPUT", "$/M OUTPUT")
	for _, m := range ai.Models() {
		fmt.Printf("%-20s %-22s %10d %10d %12.4f %12.4f %12.4f\n",
			m.ID, m.Name, m.ContextWindow, m.MaxTokens,
			m.Rates.CacheRead, m.Rates.Input, m.Rates.Output)
	}
	fmt.Printf("\nCONTEXT is the usable input budget; both models advertise %d.\n",
		ai.Models()[0].AdvertisedWindow)
	return nil
}

func printSessions(cwd string) error {
	if cwd == "" {
		var err error
		cwd, err = os.Getwd()
		if err != nil {
			return err
		}
	}
	list, err := harness.ListSessions(cwd)
	if err != nil {
		return err
	}
	if len(list) == 0 {
		fmt.Println("no sessions for this workspace")
		return nil
	}
	for _, s := range list {
		fmt.Printf("%s  %-19s  %3d msg  %s\n",
			s.ID, s.Modified.Format(time.DateTime), s.Messages, s.Preview)
		fmt.Printf("  %s\n", s.Path)
	}
	return nil
}

func latestSession(cwd string) (string, error) {
	if cwd == "" {
		var err error
		cwd, err = os.Getwd()
		if err != nil {
			return "", err
		}
	}
	list, err := harness.ListSessions(cwd)
	if err != nil || len(list) == 0 {
		return "", err
	}
	return list[0].Path, nil
}
