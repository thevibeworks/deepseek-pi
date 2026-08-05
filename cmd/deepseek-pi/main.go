// Command deepseek-pi is a coding agent for the DeepSeek v4 series.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strconv"
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
	maxCost     float64
	maxTurns    int
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
	prune       bool
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
	flag.Float64Var(&f.maxCost, "max-cost", 0, "stop the session after spending this many USD, sub-agents included")
	flag.IntVar(&f.maxTurns, "max-turns", 0, "stop a single request after this many turns")
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
	flag.BoolVar(&f.prune, "prune", false, "delete checkpoint data no session still refers to")
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

Budgets:
  -max-cost bounds the whole run in dollars, sub-agents included.
  -max-turns bounds one request, which is what catches a loop. Either
  stops at a turn boundary and exits non-zero under -p; interactively,
  /budget raises the limit and continues.

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
	s := newStyle(colorEnabled(os.Stdout))

	switch {
	case f.version:
		fmt.Printf("%s %s\n", harness.Name, harness.Version)
		return nil
	case f.listModels:
		return printModels()
	case f.sessions:
		return printSessions(f.cwd)
	case f.prune:
		return pruneBlobs(f.cwd)
	}

	// SIGTERM ends the process. SIGINT is deliberately NOT handled here: it
	// belongs to whatever is currently running, and a context cancelled at the
	// top would take the session down with it. See interrupter.
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM)
	defer stop()

	interrupts := newInterrupter(os.Stderr, s)
	defer interrupts.stop()

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

	stdin := newInput(os.Stdin, os.Stdout)
	defer stdin.Close()

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
		MaxCost:     f.maxCost,
		MaxTurns:    f.maxTurns,
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
		return runOnce(ctx, h, r, f.prompt, s, interrupts)
	}
	return runInteractive(ctx, h, r, s, stdin, interrupts)
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

func runOnce(ctx context.Context, h *harness.Harness, r *renderer, prompt string, s style, in *interrupter) error {
	// A headless run has one thing in flight, so Ctrl-C means cancel it.
	runCtx, done := in.arm(ctx)
	_, err := h.Prompt(runCtx, prompt)
	done()
	r.Finish()
	if err != nil {
		// A budget stop is returned as an error on purpose: a script that pipes
		// the answer somewhere must not treat a truncated one as complete.
		return err
	}
	if !r.Quiet {
		if line := formatUsage(h.Model, h.Session.Usage, s); line != "" {
			fmt.Fprintln(os.Stderr, line)
		}
	}
	return ctx.Err()
}

func runInteractive(
	ctx context.Context, h *harness.Harness, r *renderer, s style,
	in *input, sig *interrupter,
) error {
	fmt.Printf("%s%s %s%s — %s, %s mode, workspace %s\n",
		s.bold, harness.Name, harness.Version, s.reset, h.Model.Name,
		h.Policy.Mode, h.Workspace.Root)
	if n := len(h.Skills); n > 0 {
		fmt.Printf("%s%d skill(s) available%s\n", s.dim, n, s.reset)
	}
	fmt.Printf("%sType a prompt, or /help for commands. Ctrl-D to exit.%s\n\n", s.dim, s.reset)

	// Only the REPL brackets pastes. A headless run never reaches here, and
	// writing terminal modes it does not use would be output nobody asked for.
	in.Bracket()

	for {
		fmt.Printf("%s>%s ", s.green, s.reset)
		line, ok := in.Prompt()
		if !ok {
			fmt.Println()
			return in.Err()
		}
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		// A slash command is a single line by construction, so a pasted block
		// that merely starts with "/" is a prompt, not a command.
		if strings.HasPrefix(line, "/") && !strings.Contains(line, "\n") {
			done, err := handleCommand(h, r, line, s)
			if err != nil {
				fmt.Fprintf(os.Stderr, "%serror:%s %v\n", s.red, s.reset, err)
			}
			if done {
				return nil
			}
			continue
		}

		// Ctrl-C is armed only while a turn is in flight, so it ends the run and
		// returns to the prompt instead of tearing down the session.
		// Usage is cumulative, so it is only worth reprinting when this request
		// actually moved it. A refused one that spent nothing would otherwise
		// echo the previous total and read as if it had cost that again.
		before := h.Session.Usage.Total()

		turnCtx, done := sig.arm(ctx)
		_, err := h.Prompt(turnCtx, line)
		r.Finish()
		done()

		var stop harness.BudgetStop
		switch {
		case errors.As(err, &stop):
			// Not a failure: the work up to here is real and kept. Say what
			// stopped it and how to continue, in one line.
			fmt.Fprintf(os.Stderr, "%s%v%s\n", s.yellow, stop, s.reset)
		case err != nil:
			fmt.Fprintf(os.Stderr, "%serror:%s %v\n", s.red, s.reset, err)
		}
		if h.Session.Usage.Total() != before {
			if line := formatUsage(h.Model, h.Session.Usage, s); line != "" {
				fmt.Println(line)
			}
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
  /budget [cost N|turns N|off]
                       show or change the spending limits
  /compact             summarize earlier turns now, freeing context
  /turns               list the turns you can rewind or fork at
  /rewind [N]          drop turn N onward and restore the files it changed
  /fork [N]            branch into a copy cut before turn N, original kept
  /status              session accounting and configuration
  /prompt              print the assembled system prompt
  /session             path to this session's transcript
  /thinking            toggle streaming of reasoning
  /clear               forget the conversation, keep the transcript on disk
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

	case "budget":
		return false, budgetCommand(h, arg, s)

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

	case "turns":
		printTurns(h, s)

	case "rewind", "fork":
		n, err := parseTurn(arg)
		if err != nil {
			return false, err
		}
		branch := h.Rewind
		if cmd == "fork" {
			branch = h.Fork
		}
		rep, err := branch(n)
		if err != nil {
			return false, err
		}
		printBranch(rep, s)

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
		rep, err := h.Clear()
		if err != nil {
			return false, err
		}
		fmt.Println("conversation cleared")
		printChanged(rep, s)

	default:
		return false, fmt.Errorf("unknown command /%s (try /help)", cmd)
	}
	return false, nil
}

// budgetCommand shows or changes the session limits.
//
// Raising a limit is how a user answers a stop, so the write path has to be one
// line at the prompt. Anything that required a restart would make the budget a
// thing to avoid setting rather than a thing to use.
func budgetCommand(h *harness.Harness, arg string, s style) error {
	limits := h.Budget.Limits()

	if arg != "" {
		axis, value, _ := strings.Cut(arg, " ")
		value = strings.TrimSpace(value)
		switch axis {
		case "off":
			limits = harness.SessionBudget{}
		case "cost":
			n, err := strconv.ParseFloat(value, 64)
			if err != nil || n < 0 {
				return fmt.Errorf("cost must be a number of dollars, got %q", value)
			}
			limits.MaxCost = n
		case "turns":
			n, err := strconv.Atoi(value)
			if err != nil || n < 0 {
				return fmt.Errorf("turns must be a whole number, got %q", value)
			}
			limits.MaxTurns = n
		default:
			return fmt.Errorf("unknown budget %q (try: cost N, turns N, off)", axis)
		}
		h.Budget.SetLimits(limits)
	}

	spent := h.Budget.Spent()
	if limits.Empty() {
		fmt.Printf("%sno budget · spent %s so far%s\n", s.dim, harness.Money(spent), s.reset)
		return nil
	}
	if limits.MaxCost > 0 {
		fmt.Printf("%scost%s   %s of %s (%.0f%% used)\n",
			s.bold, s.reset, harness.Money(spent), harness.Money(limits.MaxCost),
			spent/limits.MaxCost*100)
	} else {
		fmt.Printf("%scost%s   %s spent, no limit\n", s.bold, s.reset, harness.Money(spent))
	}
	if limits.MaxTurns > 0 {
		fmt.Printf("%sturns%s  %d per request\n", s.bold, s.reset, limits.MaxTurns)
	} else {
		fmt.Printf("%sturns%s  no limit per request\n", s.bold, s.reset)
	}
	return nil
}

// parseTurn reads an optional turn number. Empty means "the default one for
// this command", which Rewind and Fork each interpret for themselves.
func parseTurn(arg string) (int, error) {
	if arg == "" {
		return 0, nil
	}
	n, err := strconv.Atoi(arg)
	if err != nil || n < 1 {
		return 0, fmt.Errorf("turn must be a positive number, got %q (/turns lists them)", arg)
	}
	return n, nil
}

func printTurns(h *harness.Harness, s style) {
	turns := harness.Turns(h.Agent.Context().Messages)
	if len(turns) == 0 {
		fmt.Printf("%sno turns yet%s\n", s.dim, s.reset)
		return
	}
	changed := false
	for _, t := range turns {
		mark := " "
		if t.Changed() {
			mark, changed = "*", true
		}
		label := t.Prompt
		if t.Synthetic {
			label = s.dim + "(compacted summary)" + s.reset
		}
		fmt.Printf("%s%3d%s %s %s\n", s.bold, t.Number, s.reset, mark, label)
	}
	if changed {
		fmt.Printf("%s* changed files or ran commands%s\n", s.dim, s.reset)
	}
	fmt.Printf("%s/rewind N drops turn N onward; /fork N branches before it%s\n", s.dim, s.reset)
}

// printBranch reports a rewind or fork, and says plainly what it did not undo.
//
// The workspace warning is not a footnote. Rewinding looks like undo, and a
// user who believes the files went back too will act on a workspace that does
// not match the conversation.
func printBranch(rep harness.BranchReport, s style) {
	where := fmt.Sprintf("before turn %d", rep.Turn)
	if rep.Turn == 0 {
		where = "at the current point"
	}
	verb := "rewound to"
	if rep.Forked {
		verb = "forked"
	}
	files := ""
	if n := rep.Files.Reverted(); n > 0 {
		files = fmt.Sprintf(", %d file(s) restored", n)
	}
	fmt.Printf("%s %s — %d turn(s) kept, %d dropped%s\n",
		verb, where, rep.Retained, len(rep.Discarded), files)
	if rep.Forked {
		fmt.Printf("%snow writing %s; the branch you left is unchanged%s\n",
			s.dim, rep.Session, s.reset)
	}

	printChanged(rep, s)
}

// printChanged reports the workspace side of a cut: what went back, and what
// did not. Silence would read as "nothing happened", which is the one wrong
// conclusion a user can draw.
func printChanged(rep harness.BranchReport, s style) {
	for _, c := range rep.Files.Changes {
		switch c.Action {
		case "reverted":
			fmt.Printf("  %sreverted%s %s\n", s.green, s.reset, c.Path)
		case "removed":
			fmt.Printf("  %sremoved%s  %s\n", s.green, s.reset, c.Path)
		}
	}

	// Everything below is what the cut could NOT undo. Shell commands are the
	// permanent limit; a kept file is a specific refusal with a reason.
	var stale []string
	for _, t := range rep.Changed() {
		for _, cmd := range t.Ran {
			stale = append(stale, fmt.Sprintf("turn %d ran %s", t.Number, cmd))
		}
		for _, role := range t.Delegated {
			stale = append(stale, fmt.Sprintf("turn %d delegated to a %s sub-agent", t.Number, role))
		}
	}
	kept := rep.Files.Kept()
	if len(stale) == 0 && len(kept) == 0 {
		return
	}

	fmt.Printf("%snot undone:%s\n", s.yellow, s.reset)
	for _, k := range kept {
		fmt.Printf("  kept %s — %s\n", k.Path, k.Why)
	}
	for _, line := range stale {
		fmt.Printf("  %s\n", line)
	}
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
	if limits := h.Budget.Limits(); !limits.Empty() {
		var parts []string
		if limits.MaxCost > 0 {
			parts = append(parts, fmt.Sprintf("%s of %s",
				harness.Money(h.Budget.Spent()), harness.Money(limits.MaxCost)))
		}
		if limits.MaxTurns > 0 {
			parts = append(parts, fmt.Sprintf("%d turns per request", limits.MaxTurns))
		}
		fmt.Printf("%sbudget%s       %s\n", s.bold, s.reset, strings.Join(parts, " · "))
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
		if s.Parent != "" {
			fmt.Printf("  forked from %s\n", s.Parent)
		}
	}
	return nil
}

// pruneBlobs reclaims checkpoint storage. Snapshots are written on every file
// change and nothing else removes them, so a long-lived workspace needs a way
// to let go of the ones no session can still rewind to.
func pruneBlobs(cwd string) error {
	if cwd == "" {
		var err error
		cwd, err = os.Getwd()
		if err != nil {
			return err
		}
	}
	removed, freed, err := harness.PruneBlobs(cwd)
	if err != nil {
		return err
	}
	if removed == 0 {
		fmt.Println("nothing to prune")
		return nil
	}
	fmt.Printf("removed %d unreferenced blob(s), %s\n", removed, formatBytes(int(freed)))
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
