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
	flag.BoolVar(&f.yolo, "yolo", false, "skip tool approval prompts")
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

	h, err := harness.New(ctx, harness.Options{
		Cwd:       f.cwd,
		Model:     normalizeModel(f.model),
		Effort:    ai.Effort(f.effort),
		MaxTokens: f.maxTokens,
		Resume:    resume,
	})
	if err != nil {
		return err
	}
	defer func() { _ = h.Close() }()

	if f.showPrompt {
		fmt.Println(h.SystemPrompt())
		return nil
	}

	s := newStyle(colorEnabled(os.Stdout))
	r := newRenderer(os.Stdout, s)
	r.Quiet = f.quiet
	r.ShowThinking = f.thinking
	h.Agent.Subscribe(r.Handle)

	if f.prompt != "" {
		return runOnce(ctx, h, r, f.prompt, s)
	}
	return runInteractive(ctx, h, r, s)
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

func runInteractive(ctx context.Context, h *harness.Harness, r *renderer, s style) error {
	fmt.Printf("%s%s %s%s — %s, workspace %s\n",
		s.bold, harness.Name, harness.Version, s.reset, h.Model.Name, h.Workspace.Root)
	if n := len(h.Skills); n > 0 {
		fmt.Printf("%s%d skill(s) available%s\n", s.dim, n, s.reset)
	}
	fmt.Printf("%sType a prompt, or /help for commands. Ctrl-D to exit.%s\n\n", s.dim, s.reset)

	in := bufio.NewScanner(os.Stdin)
	in.Buffer(make([]byte, 0, 64*1024), 1<<20)

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
  /model [flash|pro]   show or change the model for the next turn
  /effort [off|low|high|xhigh|max]
                       show or change the reasoning level
  /status              session accounting and configuration
  /prompt              print the assembled system prompt
  /session             path to this session's transcript
  /thinking            toggle streaming of reasoning
  /clear               forget the conversation, keep the session file
  /exit                leave
`)
	case "exit", "quit", "q":
		return true, nil

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
	fmt.Printf("%sworkspace%s    %s\n", s.bold, s.reset, h.Workspace.Root)
	fmt.Printf("%ssession%s      %s\n", s.bold, s.reset, h.Session.Path())
	fmt.Printf("%smessages%s     %d\n", s.bold, s.reset, len(h.Agent.Context().Messages))
	fmt.Printf("%sprompt%s       %s (system)\n", s.bold, s.reset, formatBytes(len(h.SystemPrompt())))

	// Context usage is measured against the REAL window, not the advertised
	// one. Budgeting against 1M when 616k is usable is how a harness hits hard
	// truncation while reporting 62% utilisation.
	pct := float64(u.Input) / float64(h.Model.ContextWindow) * 100
	fmt.Printf("%scontext%s      %d / %d tokens (%.1f%% of the usable window; %d advertised)\n",
		s.bold, s.reset, u.Input, h.Model.ContextWindow, pct, h.Model.AdvertisedWindow)
	fmt.Printf("%stokens%s       %d in / %d out · cache read %d (%.0f%%)\n",
		s.bold, s.reset, u.Input, u.Output, u.CacheRead, u.CacheHitRate()*100)
	fmt.Printf("%scost%s         $%.4f (cache saved $%.4f)\n",
		s.bold, s.reset, u.Cost.Total, h.Model.CacheSavings(u))

	if len(h.Instructions) > 0 {
		fmt.Printf("%sinstructions%s\n", s.bold, s.reset)
		for _, f := range h.Instructions {
			fmt.Printf("  %s\n", f.Path)
		}
	}
	if len(h.Skills) > 0 {
		fmt.Printf("%sskills%s       %d\n", s.bold, s.reset, len(h.Skills))
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
