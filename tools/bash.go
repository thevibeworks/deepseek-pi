package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"

	"github.com/thevibeworks/deepseek-pi/agent"
	"github.com/thevibeworks/deepseek-pi/ai"
)

// BashArgs are the bash tool's parameters.
type BashArgs struct {
	Command string `json:"command"`
	Timeout int    `json:"timeout"`
	Cwd     string `json:"cwd"`
}

// DefaultBashTimeout applies when the model does not set one. A command with no
// ceiling can park a session forever, and the model rarely thinks to set one.
const DefaultBashTimeout = 120 * time.Second

// MaxBashTimeout bounds what the model may ask for.
const MaxBashTimeout = 30 * time.Minute

// Bash builds the shell tool.
//
// This tool deliberately absorbs search and navigation — grep, find, ls, cat,
// sed — instead of shipping a separate tool for each. Every tool schema lives
// in the cached prompt prefix, so a 7-schema tool set is materially cheaper per
// turn than a 40-schema one, and the model already knows shell far better than
// it knows any bespoke search schema.
func Bash(w *Workspace, env []string) agent.Tool {
	return agent.Tool{
		Name:  "bash",
		Label: "Bash",
		Description: fmt.Sprintf(
			"Run a shell command in the workspace. Use this for search and navigation too: "+
				"rg/grep to search, fd/find to locate files, ls to list, sed/awk to slice.\n"+
				"Output is bounded to the last %d lines or %s; anything larger spills to a temp "+
				"file whose path is reported, so nothing is lost.\n"+
				"stdin is /dev/null: commands never block waiting for input. "+
				"Default timeout %ds, maximum %ds.",
			DefaultMaxLines, FormatSize(DefaultMaxBytes),
			int(DefaultBashTimeout.Seconds()), int(MaxBashTimeout.Seconds())),
		Parameters: ai.Object(map[string]*ai.Schema{
			"command": ai.Str("The shell command to run."),
			"timeout": ai.Int("Timeout in seconds. Omit for the default."),
			"cwd":     ai.Str("Working directory, relative to the workspace root. Omit for the root."),
		}, "command"),
		PromptSnippet: "bash(command, timeout?, cwd?) — run a shell command; also your search and navigation tool",
		PromptGuidelines: "Use bash for searching (rg), finding files (fd/find) and listing (ls) rather than asking for a dedicated tool. " +
			"Prefer one composed command over several round trips. Quote paths that may contain spaces. " +
			"Do not use bash to read or edit files when read and edit will do; they are cheaper and give better errors.",
		ExecutionMode: agent.ModeParallel,
		PrepareArguments: agent.HealArguments(map[string]string{
			"cmd": "command", "script": "command", "shell": "command",
			"timeout_seconds": "timeout", "timeoutMs": "timeout", "dir": "cwd", "workdir": "cwd",
		}),
		Execute: func(ctx context.Context, call agent.ToolCall, onUpdate agent.UpdateFunc) (agent.ToolResult, error) {
			var args BashArgs
			if err := json.Unmarshal(call.Arguments, &args); err != nil {
				return agent.ToolResult{}, fmt.Errorf("invalid arguments: %w", err)
			}
			if strings.TrimSpace(args.Command) == "" {
				return agent.ToolResult{}, fmt.Errorf("command must not be empty")
			}

			dir := w.Root
			if args.Cwd != "" {
				resolved, err := w.Resolve(args.Cwd)
				if err != nil {
					return agent.ToolResult{}, err
				}
				dir = resolved
			}

			timeout := DefaultBashTimeout
			if args.Timeout > 0 {
				timeout = time.Duration(args.Timeout) * time.Second
				if timeout > MaxBashTimeout {
					timeout = MaxBashTimeout
				}
			}

			return runShell(ctx, shellRequest{
				Command:  args.Command,
				Dir:      dir,
				Env:      env,
				Timeout:  timeout,
				OnUpdate: onUpdate,
			})
		},
	}
}

type shellRequest struct {
	Command  string
	Dir      string
	Env      []string
	Timeout  time.Duration
	OnUpdate agent.UpdateFunc
}

func runShell(parent context.Context, req shellRequest) (agent.ToolResult, error) {
	ctx, cancel := context.WithTimeout(parent, req.Timeout)
	defer cancel()

	cmd := exec.Command("bash", "-c", req.Command)
	cmd.Dir = req.Dir
	cmd.Env = append(os.Environ(), req.Env...)
	// stdin is /dev/null on purpose. A command that waits for input would
	// otherwise hang until the timeout, burning the whole budget on a prompt
	// no one can answer.
	cmd.Stdin = nil
	setProcessGroup(cmd)

	acc := newOutputAccumulator(req.OnUpdate)
	cmd.Stdout = acc
	cmd.Stderr = acc

	start := time.Now()
	if err := cmd.Start(); err != nil {
		return agent.ToolResult{}, fmt.Errorf("starting command: %w", err)
	}

	waitErr := make(chan error, 1)
	go func() { waitErr <- cmd.Wait() }()

	var runErr error
	select {
	case runErr = <-waitErr:
	case <-ctx.Done():
		// Kill the whole process group. Killing only the shell leaves its
		// children running and holding the pipe open, so the read never ends.
		killProcessGroup(cmd)
		<-waitErr
		runErr = ctx.Err()
	}
	elapsed := time.Since(start)

	acc.finish()
	output, truncation, spillPath := acc.result()

	exitCode := 0
	if cmd.ProcessState != nil {
		exitCode = cmd.ProcessState.ExitCode()
	}

	var b strings.Builder
	if output == "" {
		b.WriteString("(no output)")
	} else {
		b.WriteString(output)
	}
	if truncation.Truncated {
		hint := "Re-run with a narrower command (grep, head, tail) to see the part you need."
		if spillPath != "" {
			hint = fmt.Sprintf("Full output saved to %s — read it with offset/limit.", spillPath)
		}
		b.WriteString(truncation.Notice(hint))
	}

	details := map[string]any{
		"exitCode": exitCode, "durationMs": elapsed.Milliseconds(),
		"truncated": truncation.Truncated, "totalBytes": truncation.TotalBytes,
	}
	if spillPath != "" {
		details["fullOutputPath"] = spillPath
	}

	// A failed command is an error result, not a successful one carrying a
	// non-zero code in prose. The model reacts to the error flag; it routinely
	// misses "exit code: 1" buried under 80 lines of build log.
	switch {
	case parent.Err() != nil:
		return agent.ToolResult{}, fmt.Errorf("command cancelled after %s\n%s", elapsed.Round(time.Millisecond), b.String())
	case ctx.Err() != nil:
		return agent.ToolResult{}, fmt.Errorf("command timed out after %s\n%s", req.Timeout, b.String())
	case runErr != nil:
		return agent.ToolResult{}, fmt.Errorf("command failed (exit %d)\n%s", exitCode, b.String())
	}

	return agent.ToolResult{
		Content: []ai.Content{ai.TextContent(b.String())},
		Details: details,
	}, nil
}

// outputAccumulator bounds streaming command output.
//
// Shape ported from pi's OutputAccumulator, and the ordering matters: it keeps
// a rolling tail in memory and opens the spill file THE MOMENT limits are
// exceeded, while the command is still running. Truncating after the fact
// cannot work — by then the bytes are gone. This is what makes "spill, don't
// truncate" true for the tool that produces most of the oversized output.
type outputAccumulator struct {
	mu sync.Mutex

	tail       strings.Builder
	totalBytes int
	totalLines int
	openLine   bool

	spill     *os.File
	spillPath string
	spillErr  error

	onUpdate   agent.UpdateFunc
	lastUpdate time.Time
}

// tailBudget is how much output stays in memory: twice the reportable limit, so
// tail truncation always has enough material to fill the budget exactly.
const tailBudget = 2 * DefaultMaxBytes

func newOutputAccumulator(onUpdate agent.UpdateFunc) *outputAccumulator {
	return &outputAccumulator{onUpdate: onUpdate}
}

func (a *outputAccumulator) Write(p []byte) (int, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	text := sanitizeBinary(string(p))
	a.totalBytes += len(text)
	newlines := strings.Count(text, "\n")
	a.totalLines += newlines
	if idx := strings.LastIndexByte(text, '\n'); idx >= 0 {
		a.openLine = idx != len(text)-1
	} else if len(text) > 0 {
		a.openLine = true
	}

	a.tail.WriteString(text)
	if a.tail.Len() > tailBudget {
		trimmed := tailBytes(a.tail.String(), tailBudget)
		a.tail.Reset()
		a.tail.WriteString(trimmed)
	}

	if a.exceeded() {
		a.ensureSpill(text)
	}

	// Throttle UI updates; a build log emits thousands of writes a second and
	// a renderer does not need every one.
	if a.onUpdate != nil && time.Since(a.lastUpdate) > 100*time.Millisecond {
		a.lastUpdate = time.Now()
		snapshot := a.tail.String()
		go a.onUpdate(agent.Text(snapshot))
	}
	return len(p), nil
}

func (a *outputAccumulator) exceeded() bool {
	lines := a.totalLines
	if a.openLine {
		lines++
	}
	return a.totalBytes > DefaultMaxBytes || lines > DefaultMaxLines
}

// ensureSpill opens the spill file on first overflow, seeding it with the tail
// captured so far, then appends every later chunk.
func (a *outputAccumulator) ensureSpill(latest string) {
	if a.spillErr != nil {
		return
	}
	if a.spill == nil {
		f, err := os.CreateTemp("", "deepseek-pi-bash-*.log")
		if err != nil {
			a.spillErr = err
			return
		}
		a.spill, a.spillPath = f, f.Name()
		// Seed with everything still in the rolling tail. Output already
		// evicted from the tail before the first overflow is unrecoverable,
		// which is why the tail budget is twice the reportable limit.
		if _, err := f.WriteString(a.tail.String()); err != nil {
			a.spillErr = err
		}
		return
	}
	if _, err := a.spill.WriteString(latest); err != nil {
		a.spillErr = err
	}
}

func (a *outputAccumulator) finish() {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.spill != nil {
		_ = a.spill.Close()
	}
}

func (a *outputAccumulator) result() (string, Truncation, string) {
	a.mu.Lock()
	defer a.mu.Unlock()

	tr := TruncateTail(a.tail.String(), Limits{})
	lines := a.totalLines
	if a.openLine {
		lines++
	}
	// The rolling tail already dropped earlier output, so totals come from the
	// running counters rather than from what happens to be in memory.
	tr.TotalLines = lines
	tr.TotalBytes = a.totalBytes
	if a.totalBytes > DefaultMaxBytes || lines > DefaultMaxLines {
		tr.Truncated = true
		if tr.TruncatedBy == "" {
			tr.TruncatedBy = "bytes"
			if lines > DefaultMaxLines {
				tr.TruncatedBy = "lines"
			}
		}
	}
	return tr.Content, tr, a.spillPath
}

// sanitizeBinary strips control bytes that corrupt a terminal or a transcript,
// keeping tab, newline and carriage return.
func sanitizeBinary(s string) string {
	if !strings.ContainsFunc(s, isControlNoise) {
		return strings.ReplaceAll(s, "\r", "")
	}
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range s {
		if r == '\r' {
			continue
		}
		if isControlNoise(r) {
			continue
		}
		b.WriteRune(r)
	}
	return b.String()
}

func isControlNoise(r rune) bool {
	if r == '\t' || r == '\n' {
		return false
	}
	if r < 0x20 {
		return true
	}
	// Interlinear annotation marks: invisible and confusing in a transcript.
	return r >= 0xFFF9 && r <= 0xFFFB
}
