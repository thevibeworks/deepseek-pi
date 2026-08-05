// Package eval measures the agent on real tasks.
//
// It exists because the project's own rule is that engine changes must be
// gated on numbers, and without a harness that rule is decoration. Every claim
// about token efficiency, cache behaviour or success rate should come from
// here rather than from a plausible-sounding argument.
//
// A task is a directory: a workspace to copy, a prompt to run, and a command
// that must exit zero afterwards. The verifier is a real command on purpose —
// "did the tests pass" is a fact, while "does the diff look right" is an
// opinion, and an eval built on opinions cannot gate anything.
package eval

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/thevibeworks/deepseek-pi/agent"
	"github.com/thevibeworks/deepseek-pi/ai"
	"github.com/thevibeworks/deepseek-pi/harness"
)

// Task is one benchmark case.
type Task struct {
	// Name identifies the task; defaults to the directory name.
	Name string `json:"name"`
	// Prompt is what the agent is asked to do.
	Prompt string `json:"prompt"`
	// Verify must exit zero for the task to count as passed. It runs in the
	// task's working copy after the agent finishes.
	Verify string `json:"verify"`
	// Setup runs before the agent, for fixture preparation that does not belong
	// in version control.
	Setup string `json:"setup,omitempty"`
	// TimeoutSeconds bounds the agent run. Zero uses DefaultTimeout.
	TimeoutSeconds int `json:"timeoutSeconds,omitempty"`
	// Effort overrides the reasoning level for this task.
	Effort string `json:"effort,omitempty"`

	// Dir is the task directory on disk, filled in by LoadTasks.
	Dir string `json:"-"`
}

// DefaultTimeout bounds a single task.
const DefaultTimeout = 5 * time.Minute

// Result is one task's outcome.
//
// Tokens and cost are recorded raw so a run can be re-priced later. A rate card
// changes; the token counts a run actually consumed do not.
type Result struct {
	Task       string    `json:"task"`
	Passed     bool      `json:"passed"`
	Model      string    `json:"model"`
	Effort     string    `json:"effort,omitempty"`
	Turns      int       `json:"turns"`
	ToolCalls  int       `json:"toolCalls"`
	ToolErrors int       `json:"toolErrors"`
	Input      int       `json:"inputTokens"`
	Output     int       `json:"outputTokens"`
	CacheRead  int       `json:"cacheReadTokens"`
	CostUSD    float64   `json:"costUsd"`
	DurationMS int64     `json:"durationMs"`
	Compacts   int       `json:"compactions"`
	Repeat     int       `json:"repeat,omitempty"`
	Failure    string    `json:"failure,omitempty"`
	StartedAt  time.Time `json:"startedAt"`
}

// CacheHitRate is the share of prompt tokens served from cache.
func (r Result) CacheHitRate() float64 {
	if r.Input == 0 {
		return 0
	}
	return float64(r.CacheRead) / float64(r.Input)
}

// Run is a whole benchmark execution.
type Run struct {
	Label   string    `json:"label"`
	Model   string    `json:"model"`
	Started time.Time `json:"started"`
	Repeat  int       `json:"repeat,omitempty"`
	Results []Result  `json:"results"`
}

// Summary aggregates a run.
type Summary struct {
	Tasks     int
	Passed    int
	Input     int
	Output    int
	CacheRead int
	CostUSD   float64
	DurationS float64
	// MaxSpread is the widest relative range of input tokens seen across the
	// repeats of any single task. It is the honest measure of how much of a
	// difference between two runs is just noise.
	MaxSpread float64
}

// Summarize aggregates a run, taking the MEDIAN across repeats of each task.
//
// Median rather than mean because the outlier is the whole problem: a single
// run that wandered for ten extra turns should not decide whether a change
// ships. A task counts as passed only if EVERY repeat passed — flaky is not
// passing.
func (r Run) Summarize() Summary {
	order, byTask := r.ByTask()
	s := Summary{Tasks: len(order)}
	for _, name := range order {
		results := byTask[name]

		allPassed := true
		var in, out, cache, cost, dur []float64
		for _, res := range results {
			if !res.Passed {
				allPassed = false
			}
			in = append(in, float64(res.Input))
			out = append(out, float64(res.Output))
			cache = append(cache, float64(res.CacheRead))
			cost = append(cost, res.CostUSD)
			dur = append(dur, float64(res.DurationMS)/1000)
		}
		if allPassed {
			s.Passed++
		}
		s.Input += int(median(in))
		s.Output += int(median(out))
		s.CacheRead += int(median(cache))
		s.CostUSD += median(cost)
		s.DurationS += median(dur)
		s.MaxSpread = maxf(s.MaxSpread, Spread(in))
	}
	return s
}

// CacheHitRate is the share of prompt tokens served from cache across the run.
func (s Summary) CacheHitRate() float64 {
	if s.Input == 0 {
		return 0
	}
	return float64(s.CacheRead) / float64(s.Input)
}

// LoadTasks reads every task directory under root.
//
// Ordering is alphabetical so two runs execute the same tasks in the same
// sequence. Comparing runs that ran in different orders is comparing two
// different experiments.
func LoadTasks(root string) ([]Task, error) {
	// Absolute, because a task's commands run in a temp working copy elsewhere.
	// A relative EVAL_TASK_DIR resolves against the wrong directory and every
	// verifier silently fails to find its own files.
	root, err := filepath.Abs(root)
	if err != nil {
		return nil, fmt.Errorf("resolving task directory: %w", err)
	}
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil, fmt.Errorf("reading task directory: %w", err)
	}
	var tasks []Task
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		dir := filepath.Join(root, e.Name())
		data, err := os.ReadFile(filepath.Join(dir, "task.json"))
		if err != nil {
			continue // a directory without task.json is not a task
		}
		var t Task
		if err := json.Unmarshal(data, &t); err != nil {
			return nil, fmt.Errorf("%s/task.json: %w", e.Name(), err)
		}
		if t.Name == "" {
			t.Name = e.Name()
		}
		if t.Prompt == "" || t.Verify == "" {
			return nil, fmt.Errorf("%s: task needs both a prompt and a verify command", t.Name)
		}
		t.Dir = dir
		tasks = append(tasks, t)
	}
	sort.Slice(tasks, func(i, j int) bool { return tasks[i].Name < tasks[j].Name })
	return tasks, nil
}

// Options configure a run.
type Options struct {
	Model  string
	Effort ai.Effort
	// Repeat runs each task this many times. One is not enough to compare
	// efficiency: measured run-to-run spread on an unchanged agent reaches 40%
	// on token counts, because the model simply words things differently.
	Repeat int
	// Verbose streams agent output instead of capturing it.
	Verbose bool
	// OnProgress reports each finished task.
	OnProgress func(Result)
}

// RunTasks executes every task and returns the run record.
func RunTasks(ctx context.Context, tasks []Task, opts Options) Run {
	model := opts.Model
	if model == "" {
		model = ai.ModelFlash
	}
	repeat := opts.Repeat
	if repeat < 1 {
		repeat = 1
	}
	run := Run{Model: model, Started: time.Now(), Repeat: repeat}
	for _, t := range tasks {
		for i := range repeat {
			if ctx.Err() != nil {
				return run
			}
			res := RunTask(ctx, t, opts)
			res.Repeat = i
			run.Results = append(run.Results, res)
			if opts.OnProgress != nil {
				opts.OnProgress(res)
			}
		}
	}
	return run
}

// ByTask groups results, in task order, so repeats of one task stay together.
func (r Run) ByTask() ([]string, map[string][]Result) {
	var order []string
	byTask := map[string][]Result{}
	for _, res := range r.Results {
		if _, seen := byTask[res.Task]; !seen {
			order = append(order, res.Task)
		}
		byTask[res.Task] = append(byTask[res.Task], res)
	}
	return order, byTask
}

// median returns the middle value, which is what repeats are for: one unlucky
// run that wandered for ten extra turns should not move the number a gate
// decides on.
func median(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	mid := len(s) / 2
	if len(s)%2 == 1 {
		return s[mid]
	}
	return (s[mid-1] + s[mid]) / 2
}

// Spread is the relative range of a metric across repeats, as a fraction of the
// median. It is what tells you whether a gate tolerance is defensible or
// wishful.
func Spread(xs []float64) float64 {
	if len(xs) < 2 {
		return 0
	}
	m := median(xs)
	if m == 0 {
		return 0
	}
	lo, hi := xs[0], xs[0]
	for _, x := range xs {
		lo, hi = minf(lo, x), maxf(hi, x)
	}
	return (hi - lo) / m
}

func minf(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

func maxf(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

// RunTask executes one task in a disposable copy of its workspace.
//
// The copy matters: a task that mutates its own fixture is a task that passes
// once and then measures something different forever.
func RunTask(ctx context.Context, t Task, opts Options) Result {
	started := time.Now()
	res := Result{
		Task: t.Name, Model: opts.Model, Effort: string(opts.Effort),
		StartedAt: started,
	}
	if res.Model == "" {
		res.Model = ai.ModelFlash
	}
	if t.Effort != "" {
		res.Effort = t.Effort
	}

	workdir, cleanup, err := prepareWorkspace(t)
	if err != nil {
		res.Failure = err.Error()
		res.DurationMS = time.Since(started).Milliseconds()
		return res
	}
	defer cleanup()

	timeout := DefaultTimeout
	if t.TimeoutSeconds > 0 {
		timeout = time.Duration(t.TimeoutSeconds) * time.Second
	}
	runCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	effort := opts.Effort
	if t.Effort != "" {
		effort = ai.Effort(t.Effort)
	}

	h, err := harness.New(runCtx, harness.Options{
		Cwd:   workdir,
		Model: res.Model,
		// Yolo: an eval has no human to prompt, and the workspace is a
		// throwaway copy. Anything else would measure the permission gate
		// rather than the agent.
		Mode:   harness.ModeYolo,
		Effort: effort,
		// Skills would vary by whoever runs the eval, which makes results
		// incomparable across machines.
		NoSkills: true,
	})
	if err != nil {
		res.Failure = "harness: " + err.Error()
		res.DurationMS = time.Since(started).Milliseconds()
		return res
	}
	defer func() { _ = h.Close() }()

	h.Agent.Subscribe(func(ev agent.Event) {
		switch ev.Type {
		case agent.EventTurnEnd:
			res.Turns++
		case agent.EventToolEnd:
			res.ToolCalls++
			if ev.IsError {
				res.ToolErrors++
			}
		}
	})
	prevCompact := h.Compactor.OnEvent
	h.Compactor.OnEvent = func(ev harness.CompactionEvent) {
		if prevCompact != nil {
			prevCompact(ev)
		}
		res.Compacts++
	}

	if _, err := h.Agent.Prompt(runCtx, t.Prompt); err != nil {
		res.Failure = "agent: " + err.Error()
	}

	u := h.Session.Usage
	res.Input, res.Output, res.CacheRead = u.Input, u.Output, u.CacheRead
	res.CostUSD = u.Cost.Total

	// Verify regardless of how the agent finished. An agent that errored may
	// still have completed the task, and an agent that reported success may
	// not have; only the verifier decides.
	if out, err := runShell(context.WithoutCancel(runCtx), t.Verify, workdir, "EVAL_TASK_DIR="+t.Dir); err != nil {
		res.Passed = false
		if res.Failure == "" {
			res.Failure = strings.TrimSpace(lastLines(out, 5))
		}
	} else {
		res.Passed = true
		res.Failure = ""
	}

	res.DurationMS = time.Since(started).Milliseconds()
	return res
}

// prepareWorkspace copies a task's fixture into a temp directory.
func prepareWorkspace(t Task) (string, func(), error) {
	dir, err := os.MkdirTemp("", "dspi-eval-")
	if err != nil {
		return "", func() {}, fmt.Errorf("creating work directory: %w", err)
	}
	cleanup := func() { _ = os.RemoveAll(dir) }

	src := filepath.Join(t.Dir, "workspace")
	if _, err := os.Stat(src); err == nil {
		if err := copyTree(src, dir); err != nil {
			cleanup()
			return "", func() {}, fmt.Errorf("copying workspace: %w", err)
		}
	}
	if t.Setup != "" {
		if out, err := runShell(context.Background(), t.Setup, dir, "EVAL_TASK_DIR="+t.Dir); err != nil {
			cleanup()
			return "", func() {}, fmt.Errorf("setup failed: %v\n%s", err, out)
		}
	}
	return dir, cleanup, nil
}

func copyTree(src, dst string) error {
	return filepath.Walk(src, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(src, path)
		if err != nil {
			return err
		}
		target := filepath.Join(dst, rel)
		if info.IsDir() {
			return os.MkdirAll(target, 0o755)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		return os.WriteFile(target, data, info.Mode().Perm())
	})
}

// runShell runs a task command.
//
// EVAL_TASK_DIR is exported so a verifier can restore its own canonical files
// before checking. Without that, an agent can pass a "make the tests pass" task
// by deleting the test, which measures nothing.
func runShell(ctx context.Context, command, dir string, env ...string) (string, error) {
	cmd := exec.CommandContext(ctx, "bash", "-c", command)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), env...)
	out, err := cmd.CombinedOutput()
	return string(out), err
}

func lastLines(s string, n int) string {
	lines := strings.Split(strings.TrimRight(s, "\n"), "\n")
	if len(lines) > n {
		lines = lines[len(lines)-n:]
	}
	return strings.Join(lines, "\n")
}

// Save writes a run to disk as JSON.
func Save(path string, run Run) error {
	data, err := json.MarshalIndent(run, "", "  ")
	if err != nil {
		return err
	}
	if dir := filepath.Dir(path); dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	return os.WriteFile(path, append(data, '\n'), 0o644)
}

// Load reads a run from disk.
func Load(path string) (Run, error) {
	var run Run
	data, err := os.ReadFile(path)
	if err != nil {
		return run, err
	}
	return run, json.Unmarshal(data, &run)
}
