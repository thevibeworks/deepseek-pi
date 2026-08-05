// Command dspi-eval measures deepseek-pi on a suite of real tasks.
//
// Two uses. Run the suite and record it as a baseline; later, run it again and
// compare. The comparison is what gates an engine change: the project's rule is
// that token usage, cost and success rate must improve or hold, and this is
// what turns that rule into a number instead of an argument.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/thevibeworks/deepseek-pi/ai"
	"github.com/thevibeworks/deepseek-pi/eval"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "dspi-eval: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	var (
		tasksDir = flag.String("tasks", "eval/testdata/tasks", "directory of task definitions")
		out      = flag.String("out", "", "write the run to this JSON file")
		baseline = flag.String("baseline", "", "compare against this run and gate on it")
		model    = flag.String("m", ai.ModelFlash, "model to evaluate")
		effort   = flag.String("e", "", "reasoning effort")
		only     = flag.String("only", "", "run just this task")
		repeat   = flag.Int("repeat", 1, "run each task N times and compare medians")
		compare  = flag.String("compare", "", "compare two saved runs: -compare old.json -baseline is unused")
	)
	flag.Parse()

	// Pure comparison of two recorded runs, no API calls.
	if *compare != "" {
		if *baseline == "" {
			return fmt.Errorf("-compare needs -baseline to compare against")
		}
		base, err := eval.Load(*baseline)
		if err != nil {
			return fmt.Errorf("loading baseline: %w", err)
		}
		cur, err := eval.Load(*compare)
		if err != nil {
			return fmt.Errorf("loading run: %w", err)
		}
		comparisons, ok := eval.Compare(base, cur)
		fmt.Print(eval.FormatComparisonWithTolerance(comparisons, ok, eval.EffectiveTolerance(base)))
		if !ok {
			os.Exit(1)
		}
		return nil
	}

	tasks, err := eval.LoadTasks(*tasksDir)
	if err != nil {
		return err
	}
	if *only != "" {
		var filtered []eval.Task
		for _, t := range tasks {
			if t.Name == *only {
				filtered = append(filtered, t)
			}
		}
		if len(filtered) == 0 {
			return fmt.Errorf("no task named %q in %s", *only, *tasksDir)
		}
		tasks = filtered
	}
	if len(tasks) == 0 {
		return fmt.Errorf("no tasks found in %s", *tasksDir)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	fmt.Fprintf(os.Stderr, "running %d task(s) on %s\n\n", len(tasks), *model)
	run := eval.RunTasks(ctx, tasks, eval.Options{
		Model:  *model,
		Effort: ai.Effort(*effort),
		Repeat: *repeat,
		OnProgress: func(r eval.Result) {
			status := "pass"
			if !r.Passed {
				status = "FAIL"
			}
			fmt.Fprintf(os.Stderr, "  %-4s %-24s %d turns, %d tools, $%.4f, %.0fs\n",
				status, r.Task, r.Turns, r.ToolCalls, r.CostUSD, float64(r.DurationMS)/1000)
		},
	})

	fmt.Println()
	fmt.Print(eval.FormatRun(run))

	if *out != "" {
		if err := eval.Save(*out, run); err != nil {
			return fmt.Errorf("saving run: %w", err)
		}
		fmt.Fprintf(os.Stderr, "\nwrote %s\n", *out)
	}

	if *baseline != "" {
		base, err := eval.Load(*baseline)
		if err != nil {
			return fmt.Errorf("loading baseline: %w", err)
		}
		comparisons, ok := eval.Compare(base, run)
		fmt.Println()
		fmt.Print(eval.FormatComparisonWithTolerance(comparisons, ok, eval.EffectiveTolerance(base)))
		if !ok {
			os.Exit(1)
		}
	}

	// A failing task is a failing run, so CI notices without needing a baseline.
	if s := run.Summarize(); s.Passed < s.Tasks {
		os.Exit(1)
	}
	return nil
}
