package eval

import (
	"os"
	"path/filepath"
	"testing"
)

func result(task string, passed bool, in, out int, cost float64) Result {
	return Result{Task: task, Passed: passed, Input: in, Output: out, CostUSD: cost}
}

func TestSummarizeUsesMediansAcrossRepeats(t *testing.T) {
	// The middle value, not the mean: one run that wandered for ten extra turns
	// should not decide whether a change ships.
	run := Run{Repeat: 3, Results: []Result{
		result("a", true, 100, 10, 0.01),
		result("a", true, 900, 90, 0.09), // the outlier
		result("a", true, 200, 20, 0.02),
	}}
	s := run.Summarize()
	if s.Input != 200 || s.Output != 20 {
		t.Errorf("got in=%d out=%d, want the medians 200/20", s.Input, s.Output)
	}
	if s.Tasks != 1 {
		t.Errorf("Tasks = %d, want 1: repeats of one task are one task", s.Tasks)
	}
}

func TestSummarizeCountsFlakyAsFailed(t *testing.T) {
	// Passing two times in three is not passing. A gate that accepts flaky
	// results cannot tell a real regression from an unlucky run.
	run := Run{Repeat: 3, Results: []Result{
		result("a", true, 100, 10, 0.01),
		result("a", false, 100, 10, 0.01),
		result("a", true, 100, 10, 0.01),
	}}
	if got := run.Summarize().Passed; got != 0 {
		t.Errorf("Passed = %d, want 0: one failing repeat fails the task", got)
	}
}

func TestSpreadMeasuresNoise(t *testing.T) {
	if got := Spread([]float64{100, 100, 100}); got != 0 {
		t.Errorf("Spread of identical values = %v, want 0", got)
	}
	// range 50 over median 100
	if got := Spread([]float64{75, 100, 125}); got < 0.49 || got > 0.51 {
		t.Errorf("Spread = %v, want ~0.5", got)
	}
	if got := Spread([]float64{42}); got != 0 {
		t.Errorf("a single sample has no measurable spread, got %v", got)
	}
}

func TestEffectiveToleranceNeverBelowFloor(t *testing.T) {
	quiet := Run{Repeat: 3, Results: []Result{
		result("a", true, 100, 10, 0.01),
		result("a", true, 100, 10, 0.01),
		result("a", true, 100, 10, 0.01),
	}}
	if got := EffectiveTolerance(quiet); got != MinTolerance {
		t.Errorf("EffectiveTolerance = %v, want the floor %v", got, MinTolerance)
	}

	// A noisy suite raises its own bar rather than pretending to be precise.
	noisy := Run{Repeat: 3, Results: []Result{
		result("a", true, 50, 10, 0.01),
		result("a", true, 100, 10, 0.01),
		result("a", true, 150, 10, 0.01),
	}}
	if got := EffectiveTolerance(noisy); got <= MinTolerance {
		t.Errorf("EffectiveTolerance = %v, want above the floor for a noisy suite", got)
	}
}

func TestCompareIgnoresMovementInsideTheNoise(t *testing.T) {
	// The failure this prevents: a gate that fires on ordinary model variance
	// gets ignored, which is worse than having no gate.
	base := Run{Results: []Result{result("a", true, 1000, 100, 0.01)}}
	within := Run{Results: []Result{result("a", true, 1150, 110, 0.011)}} // +15%

	comparisons, ok := Compare(base, within)
	if !ok {
		t.Errorf("gate fired on a %v%% move, inside the measured noise floor: %+v",
			15, comparisons)
	}
}

func TestCompareCatchesRealRegression(t *testing.T) {
	base := Run{Results: []Result{result("a", true, 1000, 100, 0.01)}}
	worse := Run{Results: []Result{result("a", true, 2000, 100, 0.02)}} // +100%

	comparisons, ok := Compare(base, worse)
	if ok {
		t.Error("gate passed a doubling of input tokens")
	}
	var flagged bool
	for _, c := range comparisons {
		if c.Metric == "input tokens" && c.Regressed {
			flagged = true
		}
	}
	if !flagged {
		t.Errorf("input tokens not flagged: %+v", comparisons)
	}
}

func TestCompareHasNoToleranceForCorrectness(t *testing.T) {
	// Tokens improved, but a task that passed now fails. Correctness is not
	// tradeable against efficiency.
	base := Run{Results: []Result{
		result("a", true, 1000, 100, 0.01),
		result("b", true, 1000, 100, 0.01),
	}}
	current := Run{Results: []Result{
		result("a", true, 100, 10, 0.001),
		result("b", false, 100, 10, 0.001),
	}}

	comparisons, ok := Compare(base, current)
	if ok {
		t.Fatal("gate passed a run where a previously passing task now fails")
	}
	var named bool
	for _, c := range comparisons {
		if c.Metric == "task b" && c.Regressed {
			named = true
		}
	}
	if !named {
		t.Errorf("the specific regressed task was not named: %+v", comparisons)
	}
}

func TestLoadTasksValidatesAndSorts(t *testing.T) {
	root := t.TempDir()
	write := func(name, body string) {
		dir := filepath.Join(root, name)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "task.json"), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("zebra", `{"prompt":"p","verify":"true"}`)
	write("alpha", `{"prompt":"p","verify":"true"}`)
	// A directory with no task.json is not a task, and must not be an error.
	if err := os.MkdirAll(filepath.Join(root, "notatask"), 0o755); err != nil {
		t.Fatal(err)
	}

	tasks, err := LoadTasks(root)
	if err != nil {
		t.Fatalf("LoadTasks: %v", err)
	}
	if len(tasks) != 2 {
		t.Fatalf("got %d tasks, want 2", len(tasks))
	}
	// Stable order: comparing runs that executed in different orders compares
	// two different experiments.
	if tasks[0].Name != "alpha" || tasks[1].Name != "zebra" {
		t.Errorf("tasks not sorted: %s, %s", tasks[0].Name, tasks[1].Name)
	}
	// Absolute, because verifiers run in a temp working copy elsewhere.
	if !filepath.IsAbs(tasks[0].Dir) {
		t.Errorf("task dir %q is not absolute; EVAL_TASK_DIR would not resolve", tasks[0].Dir)
	}
}

func TestLoadTasksRejectsIncompleteTask(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, "broken")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	// No verify command: the task could never be judged, so loading it would
	// silently add a case that always passes.
	if err := os.WriteFile(filepath.Join(dir, "task.json"), []byte(`{"prompt":"p"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadTasks(root); err == nil {
		t.Error("a task with no verify command was accepted")
	}
}

func TestBundledBaselineIsLoadable(t *testing.T) {
	// The committed baseline is the reference every gate compares against. If
	// it stops parsing, the gate silently stops working.
	run, err := Load("baseline-flash.json")
	if err != nil {
		t.Skipf("no committed baseline yet: %v", err)
	}
	s := run.Summarize()
	if s.Tasks == 0 {
		t.Fatal("baseline contains no tasks")
	}
	if s.Passed != s.Tasks {
		t.Errorf("committed baseline has failing tasks (%d/%d); a baseline should be a known-good run",
			s.Passed, s.Tasks)
	}
	if run.Repeat < 2 {
		t.Errorf("baseline has %d repeat(s); a single sample is too noisy to gate against", run.Repeat)
	}
}
