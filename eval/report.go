package eval

import (
	"fmt"
	"strings"
)

// FormatRun renders a run as a table.
func FormatRun(run Run) string {
	var b strings.Builder
	fmt.Fprintf(&b, "%-24s %-6s %5s %6s %9s %8s %7s %9s %8s\n",
		"TASK", "RESULT", "TURNS", "TOOLS", "IN", "OUT", "CACHE", "COST", "TIME")
	for _, r := range run.Results {
		status := "pass"
		if !r.Passed {
			status = "FAIL"
		}
		fmt.Fprintf(&b, "%-24s %-6s %5d %6d %9d %8d %6.0f%% %9.4f %7.1fs\n",
			truncate(r.Task, 24), status, r.Turns, r.ToolCalls,
			r.Input, r.Output, r.CacheHitRate()*100, r.CostUSD,
			float64(r.DurationMS)/1000)
		if r.Failure != "" {
			for _, line := range strings.Split(r.Failure, "\n") {
				fmt.Fprintf(&b, "    %s\n", truncate(line, 100))
			}
		}
	}

	s := run.Summarize()
	fmt.Fprintf(&b, "\n%d/%d passed · %d in / %d out · cache %.0f%% · $%.4f · %.0fs\n",
		s.Passed, s.Tasks, s.Input, s.Output, s.CacheHitRate()*100, s.CostUSD, s.DurationS)
	if run.Repeat > 1 {
		fmt.Fprintf(&b, "medians over %d repeats; widest input-token spread within a task: %.0f%%\n",
			run.Repeat, s.MaxSpread*100)
	} else {
		fmt.Fprintf(&b, "single sample per task — too noisy to gate efficiency; use -repeat 3\n")
	}
	return b.String()
}

// Comparison is one metric's movement between two runs.
type Comparison struct {
	Metric   string
	Baseline float64
	Current  float64
	// Regressed marks a movement in the wrong direction beyond the tolerance.
	Regressed bool
	// LowerIsBetter records which direction counts as an improvement.
	LowerIsBetter bool
}

// Delta is the change, positive meaning the metric went up.
func (c Comparison) Delta() float64 { return c.Current - c.Baseline }

// PercentChange is the movement relative to the baseline.
func (c Comparison) PercentChange() float64 {
	if c.Baseline == 0 {
		return 0
	}
	return (c.Current - c.Baseline) / c.Baseline * 100
}

// MinTolerance is the floor on how much a token or cost metric may move before
// it counts as a regression.
//
// This number was measured, not chosen. Two consecutive runs of an unchanged
// agent over the same three tasks differed by 24% on input tokens and 42% on
// output tokens, purely because the model worded things differently and took a
// different number of turns. A 10% gate — the obvious first guess, and the one
// this file originally shipped — fails on pure noise, and a gate that cries
// wolf is worse than no gate because people learn to ignore it.
//
// The effective tolerance is the LARGER of this floor and the spread actually
// observed within the baseline's own repeats, so the gate calibrates itself to
// the noise of the suite it is gating rather than to an assumption about it.
//
// Correctness has no tolerance. A task that passed and now fails is always a
// regression, however the tokens moved.
const MinTolerance = 0.25

// EffectiveTolerance is the noise floor a change must clear to count.
func EffectiveTolerance(baseline Run) float64 {
	measured := baseline.Summarize().MaxSpread
	if measured > MinTolerance {
		return measured
	}
	return MinTolerance
}

// Compare measures a run against a baseline.
//
// The rule the project set for itself: token usage, cost and success rate must
// improve or hold. This returns the evidence and the verdict; it does not
// soften either.
func Compare(baseline, current Run) ([]Comparison, bool) {
	b, c := baseline.Summarize(), current.Summarize()
	tol := EffectiveTolerance(baseline)

	comparisons := []Comparison{
		{
			Metric: "tasks passed", Baseline: float64(b.Passed), Current: float64(c.Passed),
			LowerIsBetter: false, Regressed: c.Passed < b.Passed,
		},
		{
			Metric: "input tokens", Baseline: float64(b.Input), Current: float64(c.Input),
			LowerIsBetter: true, Regressed: regressedUp(float64(b.Input), float64(c.Input), tol),
		},
		{
			Metric: "output tokens", Baseline: float64(b.Output), Current: float64(c.Output),
			LowerIsBetter: true, Regressed: regressedUp(float64(b.Output), float64(c.Output), tol),
		},
		{
			Metric: "cost usd", Baseline: b.CostUSD, Current: c.CostUSD,
			LowerIsBetter: true, Regressed: regressedUp(b.CostUSD, c.CostUSD, tol),
		},
		{
			Metric: "cache hit rate", Baseline: b.CacheHitRate(), Current: c.CacheHitRate(),
			LowerIsBetter: false, Regressed: c.CacheHitRate() < b.CacheHitRate()-0.05,
		},
	}

	ok := true
	for _, cmp := range comparisons {
		if cmp.Regressed {
			ok = false
		}
	}
	// A task that flipped from pass to fail is a regression even when the
	// totals happen to match, so check per task as well.
	for _, name := range regressedTasks(baseline, current) {
		comparisons = append(comparisons, Comparison{
			Metric: "task " + name, Baseline: 1, Current: 0, Regressed: true,
		})
		ok = false
	}
	return comparisons, ok
}

func regressedUp(baseline, current, tolerance float64) bool {
	if baseline == 0 {
		return current > 0
	}
	return current > baseline*(1+tolerance)
}

// regressedTasks lists tasks that passed in the baseline and fail now.
func regressedTasks(baseline, current Run) []string {
	was := map[string]bool{}
	for _, r := range baseline.Results {
		was[r.Task] = r.Passed
	}
	var out []string
	for _, r := range current.Results {
		if was[r.Task] && !r.Passed {
			out = append(out, r.Task)
		}
	}
	return out
}

// FormatComparison renders a comparison table and the verdict.
func FormatComparison(comparisons []Comparison, ok bool) string {
	return FormatComparisonWithTolerance(comparisons, ok, MinTolerance)
}

// FormatComparisonWithTolerance renders the table and says which tolerance was
// applied, so a reader can tell a real win from a change inside the noise.
func FormatComparisonWithTolerance(comparisons []Comparison, ok bool, tolerance float64) string {
	var b strings.Builder
	fmt.Fprintf(&b, "%-20s %12s %12s %10s\n", "METRIC", "BASELINE", "CURRENT", "CHANGE")
	for _, c := range comparisons {
		marker := ""
		if c.Regressed {
			marker = "  <- regression"
		}
		fmt.Fprintf(&b, "%-20s %12.4f %12.4f %9.1f%%%s\n",
			c.Metric, c.Baseline, c.Current, c.PercentChange(), marker)
	}
	if ok {
		fmt.Fprintf(&b, "\nGATE PASSED (tolerance %.0f%% on tokens and cost; none on correctness)\n",
			tolerance*100)
	} else {
		fmt.Fprintf(&b, "\nGATE FAILED\n")
	}
	return b.String()
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	if n <= 1 {
		return s[:n]
	}
	return s[:n-1] + "…"
}
