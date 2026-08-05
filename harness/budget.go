package harness

import (
	"errors"
	"fmt"
	"sync"

	"github.com/thevibeworks/deepseek-pi/agent"
)

// ErrBudgetExceeded marks a run that a limit stopped rather than the model
// finishing. Wrapped by BudgetStop so a caller can tell "we ran out of budget"
// apart from "the request failed" without string matching.
var ErrBudgetExceeded = errors.New("budget exceeded")

// SessionBudget bounds the session itself, where Budget bounds one sub-agent.
//
// The two axes have deliberately DIFFERENT scopes, because the failures they
// catch are different:
//
//   - MaxCost is cumulative over the run, sub-agents included. Money only
//     accumulates, and the question it answers is "how much am I willing to
//     spend today".
//   - MaxTurns bounds ONE request. A session that has answered forty questions
//     is working; one request that took forty turns is looping. Making this
//     cumulative would stop a productive session at an arbitrary point that
//     MaxCost already covers, and would say nothing about looping.
//
// Zero on either axis means unbounded, which is the default: a budget nobody
// asked for that silently truncates a long task is worse than no budget.
type SessionBudget struct {
	// MaxCost is the ceiling in USD for the whole run.
	MaxCost float64
	// MaxTurns is the ceiling on assistant turns within a single request.
	MaxTurns int
}

// Empty reports whether this budget bounds nothing.
func (b SessionBudget) Empty() bool { return b.MaxCost <= 0 && b.MaxTurns <= 0 }

// BudgetStop says which limit fired and where the run stood when it did. It is
// an error so it cannot be returned and ignored, and carries the numbers so the
// message can be specific instead of "budget exceeded".
type BudgetStop struct {
	// Axis is "cost" or "turns".
	Axis   string
	Spent  float64
	Turns  int
	Budget SessionBudget
}

func (b BudgetStop) Error() string {
	if b.Axis == "turns" {
		return fmt.Sprintf(
			"stopped by the turn budget: %s in one request, limit %d "+
				"(raise it with /budget turns N, or -max-turns)",
			plural(b.Turns, "turn"), b.Budget.MaxTurns)
	}
	return fmt.Sprintf(
		"stopped by the cost budget: spent %s of a %s limit "+
			"(raise it with /budget cost N, or -max-cost)",
		Money(b.Spent), Money(b.Budget.MaxCost))
}

// Money renders a USD figure at the session's usual four decimals, but keeps
// enough precision that a real amount never prints as $0.0000 — a limit that
// displays as zero reads as a bug in the tool rather than a small number.
func Money(v float64) string {
	if v > 0 && v < 0.0001 {
		return fmt.Sprintf("$%.6f", v)
	}
	return fmt.Sprintf("$%.4f", v)
}

// Unwrap lets errors.Is(err, ErrBudgetExceeded) work on any stop.
func (b BudgetStop) Unwrap() error { return ErrBudgetExceeded }

func plural(n int, noun string) string {
	if n == 1 {
		return fmt.Sprintf("%d %s", n, noun)
	}
	return fmt.Sprintf("%d %ss", n, noun)
}

// BudgetGuard enforces a SessionBudget at the between-turns seam.
//
// Between turns is the only place a run can be cut without corrupting it: mid
// turn the transcript holds an assistant message whose tool calls are still
// unanswered, and a provider rejects that outright. Stopping here costs at most
// one turn of overshoot, which is the price of never producing a broken
// transcript.
type BudgetGuard struct {
	mu     sync.Mutex
	budget SessionBudget
	turns  int
	stop   *BudgetStop
	// spent reports the run's total cost. A function rather than a number so
	// the guard reads the same figure the UI prints, instead of keeping a
	// second tally that can drift from it.
	spent func() float64
}

// NewBudgetGuard builds a guard over a cost source.
func NewBudgetGuard(b SessionBudget, spent func() float64) *BudgetGuard {
	return &BudgetGuard{budget: b, spent: spent}
}

// Limits returns the budget in force.
func (g *BudgetGuard) Limits() SessionBudget {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.budget
}

// SetLimits replaces the budget and clears any recorded stop.
//
// Clearing is the point: raising a limit is how a user says "keep going", and a
// stop that survived the raise would refuse the very next request.
func (g *BudgetGuard) SetLimits(b SessionBudget) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.budget = b
	g.stop = nil
}

// Spent reports what the run has cost so far, in USD.
func (g *BudgetGuard) Spent() float64 {
	if g.spent == nil {
		return 0
	}
	return g.spent()
}

// Stop returns the limit that ended the last request, or nil.
func (g *BudgetGuard) Stop() *BudgetStop {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.stop
}

// Turns reports how many turns the request currently running has taken.
func (g *BudgetGuard) Turns() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.turns
}

// begin opens a request. It resets the per-request turn count and refuses to
// start at all when the cost ceiling is already behind us.
//
// Refusing up front matters: the seam below can only stop a request AFTER a
// turn, so without this check every prompt sent past the limit would buy one
// more turn to be told it was over budget.
func (g *BudgetGuard) begin() error {
	spent := g.Spent()

	g.mu.Lock()
	defer g.mu.Unlock()
	g.turns = 0
	g.stop = nil
	if g.budget.MaxCost > 0 && spent >= g.budget.MaxCost {
		stop := BudgetStop{Axis: "cost", Spent: spent, Budget: g.budget}
		g.stop = &stop
		return stop
	}
	return nil
}

// afterTurn is the ShouldStopAfterTurn hook.
func (g *BudgetGuard) afterTurn(agent.TurnContext) bool {
	// Read cost outside the lock: spent() reaches back into the harness, and
	// holding the guard's lock across it would invite a cycle.
	spent := g.Spent()

	g.mu.Lock()
	defer g.mu.Unlock()
	g.turns++

	switch {
	case g.budget.MaxTurns > 0 && g.turns >= g.budget.MaxTurns:
		g.stop = &BudgetStop{Axis: "turns", Spent: spent, Turns: g.turns, Budget: g.budget}
	case g.budget.MaxCost > 0 && spent >= g.budget.MaxCost:
		g.stop = &BudgetStop{Axis: "cost", Spent: spent, Turns: g.turns, Budget: g.budget}
	default:
		return false
	}
	return true
}
