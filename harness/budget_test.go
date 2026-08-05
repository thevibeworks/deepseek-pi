package harness

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/thevibeworks/deepseek-pi/agent"
	"github.com/thevibeworks/deepseek-pi/ai"
)

func guard(b SessionBudget, spent *float64) *BudgetGuard {
	return NewBudgetGuard(b, func() float64 { return *spent })
}

func TestNoBudgetNeverStops(t *testing.T) {
	spent := 1_000.0
	g := guard(SessionBudget{}, &spent)

	if err := g.begin(); err != nil {
		t.Fatalf("an unbounded run refused to start: %v", err)
	}
	for i := 0; i < 100; i++ {
		if g.afterTurn(agent.TurnContext{}) {
			t.Fatalf("stopped at turn %d with no budget set", i+1)
		}
	}
}

func TestTurnBudgetStopsOneRequest(t *testing.T) {
	spent := 0.0
	g := guard(SessionBudget{MaxTurns: 3}, &spent)

	if err := g.begin(); err != nil {
		t.Fatal(err)
	}
	for i := 1; i <= 2; i++ {
		if g.afterTurn(agent.TurnContext{}) {
			t.Fatalf("stopped early, at turn %d of 3", i)
		}
	}
	if !g.afterTurn(agent.TurnContext{}) {
		t.Fatal("the third turn did not trip a 3-turn budget")
	}

	stop := g.Stop()
	if stop == nil || stop.Axis != "turns" {
		t.Fatalf("stop = %+v, want the turns axis", stop)
	}
	if stop.Turns != 3 {
		t.Errorf("stop recorded %d turns, want 3", stop.Turns)
	}
}

// TestTurnBudgetIsPerRequestNotCumulative pins the scope asymmetry. A session
// that answers many questions is working; only one request that will not end is
// looping.
func TestTurnBudgetIsPerRequestNotCumulative(t *testing.T) {
	spent := 0.0
	g := guard(SessionBudget{MaxTurns: 3}, &spent)

	for req := 1; req <= 3; req++ {
		if err := g.begin(); err != nil {
			t.Fatalf("request %d refused: %v", req, err)
		}
		for i := 1; i <= 2; i++ {
			if g.afterTurn(agent.TurnContext{}) {
				t.Fatalf("request %d stopped at turn %d; the count did not reset", req, i)
			}
		}
	}
}

func TestCostBudgetStopsTheRun(t *testing.T) {
	spent := 0.0
	g := guard(SessionBudget{MaxCost: 0.50}, &spent)

	if err := g.begin(); err != nil {
		t.Fatal(err)
	}
	spent = 0.49
	if g.afterTurn(agent.TurnContext{}) {
		t.Fatal("stopped under the limit")
	}
	spent = 0.51
	if !g.afterTurn(agent.TurnContext{}) {
		t.Fatal("did not stop over the limit")
	}

	stop := g.Stop()
	if stop == nil || stop.Axis != "cost" {
		t.Fatalf("stop = %+v, want the cost axis", stop)
	}
	if stop.Spent != 0.51 {
		t.Errorf("stop recorded $%.4f, want the real 0.51", stop.Spent)
	}
}

// TestAnOverspentRunRefusesToStart is the reason begin exists. The seam below
// can only stop AFTER a turn, so without this check every prompt past the limit
// would buy one more turn just to be told it was over budget.
func TestAnOverspentRunRefusesToStart(t *testing.T) {
	spent := 2.0
	g := guard(SessionBudget{MaxCost: 1.0}, &spent)

	err := g.begin()
	if err == nil {
		t.Fatal("a run already over its cost limit was allowed to start")
	}
	if !errors.Is(err, ErrBudgetExceeded) {
		t.Errorf("errors.Is could not identify the stop: %v", err)
	}
	var stop BudgetStop
	if !errors.As(err, &stop) || stop.Axis != "cost" {
		t.Errorf("errors.As did not recover the detail: %v", err)
	}
}

// TestRaisingTheLimitClearsTheStop is what makes a budget usable rather than a
// wall: raising it is how the user says "keep going".
func TestRaisingTheLimitClearsTheStop(t *testing.T) {
	spent := 2.0
	g := guard(SessionBudget{MaxCost: 1.0}, &spent)
	if err := g.begin(); err == nil {
		t.Fatal("expected the first begin to refuse")
	}

	g.SetLimits(SessionBudget{MaxCost: 5.0})
	if g.Stop() != nil {
		t.Error("the old stop survived the raise")
	}
	if err := g.begin(); err != nil {
		t.Fatalf("still refusing after the limit was raised: %v", err)
	}
}

func TestTurnsBudgetDoesNotWedgeTheNextRequest(t *testing.T) {
	spent := 0.0
	g := guard(SessionBudget{MaxTurns: 1}, &spent)

	if err := g.begin(); err != nil {
		t.Fatal(err)
	}
	if !g.afterTurn(agent.TurnContext{}) {
		t.Fatal("a 1-turn budget did not stop after one turn")
	}
	// The next request must be allowed: it looped, it did not overspend.
	if err := g.begin(); err != nil {
		t.Fatalf("the following request was refused: %v", err)
	}
	if g.Stop() != nil {
		t.Error("the previous request's stop leaked into this one")
	}
}

func TestBudgetStopSaysWhichLimitAndHowToRaiseIt(t *testing.T) {
	for _, tc := range []struct {
		stop BudgetStop
		want []string
	}{
		{
			BudgetStop{Axis: "cost", Spent: 1.25, Budget: SessionBudget{MaxCost: 1.0}},
			[]string{"cost", "1.2500", "1.0000", "/budget cost"},
		},
		{
			BudgetStop{Axis: "turns", Turns: 40, Budget: SessionBudget{MaxTurns: 40}},
			[]string{"turn", "40", "/budget turns"},
		},
	} {
		msg := tc.stop.Error()
		for _, want := range tc.want {
			if !strings.Contains(msg, want) {
				t.Errorf("%q is missing %q", msg, want)
			}
		}
	}
}

func TestSpentCountsSubAgents(t *testing.T) {
	h := testHarness(t)
	h.Session.Usage.Cost.Total = 0.10
	h.children = append(h.children,
		ChildReport{Usage: ai.Usage{Cost: ai.Cost{Total: 0.05}}},
		ChildReport{Usage: ai.Usage{Cost: ai.Cost{Total: 0.02}}},
	)

	// A loop that delegates is still a loop. A budget blind to children would
	// watch a session spend most of its money outside the number it checks.
	if got := h.Spent(); !near(got, 0.17) {
		t.Errorf("Spent() = %.4f, want 0.17 (parent 0.10 + children 0.07)", got)
	}
}

// ------------------------------------------------------------------- wiring

// providerBailout ends the fake loop if no budget does. Without it a broken
// hook would hang until the test binary's timeout instead of failing with a
// turn count that says exactly what went wrong.
const providerBailout = 25

// fakeProvider serves a stream that never finishes on its own: every turn asks
// to read the same file again. Only a budget can end it.
func fakeProvider(t *testing.T) (*httptest.Server, func() int) {
	t.Helper()
	var mu sync.Mutex
	turns := 0

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		turns++
		id := turns
		mu.Unlock()

		w.Header().Set("Content-Type", "text/event-stream")
		if id >= providerBailout {
			fmt.Fprint(w, sse("message_start", fmt.Sprintf(
				`{"type":"message_start","message":{"id":"msg_%d","model":"deepseek-v4-flash",`+
					`"usage":{"input_tokens":1000,"output_tokens":0}}}`, id)))
			fmt.Fprint(w, sse("content_block_start",
				`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`))
			fmt.Fprint(w, sse("content_block_delta",
				`{"type":"content_block_delta","index":0,`+
					`"delta":{"type":"text_delta","text":"nothing stopped me"}}`))
			fmt.Fprint(w, sse("content_block_stop", `{"type":"content_block_stop","index":0}`))
			fmt.Fprint(w, sse("message_delta",
				`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":100}}`))
			fmt.Fprint(w, sse("message_stop", `{"type":"message_stop"}`))
			return
		}
		fmt.Fprint(w, sse("message_start", fmt.Sprintf(
			`{"type":"message_start","message":{"id":"msg_%d","model":"deepseek-v4-flash",`+
				`"usage":{"input_tokens":1000,"output_tokens":0,"cache_read_input_tokens":0}}}`, id)))
		fmt.Fprint(w, sse("content_block_start",
			`{"type":"content_block_start","index":0,`+
				`"content_block":{"type":"tool_use","id":"call_1","name":"read"}}`))
		fmt.Fprint(w, sse("content_block_delta",
			`{"type":"content_block_delta","index":0,`+
				`"delta":{"type":"input_json_delta","partial_json":"{\"path\":\"loop.txt\"}"}}`))
		fmt.Fprint(w, sse("content_block_stop", `{"type":"content_block_stop","index":0}`))
		fmt.Fprint(w, sse("message_delta",
			`{"type":"message_delta","delta":{"stop_reason":"tool_use"},`+
				`"usage":{"output_tokens":100}}`))
		fmt.Fprint(w, sse("message_stop", `{"type":"message_stop"}`))
	}))
	t.Cleanup(srv.Close)

	return srv, func() int {
		mu.Lock()
		defer mu.Unlock()
		return turns
	}
}

func sse(event, data string) string {
	return "event: " + event + "\ndata: " + data + "\n\n"
}

// budgetHarness assembles a real harness pointed at a fake provider.
//
// It drives the assembled loop rather than the guard alone on purpose: the
// ShouldStopAfterTurn hook is a seam a unit test cannot see, and a budget that
// is never consulted fails silently and expensively.
func budgetHarness(t *testing.T, opts Options) (*Harness, func() int) {
	t.Helper()
	t.Setenv("HOME", t.TempDir())
	t.Setenv("DEEPSEEK_PI_HOME", t.TempDir())
	t.Setenv("DEEPSEEK_API_KEY", "test-key")

	cwd := t.TempDir()
	if err := os.WriteFile(filepath.Join(cwd, "loop.txt"), []byte("hello"), 0o644); err != nil {
		t.Fatal(err)
	}
	opts.Cwd = cwd
	opts.NoSkills = true
	opts.Mode = ModeYolo

	h, err := New(context.Background(), opts)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(func() { _ = h.Close() })

	srv, turns := fakeProvider(t)
	// The stream closure holds the client pointer, so redirecting it here
	// reaches the agent built above.
	h.Client.BaseURL = srv.URL
	return h, turns
}

func TestTurnBudgetStopsTheAssembledLoop(t *testing.T) {
	h, turns := budgetHarness(t, Options{MaxTurns: 3})

	_, err := h.Prompt(context.Background(), "loop forever")

	var stop BudgetStop
	if !errors.As(err, &stop) {
		t.Fatalf("the run did not report a budget stop: %v", err)
	}
	if stop.Axis != "turns" {
		t.Errorf("stopped on the %s axis, want turns", stop.Axis)
	}
	if n := turns(); n != 3 {
		t.Errorf("the provider was called %d times, want exactly 3", n)
	}
}

func TestCostBudgetStopsTheAssembledLoop(t *testing.T) {
	// Small enough that the first turn's real billed cost clears it, whatever
	// the rate card says.
	h, turns := budgetHarness(t, Options{MaxCost: 0.000001})

	_, err := h.Prompt(context.Background(), "loop forever")

	var stop BudgetStop
	if !errors.As(err, &stop) {
		t.Fatalf("the run did not report a budget stop: %v", err)
	}
	if stop.Axis != "cost" {
		t.Errorf("stopped on the %s axis, want cost", stop.Axis)
	}
	if n := turns(); n != 1 {
		t.Errorf("the provider was called %d times, want 1", n)
	}
	if stop.Spent <= 0 {
		t.Errorf("the stop reports $%.6f spent; cost accounting is not reaching the guard", stop.Spent)
	}

	// A second request must not buy another turn to rediscover the same limit.
	if _, err := h.Prompt(context.Background(), "again"); !errors.Is(err, ErrBudgetExceeded) {
		t.Fatalf("the follow-up request was not refused: %v", err)
	}
	if n := turns(); n != 1 {
		t.Errorf("the provider was called %d times after the refusal, want still 1", n)
	}
}

func TestAnUnboundedHarnessIsStillUnbounded(t *testing.T) {
	// The default has to stay "no budget": a limit nobody asked for that
	// silently truncates a long task is worse than no limit at all.
	h, _ := budgetHarness(t, Options{})
	if !h.Budget.Limits().Empty() {
		t.Fatalf("a harness built with no budget options has limits: %+v", h.Budget.Limits())
	}
}

func near(a, b float64) bool { return a-b < 1e-9 && b-a < 1e-9 }
