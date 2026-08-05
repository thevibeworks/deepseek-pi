package harness

import (
	"strings"
	"testing"

	"github.com/thevibeworks/deepseek-pi/ai"
)

// fakeStream returns a response with the usage the test dictates, so cache
// attribution can be exercised without a provider.
func fakeStream(input, cacheRead int) ai.StreamFunc {
	return func(_ ai.Context, opts ai.StreamOptions) *ai.Stream {
		st := ai.NewStream()
		go func() {
			st.Finish(&ai.Message{
				Role: ai.RoleAssistant, StopReason: ai.StopEnd, Model: opts.Model,
				Content: []ai.Content{ai.TextContent("ok")},
				Usage:   ai.Usage{Input: input, CacheRead: cacheRead, Output: 10},
			})
		}()
		return st
	}
}

// drive sends one request through the wrapped stream and waits for the result,
// which is when attribution happens.
func drive(t *testing.T, fn ai.StreamFunc, reqCtx ai.Context) {
	t.Helper()
	st := fn(reqCtx, ai.StreamOptions{Model: ai.ModelFlash})
	for range st.Events() {
	}
	st.Result()
}

func toolSet(extra string) []ai.Tool {
	tools := []ai.Tool{{
		Name: "read", Description: "Read a file",
		Parameters: ai.Object(map[string]*ai.Schema{
			"path":   ai.Str("path"),
			"offset": ai.Int("offset"),
			"limit":  ai.Int("limit"),
		}, "path"),
	}}
	if extra != "" {
		tools = append(tools, ai.Tool{Name: extra, Description: "extra"})
	}
	return tools
}

func TestCacheTrackerQuietOnAStablePrefix(t *testing.T) {
	// The normal case, and the one that must never produce noise: the prompt
	// grows by appending, and the whole previous prompt comes back cached.
	tr := NewCacheTracker(ai.MustLookup(ai.ModelFlash))

	drive(t, tr.Wrap(fakeStream(1000, 0)), ai.Context{
		SystemPrompt: "sys", Tools: toolSet(""),
		Messages: []ai.Message{ai.UserMessage("one")},
	})
	drive(t, tr.Wrap(fakeStream(2000, 1000)), ai.Context{
		SystemPrompt: "sys", Tools: toolSet(""),
		Messages: []ai.Message{ai.UserMessage("one"), ai.UserMessage("two")},
	})

	if len(tr.Breaks) != 0 {
		t.Errorf("reported a break on a healthy append-only prefix: %+v", tr.Breaks)
	}
	if tr.WastedTokens != 0 {
		t.Errorf("WastedTokens = %d, want 0", tr.WastedTokens)
	}
}

func TestCacheTrackerAttributesSystemPromptDrift(t *testing.T) {
	// The failure this catches is the expensive one: something time-varying
	// creeps into the system prompt and every turn re-bills the whole prefix.
	tr := NewCacheTracker(ai.MustLookup(ai.ModelFlash))

	drive(t, tr.Wrap(fakeStream(1000, 0)), ai.Context{
		SystemPrompt: "sys at 10:00", Tools: toolSet(""),
		Messages: []ai.Message{ai.UserMessage("one")},
	})
	drive(t, tr.Wrap(fakeStream(2000, 0)), ai.Context{
		SystemPrompt: "sys at 10:01", Tools: toolSet(""),
		Messages: []ai.Message{ai.UserMessage("one"), ai.UserMessage("two")},
	})

	if len(tr.Breaks) != 1 {
		t.Fatalf("got %d breaks, want 1: %+v", len(tr.Breaks), tr.Breaks)
	}
	b := tr.Breaks[0]
	if b.Axis != "system" {
		t.Errorf("Axis = %q, want system", b.Axis)
	}
	if b.Wasted != 1000 {
		t.Errorf("Wasted = %d, want 1000", b.Wasted)
	}
	if b.WastedCost <= 0 {
		t.Error("wasted cost not computed")
	}
	if b.Sanctioned {
		t.Error("a drifting system prompt is not a sanctioned break")
	}
}

func TestCacheTrackerAttributesToolChange(t *testing.T) {
	// Tool schemas live in the cached prefix, which is why modes must change
	// permissions rather than the registered tool set.
	tr := NewCacheTracker(ai.MustLookup(ai.ModelFlash))

	drive(t, tr.Wrap(fakeStream(1000, 0)), ai.Context{
		SystemPrompt: "sys", Tools: toolSet(""),
		Messages: []ai.Message{ai.UserMessage("one")},
	})
	drive(t, tr.Wrap(fakeStream(2000, 0)), ai.Context{
		SystemPrompt: "sys", Tools: toolSet("write"), // a tool appeared
		Messages: []ai.Message{ai.UserMessage("one"), ai.UserMessage("two")},
	})

	if len(tr.Breaks) != 1 || tr.Breaks[0].Axis != "tools" {
		t.Fatalf("expected one break attributed to tools, got %+v", tr.Breaks)
	}
}

func TestCacheTrackerAttributesHistoryRewrite(t *testing.T) {
	// History must be append-only. Rewriting an earlier message is the classic
	// way a serialization change silently doubles the cost of a long session.
	tr := NewCacheTracker(ai.MustLookup(ai.ModelFlash))

	drive(t, tr.Wrap(fakeStream(1000, 0)), ai.Context{
		SystemPrompt: "sys", Tools: toolSet(""),
		Messages: []ai.Message{ai.UserMessage("one"), ai.UserMessage("two")},
	})
	drive(t, tr.Wrap(fakeStream(2000, 0)), ai.Context{
		SystemPrompt: "sys", Tools: toolSet(""),
		Messages: []ai.Message{ai.UserMessage("one"), ai.UserMessage("CHANGED"), ai.UserMessage("three")},
	})

	if len(tr.Breaks) != 1 {
		t.Fatalf("got %d breaks, want 1: %+v", len(tr.Breaks), tr.Breaks)
	}
	b := tr.Breaks[0]
	if b.Axis != "messages" {
		t.Errorf("Axis = %q, want messages", b.Axis)
	}
	// Naming the index is the point; "something changed" sends someone reading
	// the whole transcript by hand.
	if !strings.Contains(b.Detail, "message 1") {
		t.Errorf("Detail should name the rewritten message, got %q", b.Detail)
	}
}

func TestCacheTrackerReportsEarliestCauseOnly(t *testing.T) {
	// A cache breaks at the FIRST differing byte, so when several things
	// changed, the later ones are consequences. Reporting them all sends
	// people chasing effects.
	tr := NewCacheTracker(ai.MustLookup(ai.ModelFlash))

	drive(t, tr.Wrap(fakeStream(1000, 0)), ai.Context{
		SystemPrompt: "sys", Tools: toolSet(""),
		Messages: []ai.Message{ai.UserMessage("one")},
	})
	drive(t, tr.Wrap(fakeStream(2000, 0)), ai.Context{
		SystemPrompt: "DIFFERENT", Tools: toolSet("write"),
		Messages: []ai.Message{ai.UserMessage("CHANGED"), ai.UserMessage("two")},
	})

	if len(tr.Breaks) != 1 {
		t.Fatalf("got %d breaks, want 1", len(tr.Breaks))
	}
	if tr.Breaks[0].Axis != "system" {
		t.Errorf("Axis = %q; the system prompt comes first on the wire, so it is the cause",
			tr.Breaks[0].Axis)
	}
}

func TestCacheTrackerToleratesBlockGranularity(t *testing.T) {
	// The provider caches in blocks, so a small shortfall is granularity, not a
	// broken prefix. Reporting those would bury the real breaks.
	tr := NewCacheTracker(ai.MustLookup(ai.ModelFlash))

	ctx1 := ai.Context{SystemPrompt: "sys", Tools: toolSet(""),
		Messages: []ai.Message{ai.UserMessage("one")}}
	drive(t, tr.Wrap(fakeStream(1000, 0)), ctx1)

	ctx2 := ctx1
	ctx2.Messages = append(append([]ai.Message{}, ctx1.Messages...), ai.UserMessage("two"))
	drive(t, tr.Wrap(fakeStream(2000, 900)), ctx2) // 100 short of 1000

	if len(tr.Breaks) != 0 {
		t.Errorf("reported a break for block granularity: %+v", tr.Breaks)
	}
}

func TestCacheTrackerBlamesProviderWhenPrefixIsIntact(t *testing.T) {
	// If nothing we control changed, saying so is more useful than inventing a
	// cause we cannot support.
	tr := NewCacheTracker(ai.MustLookup(ai.ModelFlash))

	ctx1 := ai.Context{SystemPrompt: "sys", Tools: toolSet(""),
		Messages: []ai.Message{ai.UserMessage("one")}}
	drive(t, tr.Wrap(fakeStream(1000, 0)), ctx1)

	ctx2 := ctx1
	ctx2.Messages = append(append([]ai.Message{}, ctx1.Messages...), ai.UserMessage("two"))
	drive(t, tr.Wrap(fakeStream(2000, 0)), ctx2) // full miss, nothing changed

	if len(tr.Breaks) != 1 {
		t.Fatalf("got %d breaks, want 1", len(tr.Breaks))
	}
	if tr.Breaks[0].Axis != "provider" {
		t.Errorf("Axis = %q, want provider", tr.Breaks[0].Axis)
	}
}

func TestCacheTrackerMarksSanctionedBreaks(t *testing.T) {
	// Compaction breaks the prefix on purpose. It should be visible as a cost
	// but must not read as a defect, or the report cries wolf.
	tr := NewCacheTracker(ai.MustLookup(ai.ModelFlash))

	drive(t, tr.Wrap(fakeStream(1000, 0)), ai.Context{
		SystemPrompt: "sys", Tools: toolSet(""),
		Messages: []ai.Message{ai.UserMessage("one"), ai.UserMessage("two")},
	})

	tr.ExpectBreak("compaction rewrote the transcript")
	drive(t, tr.Wrap(fakeStream(2000, 0)), ai.Context{
		SystemPrompt: "sys", Tools: toolSet(""),
		Messages: []ai.Message{ai.UserMessage("summary"), ai.UserMessage("two")},
	})

	if len(tr.Breaks) != 1 {
		t.Fatalf("got %d breaks, want 1", len(tr.Breaks))
	}
	b := tr.Breaks[0]
	if !b.Sanctioned {
		t.Error("a declared compaction break should be marked sanctioned")
	}
	if !strings.Contains(b.Detail, "compaction") {
		t.Errorf("Detail = %q, want the declared reason", b.Detail)
	}
	if !strings.Contains(b.String(), "expected cache break") {
		t.Errorf("rendering should distinguish it: %q", b.String())
	}

	// The expectation is single-use: the turn after must be judged normally.
	drive(t, tr.Wrap(fakeStream(3000, 0)), ai.Context{
		SystemPrompt: "CHANGED", Tools: toolSet(""),
		Messages: []ai.Message{ai.UserMessage("summary"), ai.UserMessage("two"), ai.UserMessage("three")},
	})
	if len(tr.Breaks) != 2 || tr.Breaks[1].Sanctioned {
		t.Errorf("the expectation leaked past its turn: %+v", tr.Breaks)
	}
}

func TestToolFingerprintIsStableAcrossRuns(t *testing.T) {
	// Schemas hold maps, and Go randomizes map iteration. A fingerprint that
	// varied on its own would report a cache break every single turn — the
	// tool would manufacture the very problem it exists to detect.
	first := hashTools(toolSet("write"))
	for range 200 {
		if got := hashTools(toolSet("write")); got != first {
			t.Fatal("tool fingerprint is not deterministic across map iterations")
		}
	}
	if hashTools(toolSet("")) == first {
		t.Error("fingerprint does not distinguish different tool sets")
	}
}

func TestReportWhenClean(t *testing.T) {
	tr := NewCacheTracker(ai.MustLookup(ai.ModelFlash))
	if !strings.Contains(tr.Report(), "no prompt-cache breaks") {
		t.Errorf("unexpected clean report: %q", tr.Report())
	}
}
