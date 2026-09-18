package harness

import (
	"fmt"
	"hash/fnv"
	"strings"
	"sync"
	"time"

	"github.com/thevibeworks/deepseek-pi/ai"
)

// CacheTracker explains prompt-cache misses.
//
// The whole cost model of this project rests on the prefix cache: cached input
// is 0.003/M against 0.15/M for a miss on V4.1 Flash, a 50x swing. Everything is arranged to
// protect it — a byte-stable system prompt, append-only history, tool schemas
// that never change with a mode switch. But "protected" was an assumption. This
// makes it an observation.
//
// The mechanism: the prompt sent on turn N is, by construction, the prompt from
// turn N-1 plus whatever the turn appended. So the ENTIRE previous prompt should
// come back as cache_read. When it does not, something before the append point
// changed, and the first thing that differs is the culprit — a cache breaks at
// the first differing byte, so anything after it is a consequence, not a cause.
type CacheTracker struct {
	mu   sync.Mutex
	prev *snapshot
	turn int

	model ai.Model

	// Breaks records every unexplained miss, newest last.
	Breaks []CacheBreak
	// WastedTokens is the running total of prompt tokens re-billed at the input
	// rate that a stable prefix would have served from cache.
	WastedTokens int

	// OnBreak reports a break as it happens.
	OnBreak func(CacheBreak)
	// Expect suppresses the next break report, for a prefix break we chose:
	// compaction is the one sanctioned one.
	expected string
}

// snapshot is what a request looked like, reduced to comparable fingerprints.
type snapshot struct {
	system   uint64
	tools    uint64
	messages []uint64
	// inputTokens is the provider-reported prompt size for this request, which
	// is what the NEXT request should find in cache.
	inputTokens int
}

// CacheBreak is one prefix-cache miss with its cause.
type CacheBreak struct {
	Turn int
	// Axis is which part of the prompt changed: "system", "tools", "messages".
	Axis string
	// Detail names the specific change, e.g. "message 4 (toolResult) rewritten".
	Detail string
	// Expected is what should have been served from cache.
	Expected int
	// Actual is what was.
	Actual int
	// Wasted is the shortfall, re-billed at the full input rate.
	Wasted int
	// WastedCost is that shortfall in USD.
	WastedCost float64
	// Sanctioned marks a break we chose to take, such as compaction.
	Sanctioned bool
}

func (b CacheBreak) String() string {
	kind := "cache break"
	if b.Sanctioned {
		kind = "expected cache break"
	}
	return fmt.Sprintf("%s at turn %d: %s changed (%s) — %d tokens re-billed, $%.4f",
		kind, b.Turn, b.Axis, b.Detail, b.Wasted, b.WastedCost)
}

// NewCacheTracker builds a tracker for a model.
func NewCacheTracker(model ai.Model) *CacheTracker {
	return &CacheTracker{model: model}
}

// ExpectBreak declares that the next request will legitimately break the
// prefix, so it is reported as sanctioned rather than as a defect. Compaction
// is the only caller: it rewrites history on purpose.
func (t *CacheTracker) ExpectBreak(reason string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.expected = reason
}

// Reset forgets history. Used when the transcript is replaced wholesale, where
// comparing against the old prompt would attribute a break to the wrong thing.
func (t *CacheTracker) Reset() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.prev = nil
}

// Wrap instruments a StreamFunc.
//
// A middleware rather than a hook inside the loop, because attribution needs
// the request and the response together, and the StreamFunc boundary is the
// only place both exist.
func (t *CacheTracker) Wrap(next ai.StreamFunc) ai.StreamFunc {
	return func(reqCtx ai.Context, opts ai.StreamOptions) *ai.Stream {
		current := fingerprint(reqCtx)

		t.mu.Lock()
		prev := t.prev
		t.turn++
		turn := t.turn
		expected := t.expected
		t.expected = ""
		t.mu.Unlock()

		stream := next(reqCtx, opts)

		// Observe the result without holding up the consumer: wrapping the
		// stream would mean buffering it, and the renderer needs the deltas as
		// they arrive.
		out := ai.NewStream()
		go func() {
			for ev := range stream.Events() {
				if !out.Push(ev) {
					break
				}
			}
			final := stream.Result()
			current.inputTokens = final.Usage.Input
			t.record(turn, prev, current, final.Usage, expected)
			out.Finish(final)
		}()
		return out
	}
}

// record compares this request against the previous one and attributes any
// shortfall in cache reads.
func (t *CacheTracker) record(turn int, prev *snapshot, current snapshot, usage ai.Usage, expectedReason string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.prev = &current

	if prev == nil || prev.inputTokens == 0 {
		return // nothing to compare against; the first turn cannot hit a cache
	}

	// The previous prompt is a prefix of this one when nothing before the
	// append point changed, so all of it should come back as cache_read.
	expected := prev.inputTokens
	if expected > usage.Input {
		expected = usage.Input
	}
	shortfall := expected - usage.CacheRead

	// DeepSeek caches in blocks, so a small shortfall is granularity rather
	// than a broken prefix. Reporting those would bury the real breaks.
	const blockSlack = 256
	if shortfall <= blockSlack {
		return
	}

	axis, detail := attribute(prev, &current)
	if expectedReason != "" {
		axis, detail = "messages", expectedReason
	}

	brk := CacheBreak{
		Turn: turn, Axis: axis, Detail: detail,
		Expected: expected, Actual: usage.CacheRead, Wasted: shortfall,
		WastedCost: float64(shortfall) * t.model.RatesAt(time.Now()).Input / 1_000_000,
		Sanctioned: expectedReason != "",
	}
	t.Breaks = append(t.Breaks, brk)
	t.WastedTokens += shortfall
	if t.OnBreak != nil {
		t.OnBreak(brk)
	}
}

// attribute finds the first component that differs.
//
// Order matters and mirrors the wire layout: system prompt, then tool
// definitions, then messages. A cache breaks at the first differing byte, so
// the earliest difference is the cause and everything after it is downstream.
func attribute(prev, current *snapshot) (axis, detail string) {
	if prev.system != current.system {
		return "system", "the system prompt is not byte-identical to the previous turn"
	}
	if prev.tools != current.tools {
		return "tools", "a tool definition changed; schemas live in the cached prefix"
	}

	for i := range prev.messages {
		if i >= len(current.messages) {
			return "messages", fmt.Sprintf(
				"history shrank from %d to %d messages", len(prev.messages), len(current.messages))
		}
		if prev.messages[i] != current.messages[i] {
			return "messages", fmt.Sprintf(
				"message %d of %d was rewritten; history must be append-only", i, len(prev.messages))
		}
	}
	// Every retained message matches, so the prefix is intact and the miss is
	// the provider's, not ours. Worth saying plainly rather than inventing a
	// cause.
	return "provider", "the prefix was unchanged; the provider did not serve it from cache"
}

func fingerprint(reqCtx ai.Context) snapshot {
	s := snapshot{
		system:   hashString(reqCtx.SystemPrompt),
		tools:    hashTools(reqCtx.Tools),
		messages: make([]uint64, 0, len(reqCtx.Messages)),
	}
	for _, m := range reqCtx.Messages {
		s.messages = append(s.messages, hashMessage(m))
	}
	return s
}

func hashString(s string) uint64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(s))
	return h.Sum64()
}

func hashTools(tools []ai.Tool) uint64 {
	var b strings.Builder
	for _, t := range tools {
		b.WriteString(t.Name)
		b.WriteByte(0)
		b.WriteString(t.Description)
		b.WriteByte(0)
		writeSchema(&b, t.Parameters)
		b.WriteByte(0)
	}
	return hashString(b.String())
}

// writeSchema renders a schema deterministically. Required order is preserved
// as declared, and properties are walked in sorted order, because Go randomizes
// map iteration and a fingerprint that changes on its own would report a cache
// break on every turn.
func writeSchema(b *strings.Builder, s *ai.Schema) {
	if s == nil {
		return
	}
	b.WriteString(s.Type)
	b.WriteByte(1)
	b.WriteString(s.Description)
	b.WriteByte(1)
	for _, r := range s.Required {
		b.WriteString(r)
		b.WriteByte(2)
	}
	for _, k := range sortedKeys(s.Properties) {
		b.WriteString(k)
		b.WriteByte(3)
		writeSchema(b, s.Properties[k])
	}
	writeSchema(b, s.Items)
}

func sortedKeys(m map[string]*ai.Schema) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	// Insertion sort: tool schemas have a handful of properties, and this
	// avoids a sort import for the sake of five elements.
	for i := 1; i < len(keys); i++ {
		for j := i; j > 0 && keys[j] < keys[j-1]; j-- {
			keys[j], keys[j-1] = keys[j-1], keys[j]
		}
	}
	return keys
}

func hashMessage(m ai.Message) uint64 {
	var b strings.Builder
	b.WriteString(string(m.Role))
	b.WriteByte(0)
	b.WriteString(m.ToolCallID)
	b.WriteByte(0)
	for _, c := range m.Content {
		b.WriteString(string(c.Type))
		b.WriteByte(1)
		b.WriteString(c.Text)
		b.WriteString(c.Thinking)
		b.WriteString(c.Signature)
		b.WriteString(c.ID)
		b.WriteString(c.Name)
		b.Write(c.Arguments)
		b.WriteByte(1)
	}
	return hashString(b.String())
}

// Report renders the accumulated breaks.
func (t *CacheTracker) Report() string {
	t.mu.Lock()
	defer t.mu.Unlock()

	if len(t.Breaks) == 0 {
		return "no prompt-cache breaks recorded\n"
	}
	var b strings.Builder
	var cost float64
	for _, brk := range t.Breaks {
		fmt.Fprintf(&b, "  %s\n", brk.String())
		cost += brk.WastedCost
	}
	fmt.Fprintf(&b, "  total: %d tokens re-billed, $%.4f\n", t.WastedTokens, cost)
	return b.String()
}
