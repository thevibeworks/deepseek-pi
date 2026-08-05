package ai

import "strings"

// Model ids served by api.deepseek.com.
const (
	ModelFlash = "deepseek-v4-flash"
	ModelPro   = "deepseek-v4-pro"
)

// DefaultBaseURL is the DeepSeek API root. The Anthropic-compatible Messages
// endpoint hangs off /anthropic/v1/messages.
const DefaultBaseURL = "https://api.deepseek.com"

// Cost is a USD breakdown for one request.
type Cost struct {
	Input     float64 `json:"input"`
	Output    float64 `json:"output"`
	CacheRead float64 `json:"cacheRead"`
	Total     float64 `json:"total"`
}

// Add accumulates another cost into this one.
func (c *Cost) Add(other Cost) {
	c.Input += other.Input
	c.Output += other.Output
	c.CacheRead += other.CacheRead
	c.Total += other.Total
}

// Rates is a published per-million-token rate card, in USD.
type Rates struct {
	// CacheRead is the rate for prompt tokens served from the automatic cache.
	CacheRead float64
	// Input is the rate for prompt tokens that missed the cache. On DeepSeek
	// a cache miss is billed as ordinary input, not as a cache write.
	Input float64
	// Output is the rate for generated tokens, thinking included.
	Output float64
}

// Model is everything the harness needs to know about a DeepSeek model.
//
// The numbers here are the real ones, not the advertised ones. v4 advertises a
// 1M context window but only ~616k of it is usable input; budgeting against
// the advertised figure is how a harness walks into hard truncation at 62%
// utilisation and blames the model.
type Model struct {
	ID   string
	Name string
	// ContextWindow is the usable input budget in tokens. All budget math uses
	// this, never the advertised window.
	ContextWindow int
	// AdvertisedWindow is what the docs claim. Recorded so the gap is visible
	// rather than folklore.
	AdvertisedWindow int
	MaxTokens        int
	Reasoning        bool
	Efforts          []Effort
	Rates            Rates
}

var catalog = map[string]Model{
	ModelFlash: {
		ID:               ModelFlash,
		Name:             "DeepSeek V4 Flash",
		ContextWindow:    616_000,
		AdvertisedWindow: 1_000_000,
		MaxTokens:        384_000,
		Reasoning:        true,
		Efforts:          []Effort{EffortLow, EffortHigh, EffortXHigh, EffortMax},
		Rates:            Rates{CacheRead: 0.0028, Input: 0.14, Output: 0.28},
	},
	ModelPro: {
		ID:               ModelPro,
		Name:             "DeepSeek V4 Pro",
		ContextWindow:    616_000,
		AdvertisedWindow: 1_000_000,
		MaxTokens:        384_000,
		Reasoning:        true,
		Efforts:          []Effort{EffortLow, EffortHigh, EffortXHigh, EffortMax},
		Rates:            Rates{CacheRead: 0.003625, Input: 0.435, Output: 0.87},
	},
}

// Models returns the catalog in a stable order: flash first, since it is the
// default executor and the one a cost-conscious loop should reach for.
func Models() []Model { return []Model{catalog[ModelFlash], catalog[ModelPro]} }

// Lookup resolves a model id to its profile.
//
// The Anthropic-compatible endpoint also accepts Claude model names and remaps
// them server-side, so a harness pointed at DeepSeek with Claude defaults still
// works. Resolve those to the model that actually ran, otherwise cost lands on
// the wrong rate card.
func Lookup(id string) (Model, bool) {
	if m, ok := catalog[id]; ok {
		return m, true
	}
	switch {
	case strings.HasPrefix(id, "claude-opus"):
		return catalog[ModelPro], true
	case strings.HasPrefix(id, "claude-"):
		return catalog[ModelFlash], true
	}
	return Model{}, false
}

// MustLookup resolves a model id, falling back to flash for unknown ids. Used
// on response paths where an unknown id must not stop the run.
func MustLookup(id string) Model {
	if m, ok := Lookup(id); ok {
		return m
	}
	return catalog[ModelFlash]
}

// SupportsEffort reports whether the model accepts a reasoning level.
func (m Model) SupportsEffort(e Effort) bool {
	if e == EffortOff {
		return true
	}
	for _, v := range m.Efforts {
		if v == e {
			return true
		}
	}
	return false
}

// Price computes cost from raw token counts.
//
// Always compute cost here from provider-reported tokens. Never trust a cost
// field returned by a provider or another harness, including a previous version
// of this one: token counts are facts, cost fields are somebody's arithmetic.
func (m Model) Price(u Usage) Cost {
	const perMillion = 1_000_000.0
	c := Cost{
		CacheRead: float64(u.CacheRead) * m.Rates.CacheRead / perMillion,
		Input:     float64(u.CacheMiss()) * m.Rates.Input / perMillion,
		Output:    float64(u.Output) * m.Rates.Output / perMillion,
	}
	c.Total = c.CacheRead + c.Input + c.Output
	return c
}

// CacheSavings is what the cached prompt tokens would have cost at the miss
// rate minus what they did cost. This is the number that justifies structuring
// a prompt for cache reuse, and it is the reason the loop never rebuilds
// history per turn.
func (m Model) CacheSavings(u Usage) float64 {
	const perMillion = 1_000_000.0
	return float64(u.CacheRead) * (m.Rates.Input - m.Rates.CacheRead) / perMillion
}
