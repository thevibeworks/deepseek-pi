package ai

import "time"

// Tier is a row of DeepSeek's rate card. Every model name the API accepts
// bills on one of the two.
type Tier string

// The two rows of the card.
const (
	TierFlash Tier = "flash"
	TierPro   Tier = "pro"
)

// Card is one published rate card: what each tier cost from Since until the
// next card. With TimeOfUse the rates are OFF-PEAK, and PeakMultiplier times
// them during peak hours.
//
// Superseded cards stay: a stored session's cost was computed at the time,
// and anything that re-prices a past call must use the card in force then.
// Source: https://api-docs.deepseek.com/quick_start/pricing and the changelog.
type Card struct {
	Label     string
	Since     time.Time
	TimeOfUse bool
	Flash     Rates
	Pro       Rates
}

var proV4 = Rates{CacheRead: 0.022, Input: 0.66, Output: 1.98}

// Cards are in time order.
var Cards = []Card{
	{
		// Published 2026-08-02, flat, no peak.
		Label: "flat",
		Flash: Rates{CacheRead: 0.0028, Input: 0.14, Output: 0.28},
		Pro:   Rates{CacheRead: 0.003625, Input: 0.435, Output: 0.87},
	},
	{
		// V4 GA repricing, effective 16:00 UTC 2026-08-16 (changelog 2026-08-13).
		Label:     "V4",
		Since:     time.Date(2026, time.August, 16, 16, 0, 0, 0, time.UTC),
		TimeOfUse: true,
		Flash:     Rates{CacheRead: 0.007, Input: 0.22, Output: 0.66},
		Pro:       proV4,
	},
	{
		// V4.1 Flash (changelog 2026-09-10): Flash cut on every item, Pro
		// unchanged. The instant is INFERRED: our docs mirror read the old card
		// at 04:50 UTC and the new one at 11:27 UTC that day, and 11:00 is the
		// last whole hour between, so a call in the gap can only be overstated.
		Label:     "V4.1",
		Since:     time.Date(2026, time.September, 10, 11, 0, 0, 0, time.UTC),
		TimeOfUse: true,
		Flash:     Rates{CacheRead: 0.003, Input: 0.15, Output: 0.6},
		Pro:       proV4,
	},
}

// PeakMultiplier scales the off-peak card during peak hours. DeepSeek
// publishes the peak figures, and they are exactly double.
const PeakMultiplier = 2.0

// weekdaysOnlySince is when weekends stopped billing peak: 16:00 UTC on
// 2026-08-22, midnight Beijing going into Sunday. Before it peak ran daily.
var weekdaysOnlySince = time.Date(2026, time.August, 22, 16, 0, 0, 0, time.UTC)

// CardAt returns the card in force at t.
func CardAt(t time.Time) Card {
	card := Cards[0]
	for _, c := range Cards {
		if !t.Before(c.Since) {
			card = c
		}
	}
	return card
}

// IsPeak reports whether t is in a peak window: 01:00-04:00 and 06:00-10:00
// UTC (09-12 and 14-18 Beijing), Monday to Friday. Every peak hour falls on
// the same calendar day in UTC and Beijing, so the UTC weekday decides.
func IsPeak(t time.Time) bool {
	if !CardAt(t).TimeOfUse {
		return false
	}
	u := t.UTC()
	if !u.Before(weekdaysOnlySince) && (u.Weekday() == time.Saturday || u.Weekday() == time.Sunday) {
		return false
	}
	h := u.Hour()
	return (h >= 1 && h < 4) || (h >= 6 && h < 10)
}

// RatesAt is the model's per-million rate card at t, peak applied.
func (m Model) RatesAt(t time.Time) Rates {
	card := CardAt(t)
	r := card.Flash
	if m.Tier == TierPro {
		r = card.Pro
	}
	if IsPeak(t) {
		r = Rates{CacheRead: r.CacheRead * PeakMultiplier, Input: r.Input * PeakMultiplier, Output: r.Output * PeakMultiplier}
	}
	return r
}
