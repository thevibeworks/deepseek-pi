package ai

import (
	"testing"
	"time"
)

func utc(s string) time.Time {
	t, err := time.Parse(time.RFC3339, s)
	if err != nil {
		panic(err)
	}
	return t
}

func TestDefaultIsV41FlashAndProStaysOffered(t *testing.T) {
	if ModelFlash != "deepseek-flash" {
		t.Errorf("ModelFlash = %q, want deepseek-flash", ModelFlash)
	}
	var ids []string
	for _, m := range Models() {
		ids = append(ids, m.ID)
	}
	if len(ids) != 2 || ids[0] != "deepseek-flash" || ids[1] != "deepseek-v4-pro" {
		t.Errorf("Models() = %v, want [deepseek-flash deepseek-v4-pro]", ids)
	}
}

func TestRetiredFlashNamesResolveToFlash(t *testing.T) {
	for _, id := range []string{"deepseek-v4-flash", "deepseek-v4-flash-vision-exp"} {
		m, ok := Lookup(id)
		if !ok || m.ID != ModelFlash || m.Tier != TierFlash {
			t.Errorf("Lookup(%q) = %q %v, want %q", id, m.ID, ok, ModelFlash)
		}
	}
	if _, ok := Lookup("deepseek-v4.1-flash"); ok {
		t.Error("deepseek-v4.1-flash resolved; the direct API refuses that name")
	}
}

func TestCardsAreInTimeOrder(t *testing.T) {
	for i := 1; i < len(Cards); i++ {
		if !Cards[i].Since.After(Cards[i-1].Since) {
			t.Errorf("card %d (%s) does not follow card %d", i, Cards[i].Label, i-1)
		}
	}
}

func TestRatesAtEachBoundary(t *testing.T) {
	flash, pro := MustLookup(ModelFlash), MustLookup(ModelPro)
	cases := []struct {
		name string
		m    Model
		at   string
		want Rates
	}{
		{"flat card, no peak even in a later peak hour", flash, "2026-08-12T02:00:00Z", Rates{0.0028, 0.14, 0.28}},
		{"V4 peak ran on Saturdays before the weekend rule", flash, "2026-08-22T02:00:00Z", Rates{0.014, 0.44, 1.32}},
		{"V4 Saturday after the weekend rule is off-peak", flash, "2026-08-29T02:00:00Z", Rates{0.007, 0.22, 0.66}},
		{"last minute of the V4 card", flash, "2026-09-10T10:59:00Z", Rates{0.007, 0.22, 0.66}},
		{"first minute of the V4.1 card", flash, "2026-09-10T11:00:00Z", Rates{0.003, 0.15, 0.6}},
		{"V4.1 weekday peak", flash, "2026-09-14T01:00:00Z", Rates{0.006, 0.3, 1.2}},
		{"peak end is exclusive", flash, "2026-09-14T04:00:00Z", Rates{0.003, 0.15, 0.6}},
		{"V4.1 Sunday in a peak hour", flash, "2026-09-13T02:00:00Z", Rates{0.003, 0.15, 0.6}},
		{"pro before V4.1", pro, "2026-09-10T10:59:00Z", Rates{0.022, 0.66, 1.98}},
		{"pro unchanged after V4.1", pro, "2026-09-15T12:00:00Z", Rates{0.022, 0.66, 1.98}},
	}
	for _, c := range cases {
		got := c.m.RatesAt(utc(c.at))
		if !approx(got.CacheRead, c.want.CacheRead) || !approx(got.Input, c.want.Input) || !approx(got.Output, c.want.Output) {
			t.Errorf("%s: RatesAt(%s) = %+v, want %+v", c.name, c.at, got, c.want)
		}
	}
}
