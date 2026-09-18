package main

import (
	"os"
	"regexp"
	"strconv"
	"testing"
	"time"

	"github.com/thevibeworks/deepseek-pi/ai"
)

// The README's model and price claims, asserted against the catalog so a
// stale copy turns CI red in the run that caused the drift. Every occurrence
// is checked, not the first: a fresh first copy must not hide a stale second.

func readme(t *testing.T) string {
	t.Helper()
	b, err := os.ReadFile("../../README.md")
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

// newestCard is the last card, which is what the README publishes. Read off the
// list rather than time.Now() so this cannot flip on a date or at peak.
func newestCard() ai.Card { return ai.Cards[len(ai.Cards)-1] }

func TestReadmeDefaultModelClaims(t *testing.T) {
	doc := readme(t)
	if normalizeModel("") != ai.ModelFlash {
		t.Fatalf("an empty -m resolves to %q, not ai.ModelFlash", normalizeModel(""))
	}

	// "defaulting to DeepSeek-V4.1-Flash (`deepseek-flash`)" in the intro.
	intro := regexp.MustCompile("defaulting to [^(]+\\(`([^`]+)`\\)").FindAllStringSubmatch(doc, -1)
	// "| `flash` (default) | `deepseek-flash`, ..." in the models table.
	table := regexp.MustCompile("\\| `(\\w+)` \\(default\\) \\| `([^`]+)`").FindAllStringSubmatch(doc, -1)
	// "(model: deepseek-flash)" in the usage block.
	usage := regexp.MustCompile(`\(model: (deepseek-[\w.-]+)\)`).FindAllStringSubmatch(doc, -1)
	if len(intro) != 1 || len(table) != 1 || len(usage) != 1 {
		t.Fatalf("expected one of each default claim, got intro=%d table=%d usage=%d", len(intro), len(table), len(usage))
	}
	for _, id := range []string{intro[0][1], table[0][2], usage[0][1]} {
		if id != ai.ModelFlash {
			t.Errorf("README names %q as the default; the code defaults to %q", id, ai.ModelFlash)
		}
	}
	if normalizeModel(table[0][1]) != ai.ModelFlash {
		t.Errorf("README's default -m alias %q resolves to %q", table[0][1], normalizeModel(table[0][1]))
	}
}

func TestReadmeModelsTableMatchesNewestCard(t *testing.T) {
	rows := regexp.MustCompile("(?m)^\\| `(\\w+)`[^|]*\\| `([^`]+)`[^|]*\\| ([\\d.]+) / ([\\d.]+) / ([\\d.]+) \\|$").
		FindAllStringSubmatch(readme(t), -1)
	models := ai.Models()
	if len(rows) != len(models) {
		t.Fatalf("README models table has %d rows, catalog has %d models", len(rows), len(models))
	}
	card := newestCard()
	for i, row := range rows {
		m := models[i]
		if normalizeModel(row[1]) != m.ID || row[2] != m.ID {
			t.Errorf("row %d: -m %s / %s, catalog order says %s", i, row[1], row[2], m.ID)
		}
		want := card.Flash
		if m.Tier == ai.TierPro {
			want = card.Pro
		}
		got := [3]float64{num(t, row[3]), num(t, row[4]), num(t, row[5])}
		if got != [3]float64{want.CacheRead, want.Input, want.Output} {
			t.Errorf("%s: README says %v, the %s card says %+v", m.ID, got, card.Label, want)
		}
	}
}

func TestReadmeCacheSwingMatchesNewestFlashCard(t *testing.T) {
	m := regexp.MustCompile(`Cached input runs ([\d.]+)/M against ([\d.]+)/M`).FindAllStringSubmatch(readme(t), -1)
	if len(m) != 1 {
		t.Fatalf("expected one cache-swing claim, found %d", len(m))
	}
	f := newestCard().Flash
	if num(t, m[0][1]) != f.CacheRead || num(t, m[0][2]) != f.Input {
		t.Errorf("README cache swing %s/%s, flash card %v/%v", m[0][1], m[0][2], f.CacheRead, f.Input)
	}
}

func TestNewestCardIsTheOneInForce(t *testing.T) {
	// A card dated in the future would make the README publish prices nobody
	// is billed yet.
	if newestCard().Since.After(time.Now()) {
		t.Errorf("newest card %s starts %s, in the future", newestCard().Label, newestCard().Since)
	}
}

func num(t *testing.T, s string) float64 {
	t.Helper()
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		t.Fatal(err)
	}
	return f
}
