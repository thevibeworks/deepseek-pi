package kv

import "testing"

func TestParse(t *testing.T) {
	got, err := Parse("a=1\nb = two \n\n# comment\nc=x=y\n")
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	want := map[string]string{"a": "1", "b": "two", "c": "x=y"}
	if len(got) != len(want) {
		t.Fatalf("got %d keys (%v), want %d", len(got), got, len(want))
	}
	for k, v := range want {
		if got[k] != v {
			t.Errorf("key %q = %q, want %q", k, got[k], v)
		}
	}
}

func TestParseRejectsMissingEquals(t *testing.T) {
	if _, err := Parse("novalue\n"); err == nil {
		t.Error("expected an error for a line with no '='")
	}
}

func TestParseEmpty(t *testing.T) {
	got, err := Parse("")
	if err != nil {
		t.Fatalf("Parse(\"\"): %v", err)
	}
	if len(got) != 0 {
		t.Errorf("got %v, want an empty map", got)
	}
}
