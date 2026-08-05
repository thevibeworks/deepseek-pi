package main

import (
	"os"
	"strings"
	"testing"

	"app/store"
)

// The rename must be complete: the new names work, the old ones are gone from
// the source, and behaviour is unchanged.
func TestRenameApplied(t *testing.T) {
	if _, err := store.Get("x"); err != nil {
		t.Fatalf("store.Get should exist and succeed: %v", err)
	}
	if _, err := store.Get(""); err != store.ErrNotFound {
		t.Errorf("empty id should return store.ErrNotFound, got %v", err)
	}
}

func TestBehaviourUnchanged(t *testing.T) {
	if got := describe("a"); got != "found value:a" {
		t.Errorf("describe(a) = %q, want \"found value:a\"", got)
	}
	if got := describe(""); got != "missing" {
		t.Errorf("describe(\"\") = %q, want \"missing\"", got)
	}
}

func TestOldNamesGone(t *testing.T) {
	for _, path := range []string{"store/store.go", "service.go"} {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("reading %s: %v", path, err)
		}
		for _, old := range []string{"Fetch", "ErrNoRecord"} {
			if strings.Contains(string(data), old) {
				t.Errorf("%s still refers to the old name %q", path, old)
			}
		}
	}
}
