package main

import (
	"fmt"

	"app/store"
)

func describe(id string) string {
	v, err := store.Fetch(id)
	if err == store.ErrNoRecord {
		return "missing"
	}
	return fmt.Sprintf("found %s", v)
}
