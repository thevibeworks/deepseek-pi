package store

import "errors"

// ErrNoRecord is returned when a lookup finds nothing.
var ErrNoRecord = errors.New("no record")

// Fetch returns the value for id.
func Fetch(id string) (string, error) {
	if id == "" {
		return "", ErrNoRecord
	}
	return "value:" + id, nil
}
