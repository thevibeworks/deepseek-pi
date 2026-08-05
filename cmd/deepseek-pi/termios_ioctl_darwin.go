//go:build darwin

package main

import "syscall"

// See the Linux variant: same call, different name.
const (
	tcGet = syscall.TIOCGETA
	tcSet = syscall.TIOCSETA
)
