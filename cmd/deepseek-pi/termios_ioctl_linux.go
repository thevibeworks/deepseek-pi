//go:build linux

package main

import "syscall"

// Linux names the termios ioctls TCGETS/TCSETS; the BSDs use TIOCGETA/TIOCSETA.
// The only per-platform difference in this package.
const (
	tcGet = syscall.TCGETS
	tcSet = syscall.TCSETS
)
