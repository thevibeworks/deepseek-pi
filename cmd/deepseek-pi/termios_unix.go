//go:build linux || darwin

package main

import (
	"os"
	"syscall"
	"unsafe"
)

// quietEcho stops the terminal driver echoing control characters as ^X, and
// returns a function that puts the setting back.
//
// Only one control character matters here: ESC. With ECHOCTL set — the default
// nearly everywhere — the bracketed-paste markers the terminal sends are echoed
// back as a literal "^[[200~" wrapped around every paste. Cleared, they echo as
// the escape sequences they actually are and the terminal ignores them, which
// is the difference between a paste that looks pasted and one that looks like
// the tool is leaking its own protocol.
//
// This is the whole of the terminal handling. Raw mode would buy line editing
// and history too, and cost a real line editor plus a terminal left unusable
// whenever the process dies badly. One flag, restored on exit, is not that.
//
// ok is false when there is no terminal to change, which is not an error.
func quietEcho(f *os.File) (restore func(), ok bool) {
	var before syscall.Termios
	if err := ioctlTermios(f.Fd(), tcGet, &before); err != nil {
		return nil, false
	}

	after := before
	after.Lflag &^= syscall.ECHOCTL
	if err := ioctlTermios(f.Fd(), tcSet, &after); err != nil {
		return nil, false
	}

	return func() {
		// Best effort: if the terminal has gone away there is nothing to
		// restore and nobody to tell.
		_ = ioctlTermios(f.Fd(), tcSet, &before)
	}, true
}

func ioctlTermios(fd uintptr, req uintptr, t *syscall.Termios) error {
	_, _, errno := syscall.Syscall(syscall.SYS_IOCTL, fd, req, uintptr(unsafe.Pointer(t)))
	if errno != 0 {
		return errno
	}
	return nil
}
