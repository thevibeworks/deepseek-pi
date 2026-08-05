//go:build !linux && !darwin

package main

import "os"

// quietEcho does nothing where we cannot speak termios. Bracketed paste still
// works; the markers are just echoed as ^[[200~ around a paste.
func quietEcho(*os.File) (func(), bool) { return nil, false }
