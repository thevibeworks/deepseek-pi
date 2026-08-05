//go:build !unix

package tools

import "os/exec"

// setProcessGroup is a no-op on platforms without POSIX process groups.
func setProcessGroup(_ *exec.Cmd) {}

// killProcessGroup falls back to killing the direct child. Grandchildren may
// survive; that is a known limitation of this build target rather than
// something silently pretended away.
func killProcessGroup(cmd *exec.Cmd) {
	if cmd.Process != nil {
		_ = cmd.Process.Kill()
	}
}
