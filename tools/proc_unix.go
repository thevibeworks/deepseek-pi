//go:build unix

package tools

import (
	"os/exec"
	"syscall"
)

// setProcessGroup puts the command in its own process group so the whole tree
// can be signalled at once.
func setProcessGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

// killProcessGroup terminates the command and every process it spawned.
//
// Killing only the shell is the classic bug: `bash -c "make -j8"` leaves eight
// compilers running, still holding the output pipe, so the read never returns
// and the timeout accomplishes nothing. SIGKILL to the negative pid targets the
// whole group.
func killProcessGroup(cmd *exec.Cmd) {
	if cmd.Process == nil {
		return
	}
	pgid, err := syscall.Getpgid(cmd.Process.Pid)
	if err != nil {
		_ = cmd.Process.Kill()
		return
	}
	_ = syscall.Kill(-pgid, syscall.SIGKILL)
}
