package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/signal"
	"sync"
)

// interrupter routes Ctrl-C to whatever is currently running.
//
// The obvious arrangement — one signal-cancelled context for the whole process
// — is wrong for a REPL, and wrong in a way that only shows up when nothing is
// running. A Ctrl-C at an idle prompt cancels that context, the session is
// finished before the user has typed anything, and the next prompt fails
// instantly with "cancelled". Ctrl-C is the key people press to abandon a
// half-typed line; losing the conversation to it is a bad trade.
//
// So the signal is routed rather than broadcast: while a turn is in flight it
// cancels that turn, and while the prompt is idle it says what it did not do.
// SIGTERM keeps the process-wide meaning, because that one really is "stop".
type interrupter struct {
	w  io.Writer
	s  style
	ch chan os.Signal

	mu     sync.Mutex
	cancel context.CancelFunc

	closeOnce sync.Once
	done      chan struct{}
}

func newInterrupter(w io.Writer, s style) *interrupter {
	i := &interrupter{w: w, s: s, ch: make(chan os.Signal, 1), done: make(chan struct{})}
	signal.Notify(i.ch, os.Interrupt)
	go i.run()
	return i
}

func (i *interrupter) run() {
	for {
		select {
		case <-i.ch:
			i.deliver()
		case <-i.done:
			return
		}
	}
}

// deliver routes one interrupt: to the running turn if there is one, to the
// user otherwise.
func (i *interrupter) deliver() {
	i.mu.Lock()
	cancel := i.cancel
	i.mu.Unlock()
	if cancel != nil {
		cancel()
		return
	}
	_, _ = fmt.Fprintf(i.w, "\n%s(nothing running — Ctrl-D or /exit to leave)%s\n", i.s.dim, i.s.reset)
}

// arm makes the next Ctrl-C cancel the returned context. The caller must call
// the returned function when the work finishes, which disarms and releases the
// context — after that a Ctrl-C is idle again.
func (i *interrupter) arm(parent context.Context) (context.Context, func()) {
	ctx, cancel := context.WithCancel(parent)
	i.mu.Lock()
	i.cancel = cancel
	i.mu.Unlock()

	return ctx, func() {
		i.mu.Lock()
		i.cancel = nil
		i.mu.Unlock()
		cancel()
	}
}

func (i *interrupter) stop() {
	i.closeOnce.Do(func() {
		signal.Stop(i.ch)
		close(i.done)
	})
}
