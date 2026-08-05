package main

import (
	"bytes"
	"context"
	"strings"
	"testing"
)

// newTestInterrupter builds one without installing a real signal handler, so
// the routing can be driven directly.
func newTestInterrupter(w *bytes.Buffer) *interrupter {
	return &interrupter{w: w, s: style{}, done: make(chan struct{})}
}

func TestIdleInterruptDoesNotEndTheSession(t *testing.T) {
	// The bug this exists to prevent: one signal-cancelled context for the whole
	// process means a Ctrl-C at an idle prompt finishes the session before the
	// user has typed anything, and the next prompt fails instantly.
	var out bytes.Buffer
	i := newTestInterrupter(&out)

	i.deliver()

	if !strings.Contains(out.String(), "nothing running") {
		t.Errorf("an idle interrupt said nothing useful: %q", out.String())
	}

	// The session context must be untouched, so the next turn still runs.
	ctx, done := i.arm(context.Background())
	defer done()
	if ctx.Err() != nil {
		t.Fatalf("the turn after an idle interrupt was already cancelled: %v", ctx.Err())
	}
}

func TestInterruptCancelsOnlyTheRunningTurn(t *testing.T) {
	var out bytes.Buffer
	i := newTestInterrupter(&out)

	session, cancelSession := context.WithCancel(context.Background())
	defer cancelSession()

	turn, done := i.arm(session)
	i.deliver()

	if turn.Err() == nil {
		t.Error("the running turn was not cancelled")
	}
	if session.Err() != nil {
		t.Error("cancelling a turn took the session down with it")
	}
	if out.Len() != 0 {
		t.Errorf("an interrupt that cancelled a turn also nagged the user: %q", out.String())
	}
	done()

	// Disarmed again: the next interrupt is idle, not a cancel of stale state.
	i.deliver()
	if !strings.Contains(out.String(), "nothing running") {
		t.Errorf("the interrupter stayed armed after the turn finished: %q", out.String())
	}
}

func TestArmedTurnInheritsSessionCancellation(t *testing.T) {
	// SIGTERM still means stop, and it arrives on the session context.
	var out bytes.Buffer
	i := newTestInterrupter(&out)

	session, cancelSession := context.WithCancel(context.Background())
	turn, done := i.arm(session)
	defer done()

	cancelSession()
	if turn.Err() == nil {
		t.Error("a cancelled session left its turn running")
	}
}
