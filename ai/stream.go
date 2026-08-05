package ai

import (
	"sync"
	"time"
)

// Stream carries the events of one assistant response plus its final message.
//
// Ownership rule, which the whole type depends on: the PRODUCER owns the event
// channel and is the only thing that may Push or Finish. A consumer that wants
// out calls Close, which signals the producer rather than closing the channel
// under it. That keeps "send on closed channel" structurally impossible instead
// of merely unlikely.
//
// Consumption:
//
//	st := streamFn(ctx, opts)
//	for ev := range st.Events() {
//	    ...render...
//	}
//	final := st.Result()
//
// Result blocks until the producer finishes, never returns nil, and is safe to
// call repeatedly from any goroutine.
type Stream struct {
	events chan Event
	done   chan struct{}
	quit   chan struct{}

	finishOnce sync.Once
	quitOnce   sync.Once

	mu     sync.Mutex
	result *Message
}

// NewStream creates an empty stream for a producer to fill.
func NewStream() *Stream {
	return &Stream{
		events: make(chan Event, 64),
		done:   make(chan struct{}),
		quit:   make(chan struct{}),
	}
}

// Events returns the event channel. It closes when the producer finishes.
func (s *Stream) Events() <-chan Event { return s.events }

// Quit closes when a consumer abandons the stream. Producers doing slow work
// between pushes should select on it and stop early.
func (s *Stream) Quit() <-chan struct{} { return s.quit }

// Push emits an event. Producer-only. It reports false once the consumer has
// abandoned the stream, which is the producer's cue to stop and Finish.
func (s *Stream) Push(ev Event) bool {
	select {
	case s.events <- ev:
		return true
	case <-s.quit:
		return false
	}
}

// Finish publishes the final message and closes the stream. Producer-only.
// Only the first call wins, so it is safe to defer.
func (s *Stream) Finish(msg *Message) {
	s.finishOnce.Do(func() {
		s.mu.Lock()
		s.result = msg
		s.mu.Unlock()
		close(s.events)
		close(s.done)
	})
}

// Close abandons the stream from the consumer side. It signals the producer and
// drains any events still in flight so the producer can reach its Finish call
// instead of blocking on a full buffer forever.
func (s *Stream) Close() {
	s.quitOnce.Do(func() {
		close(s.quit)
		go func() {
			for range s.events { //nolint:revive // drain to unblock the producer
			}
		}()
	})
}

// Result blocks until the response finishes and returns the final message.
// It never returns nil: a failed request yields a message with StopError.
func (s *Stream) Result() *Message {
	<-s.done
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.result == nil {
		return &Message{
			Role:         RoleAssistant,
			StopReason:   StopError,
			ErrorMessage: "stream finished without a result",
			Timestamp:    time.Now().UnixMilli(),
		}
	}
	return s.result
}

// ErrorStream builds an already-finished stream carrying a failure.
//
// This is how the StreamFunc contract is honoured: setup failures, bad
// configuration and transport errors all become an ordinary stream whose final
// message has StopError, so the agent loop has exactly one failure path to
// handle rather than two.
func ErrorStream(model, msg string) *Stream {
	s := NewStream()
	final := &Message{
		Role:         RoleAssistant,
		Model:        model,
		StopReason:   StopError,
		ErrorMessage: msg,
		Timestamp:    time.Now().UnixMilli(),
	}
	s.Push(Event{Type: EventError, Reason: StopError, Message: final, Partial: final})
	s.Finish(final)
	return s
}
