package ai

import (
	"context"
	"math/rand/v2"
	"net/http"
	"strconv"
	"time"
)

// RetryPolicy is the transport-level retry rule.
//
// It defaults to ZERO retries, which is deliberate and copied from pi. There
// are two retry layers and they answer different questions:
//
//   - This one asks "was the HTTP request itself unlucky?" It can only replay
//     the request, so it is useless for anything that already produced tokens.
//   - The session layer (package agent) asks "is this failure worth replaying
//     a whole turn for?" It can see the transcript, classify the error, and
//     continue the loop.
//
// Retrying in both places multiplies attempts and hides real failures behind
// long stalls, so the transport stays quiet by default and the session layer
// owns the policy.
type RetryPolicy struct {
	// MaxRetries is the number of additional attempts after the first.
	MaxRetries int
	// BaseDelay is the first backoff step; it doubles per attempt.
	BaseDelay time.Duration
	// MaxDelay caps exponential backoff.
	MaxDelay time.Duration
	// MaxRetryAfter caps a server-requested delay. A Retry-After longer than
	// this fails immediately instead of stalling the harness, so the caller
	// can decide with the user watching.
	MaxRetryAfter time.Duration
}

// DefaultRetryPolicy returns the transport policy: no retries, sane bounds if
// a caller opts in by raising MaxRetries.
func DefaultRetryPolicy() RetryPolicy {
	return RetryPolicy{
		MaxRetries:    0,
		BaseDelay:     500 * time.Millisecond,
		MaxDelay:      8 * time.Second,
		MaxRetryAfter: 60 * time.Second,
	}
}

// retryableStatus reports whether an HTTP status is worth another attempt.
func retryableStatus(status int) bool {
	switch status {
	case http.StatusRequestTimeout, // 408
		http.StatusConflict,        // 409
		http.StatusTooManyRequests: // 429
		return true
	}
	return status >= 500
}

// next decides whether to retry attempt n (0-based) and how long to wait.
//
// status 0 means a transport error with no response. A retryAfter header is
// honoured when present and within MaxRetryAfter; beyond that the request fails
// rather than silently parking the session for minutes.
func (p RetryPolicy) next(attempt, status int, retryAfter string) (time.Duration, bool) {
	if attempt >= p.MaxRetries {
		return 0, false
	}
	if status != 0 && !retryableStatus(status) {
		return 0, false
	}

	if d, ok := parseRetryAfter(retryAfter); ok {
		if p.MaxRetryAfter > 0 && d > p.MaxRetryAfter {
			return 0, false
		}
		return d, true
	}

	base := p.BaseDelay
	if base <= 0 {
		base = 500 * time.Millisecond
	}
	delay := base << attempt
	if p.MaxDelay > 0 && delay > p.MaxDelay {
		delay = p.MaxDelay
	}
	// Full jitter on the upper half: enough to break up thundering herds from
	// parallel sub-agents without making the wait unpredictable to a watcher.
	jitter := time.Duration(rand.Int64N(int64(delay/2) + 1))
	return delay/2 + jitter, true
}

// parseRetryAfter reads both Retry-After forms: delta-seconds and HTTP-date.
func parseRetryAfter(v string) (time.Duration, bool) {
	if v == "" {
		return 0, false
	}
	if secs, err := strconv.Atoi(v); err == nil {
		if secs < 0 {
			return 0, false
		}
		return time.Duration(secs) * time.Second, true
	}
	if t, err := http.ParseTime(v); err == nil {
		d := time.Until(t)
		if d < 0 {
			return 0, true
		}
		return d, true
	}
	return 0, false
}

// sleepCtx waits for d, reporting false if the context ended first.
func sleepCtx(ctx context.Context, d time.Duration) bool {
	if d <= 0 {
		return ctx.Err() == nil
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return true
	case <-ctx.Done():
		return false
	}
}
