package ai

import (
	"bufio"
	"bytes"
	"io"
	"strings"
)

// sseEvent is one server-sent event.
type sseEvent struct {
	Name string
	Data []byte
}

// maxSSELine bounds a single SSE data line. Tool call arguments and long
// thinking blocks arrive as one data line per delta, but a whole file written
// through a write tool can come back as a single large frame, so the ceiling is
// generous. Exceeding it is a protocol error, not something to silently drop.
const maxSSELine = 8 << 20 // 8 MiB

// scanSSE reads server-sent events from r and hands each to fn.
//
// It returns the first read error, or nil at clean EOF. Callers turn that into
// a final message with StopError rather than surfacing an error, per the
// StreamFunc contract.
func scanSSE(r io.Reader, fn func(sseEvent) bool) error {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), maxSSELine)

	var name string
	var data bytes.Buffer

	flush := func() bool {
		if data.Len() == 0 && name == "" {
			return true
		}
		// Trim the trailing newline the multi-line data accumulation adds.
		payload := bytes.TrimSuffix(data.Bytes(), []byte("\n"))
		ev := sseEvent{Name: name, Data: append([]byte(nil), payload...)}
		name = ""
		data.Reset()
		return fn(ev)
	}

	for sc.Scan() {
		line := sc.Text()
		switch {
		case line == "":
			// Blank line dispatches the accumulated event.
			if !flush() {
				return nil
			}
		case strings.HasPrefix(line, ":"):
			// Comment / keep-alive. Ignore.
		case strings.HasPrefix(line, "event:"):
			name = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
		case strings.HasPrefix(line, "data:"):
			data.WriteString(strings.TrimPrefix(strings.TrimPrefix(line, "data:"), " "))
			data.WriteByte('\n')
		default:
			// Unknown field (id:, retry:, ...). Ignore per the SSE spec.
		}
	}
	if err := sc.Err(); err != nil {
		return err
	}
	// A stream that ends without a trailing blank line still has a final event.
	flush()
	return nil
}
