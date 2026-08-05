package harness

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/thevibeworks/deepseek-pi/ai"
)

// EntryKind discriminates session log entries.
type EntryKind string

const (
	// EntryHeader is the first line of every session file.
	EntryHeader EntryKind = "header"
	// EntryMessage is one transcript message.
	EntryMessage EntryKind = "message"
	// EntryModelChange records a model switch.
	EntryModelChange EntryKind = "model_change"
	// EntryCompaction records that the context view was compacted. Storage
	// keeps every message; only the view the model sees shrinks, so this marks
	// where that divergence happened.
	EntryCompaction EntryKind = "compaction"
	// EntryRewind records that the context view was cut back to an earlier
	// point. Same principle as compaction: the file keeps every message and
	// only the replayed view shrinks, so the transcript still shows the path
	// that was abandoned.
	EntryRewind EntryKind = "rewind"
	// EntrySnapshot records one file's state around a tool call, so a later
	// rewind can put the workspace back as well as the conversation.
	EntrySnapshot EntryKind = "snapshot"
)

// Entry is one JSONL line.
type Entry struct {
	Kind      EntryKind   `json:"kind"`
	Timestamp int64       `json:"timestamp"`
	Message   *ai.Message `json:"message,omitempty"`

	// Header fields.
	SessionID string `json:"sessionId,omitempty"`
	Cwd       string `json:"cwd,omitempty"`
	Model     string `json:"model,omitempty"`
	Version   int    `json:"version,omitempty"`
	// Parent links a forked session to the one it branched from.
	Parent string `json:"parent,omitempty"`

	// Compaction fields.
	BeforeTokens int    `json:"beforeTokens,omitempty"`
	AfterTokens  int    `json:"afterTokens,omitempty"`
	Reclaimed    int    `json:"reclaimedBytes,omitempty"`
	UsedLLM      bool   `json:"usedLlm,omitempty"`
	Note         string `json:"note,omitempty"`

	// Retained is how many messages survive a rewind. Zero is a meaningful
	// value — rewinding to an empty context — and omitempty drops it, which is
	// harmless only because an absent field unmarshals back to the same zero.
	Retained int `json:"retained,omitempty"`

	// Snapshot carries a file capture.
	Snapshot *FileSnapshot `json:"snapshot,omitempty"`
}

// SessionVersion is the on-disk format version.
const SessionVersion = 1

// Session appends a transcript to a JSONL file.
//
// The file is created LAZILY, on the first entry written. A session that the
// user abandons at the prompt leaves nothing behind, so `deepseek-pi --resume`
// lists real work rather than a graveyard of empty files.
type Session struct {
	mu sync.Mutex

	id      string
	path    string
	cwd     string
	model   string
	parent  string
	created bool
	file    *os.File
	writer  *bufio.Writer

	// Usage accumulates across the whole session.
	Usage ai.Usage
}

// NewSession prepares a session. Nothing touches disk until the first append.
func NewSession(id, cwd, model string) *Session {
	return &Session{
		id:    id,
		cwd:   cwd,
		model: model,
		path:  sessionPath(cwd, id),
	}
}

// ID returns the session identifier.
func (s *Session) ID() string { return s.id }

// Path returns the file this session writes to, whether or not it exists yet.
func (s *Session) Path() string { return s.path }

func sessionsRoot() string {
	if dir := os.Getenv("DEEPSEEK_PI_HOME"); dir != "" {
		return filepath.Join(dir, "sessions")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return filepath.Join(os.TempDir(), "deepseek-pi", "sessions")
	}
	return filepath.Join(home, ".deepseek-pi", "sessions")
}

// slugForCwd turns a workspace path into a directory name that is unique,
// readable, and free of separators.
func slugForCwd(cwd string) string {
	s := strings.TrimPrefix(filepath.Clean(cwd), string(filepath.Separator))
	s = strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_':
			return r
		default:
			return '-'
		}
	}, s)
	if s == "" {
		s = "root"
	}
	return s
}

func sessionPath(cwd, id string) string {
	name := fmt.Sprintf("%s_%s.jsonl", time.Now().UTC().Format("20060102T150405Z"), id)
	return filepath.Join(sessionsRoot(), slugForCwd(cwd), name)
}

func (s *Session) ensureFile() error {
	if s.created {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(s.path), 0o755); err != nil {
		return fmt.Errorf("creating session directory: %w", err)
	}
	f, err := os.OpenFile(s.path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return fmt.Errorf("creating session file: %w", err)
	}
	s.file, s.writer, s.created = f, bufio.NewWriter(f), true

	return s.writeEntry(Entry{
		Kind: EntryHeader, Timestamp: time.Now().UnixMilli(),
		SessionID: s.id, Cwd: s.cwd, Model: s.model, Version: SessionVersion,
		Parent: s.parent,
	})
}

// writeEntry appends and flushes. Flushing per entry costs a syscall and buys
// a transcript that survives a crash, which is the case that matters: a session
// log is only useful if it records the run that went wrong.
func (s *Session) writeEntry(e Entry) error {
	data, err := json.Marshal(e)
	if err != nil {
		return err
	}
	if _, err := s.writer.Write(append(data, '\n')); err != nil {
		return err
	}
	return s.writer.Flush()
}

// Append records a message and accumulates its usage.
func (s *Session) Append(msg ai.Message) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if msg.Role == ai.RoleAssistant {
		s.Usage.Add(msg.Usage)
	}
	if err := s.ensureFile(); err != nil {
		return err
	}
	m := msg
	return s.writeEntry(Entry{
		Kind: EntryMessage, Timestamp: time.Now().UnixMilli(), Message: &m,
	})
}

// RecordModelChange notes a model switch in the transcript.
func (s *Session) RecordModelChange(model string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.model = model
	if err := s.ensureFile(); err != nil {
		return err
	}
	return s.writeEntry(Entry{
		Kind: EntryModelChange, Timestamp: time.Now().UnixMilli(), Model: model,
	})
}

// RecordCompaction notes a compaction in the transcript.
//
// Compaction shrinks the CONTEXT VIEW, never storage: every message stays on
// disk. This entry is what lets a later reader tell the difference between
// "the model never saw this" and "the model saw it and then it aged out".
func (s *Session) RecordCompaction(ev CompactionEvent) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ensureFile(); err != nil {
		return err
	}
	note := ""
	if ev.Err != nil {
		note = "summarization failed, used the deterministic fallback: " + ev.Err.Error()
	}
	return s.writeEntry(Entry{
		Kind: EntryCompaction, Timestamp: time.Now().UnixMilli(),
		BeforeTokens: ev.BeforeToken, AfterTokens: ev.AfterToken,
		Reclaimed: ev.Reclaimed, UsedLLM: ev.UsedLLM, Note: note,
	})
}

// RecordRewind marks that the context was cut back to the first n messages.
//
// Deleting the lines instead would destroy the record of what was tried, which
// is the one thing a transcript is for. A marker keeps both the history and the
// view, and LoadSession replays it.
func (s *Session) RecordRewind(retained int) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ensureFile(); err != nil {
		return err
	}
	return s.writeEntry(Entry{
		Kind: EntryRewind, Timestamp: time.Now().UnixMilli(), Retained: retained,
	})
}

// RecordSnapshot appends a file capture.
func (s *Session) RecordSnapshot(snap FileSnapshot) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ensureFile(); err != nil {
		return err
	}
	return s.writeEntry(Entry{
		Kind: EntrySnapshot, Timestamp: time.Now().UnixMilli(), Snapshot: &snap,
	})
}

// Fork writes msgs to a NEW session file that records where it branched from.
//
// The receiver is untouched, and the copy is a full copy rather than a pointer
// back to it: a transcript that stops making sense when another file is deleted
// is not a record. Disk is cheap.
//
// Workspace and model come from the parent rather than from the caller, so a
// branch cannot land in a different session directory than the session it came
// from — which would quietly hide it from `deepseek-pi -sessions`.
func (s *Session) Fork(msgs []ai.Message) (*Session, error) {
	s.mu.Lock()
	child := NewSession(newID(), s.cwd, s.model)
	child.parent = s.id
	// Accounting carries over: the branch continues the same run's spend, and
	// resetting it would make a forked session look free.
	child.Usage = s.Usage
	s.mu.Unlock()

	child.mu.Lock()
	defer child.mu.Unlock()
	if err := child.ensureFile(); err != nil {
		return nil, err
	}
	for i := range msgs {
		m := msgs[i]
		if err := child.writeEntry(Entry{
			Kind: EntryMessage, Timestamp: time.Now().UnixMilli(), Message: &m,
		}); err != nil {
			_ = child.file.Close()
			return nil, err
		}
	}
	return child, nil
}

// Close flushes and closes the file.
func (s *Session) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.created {
		return nil
	}
	if err := s.writer.Flush(); err != nil {
		_ = s.file.Close()
		return err
	}
	return s.file.Close()
}

// SessionInfo summarizes a stored session for listings.
type SessionInfo struct {
	ID       string
	Path     string
	Cwd      string
	Model    string
	Started  time.Time
	Modified time.Time
	Messages int
	Preview  string
	// Parent is the session this one was forked from, empty if it was not.
	Parent string
	// Snapshots are the file captures still live after replaying rewinds.
	// Populated by LoadSession; listings do not need them and skip the work.
	Snapshots []FileSnapshot
}

// ListSessions returns stored sessions for a workspace, newest first.
func ListSessions(cwd string) ([]SessionInfo, error) {
	dir := filepath.Join(sessionsRoot(), slugForCwd(cwd))
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}

	var out []SessionInfo
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".jsonl") {
			continue
		}
		path := filepath.Join(dir, e.Name())
		info, err := summarize(path)
		if err != nil {
			continue // a corrupt file should not break the listing
		}
		out = append(out, info)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Modified.After(out[j].Modified) })
	return out, nil
}

func summarize(path string) (SessionInfo, error) {
	f, err := os.Open(path)
	if err != nil {
		return SessionInfo{}, err
	}
	defer func() { _ = f.Close() }()

	stat, err := f.Stat()
	if err != nil {
		return SessionInfo{}, err
	}

	info := SessionInfo{Path: path, Modified: stat.ModTime()}
	// Previews are kept with the message index they came from so a rewind can
	// drop the ones it discarded. A listing that still advertises a prompt the
	// session no longer contains sends you into the wrong transcript.
	type preview struct {
		at   int
		text string
	}
	var previews []preview
	count := 0

	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 16<<20)
	for sc.Scan() {
		var e Entry
		if json.Unmarshal(sc.Bytes(), &e) != nil {
			continue
		}
		switch e.Kind {
		case EntryHeader:
			info.ID, info.Cwd, info.Model = e.SessionID, e.Cwd, e.Model
			info.Parent = e.Parent
			info.Started = time.UnixMilli(e.Timestamp)
		case EntryMessage:
			if e.Message != nil && e.Message.Role == ai.RoleUser {
				previews = append(previews, preview{at: count, text: firstLine(e.Message.Text(), 80)})
			}
			count++
		case EntryRewind:
			if e.Retained < count {
				count = e.Retained
				for len(previews) > 0 && previews[len(previews)-1].at >= count {
					previews = previews[:len(previews)-1]
				}
			}
		}
	}
	info.Messages = count
	if len(previews) > 0 {
		info.Preview = previews[0].text
	}
	return info, sc.Err()
}

// LoadSession reads a transcript back for resuming.
//
// Streaming partials are never written, so every message on disk is final.
// Errored assistant turns ARE kept: they are part of what happened, and the
// provider payload encoder drops them at send time rather than here.
//
// Rewind markers are replayed in file order, which is what makes the record and
// the view separable: the file holds every message ever written, and this
// returns the view the model was left with.
func LoadSession(path string) ([]ai.Message, SessionInfo, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, SessionInfo{}, err
	}
	defer func() { _ = f.Close() }()

	var msgs []ai.Message
	info := SessionInfo{Path: path}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 16<<20)
	for sc.Scan() {
		var e Entry
		if json.Unmarshal(sc.Bytes(), &e) != nil {
			continue
		}
		switch e.Kind {
		case EntryHeader:
			info.ID, info.Cwd, info.Model = e.SessionID, e.Cwd, e.Model
			info.Parent = e.Parent
			info.Started = time.UnixMilli(e.Timestamp)
		case EntryMessage:
			if e.Message != nil {
				msgs = append(msgs, *e.Message)
			}
		case EntryModelChange:
			info.Model = e.Model
		case EntrySnapshot:
			if e.Snapshot != nil {
				info.Snapshots = append(info.Snapshots, *e.Snapshot)
			}
		case EntryRewind:
			// Guard against a file that claims to retain more than it holds,
			// which a truncated write or a newer format version could produce.
			if e.Retained < len(msgs) {
				msgs = msgs[:e.Retained]
			}
			// Snapshots replay by the same rule as messages: a rewind consumed
			// everything at or after the cut, so a resume must not offer to
			// revert changes an earlier run already reverted.
			kept := info.Snapshots[:0]
			for _, s := range info.Snapshots {
				if s.At < e.Retained {
					kept = append(kept, s)
				}
			}
			info.Snapshots = kept
		}
	}
	if err := sc.Err(); err != nil {
		return nil, info, err
	}
	info.Messages = len(msgs)
	return msgs, info, nil
}

// ResumeSession reopens an existing session file for appending.
//
// The info comes back too, because a resumed session needs more than its
// messages: the file snapshots it replayed are what let a later rewind undo
// work done in an earlier run.
func ResumeSession(path string) (*Session, []ai.Message, SessionInfo, error) {
	msgs, info, err := LoadSession(path)
	if err != nil {
		return nil, nil, SessionInfo{}, err
	}
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return nil, nil, SessionInfo{}, err
	}
	s := &Session{
		id: info.ID, path: path, cwd: info.Cwd, model: info.Model,
		parent:  info.Parent,
		created: true, file: f, writer: bufio.NewWriter(f),
	}
	// Usage on resumed assistant messages is deliberately NOT re-accumulated:
	// those tokens were billed in the earlier run, and counting them again
	// would make a resumed session look like it cost double.
	return s, msgs, info, nil
}

func firstLine(s string, max int) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		s = s[:i]
	}
	s = strings.TrimSpace(s)
	if len(s) > max {
		return s[:max] + "…"
	}
	return s
}
