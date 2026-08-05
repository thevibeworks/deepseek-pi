package harness

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"

	"github.com/thevibeworks/deepseek-pi/agent"
	"github.com/thevibeworks/deepseek-pi/tools"
)

// MaxSnapshotBytes bounds what a checkpoint will store for one file.
//
// The blobs are for getting a source tree back, not for backing up build
// output. Something enormous is far more likely to be a database, a binary or a
// log than a file worth rewinding, and a rewind that quietly triples disk usage
// is a worse failure than one that says it cannot restore a 40MB file.
const MaxSnapshotBytes = 8 << 20

// FileSnapshot is one file's state around a single tool call.
//
// Both sides are recorded for different reasons. Before is what a rewind puts
// back. After is what proves nobody else has touched the file since — restoring
// over an edit the user made by hand would destroy work the conversation never
// knew about, which is worse than not restoring at all.
type FileSnapshot struct {
	// At is the number of messages in the transcript when this was taken, which
	// is what makes a snapshot comparable against a rewind's cut point.
	At int `json:"at"`
	// Path is workspace-relative.
	Path string `json:"path"`
	// Before is the blob hash of the prior content, meaningful only when the
	// file existed and was not skipped.
	Before string `json:"before,omitempty"`
	// After is the blob hash of the resulting content, empty if the call left
	// no file behind.
	After string `json:"after,omitempty"`
	// Existed records whether there was a file here before the call, which is
	// what tells a restore to delete rather than rewrite.
	Existed bool `json:"existed,omitempty"`
	// Skipped explains why this path cannot be restored. Recording the fact of
	// the change even when the content was not kept is the point: the report
	// can then say what it could not undo instead of omitting it.
	Skipped string `json:"skipped,omitempty"`
}

// BlobStore is content-addressed file storage shared by a workspace's sessions.
//
// Content addressing is what makes a fork cheap and a repeated edit cheap: the
// same bytes are stored once no matter how many turns or sessions refer to
// them, and no reference counting is needed to keep that true.
type BlobStore struct{ root string }

// NewBlobStore returns the store for a workspace.
func NewBlobStore(cwd string) *BlobStore {
	return &BlobStore{root: filepath.Join(sessionsRoot(), slugForCwd(cwd), "blobs")}
}

func (b *BlobStore) pathFor(hash string) string {
	// Two-character shard: a workspace can accumulate a lot of blobs, and some
	// filesystems degrade badly on a single directory with tens of thousands of
	// entries.
	return filepath.Join(b.root, hash[:2], hash[2:])
}

// Put stores data and returns its hash. Storing the same bytes twice is a
// no-op, so callers do not have to check first.
func (b *BlobStore) Put(data []byte) (string, error) {
	sum := sha256.Sum256(data)
	hash := hex.EncodeToString(sum[:])
	path := b.pathFor(hash)
	if _, err := os.Stat(path); err == nil {
		return hash, nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return "", err
	}
	// Write to a temp file and rename, so a crash mid-write cannot leave a
	// truncated blob under a hash that claims to describe complete content.
	tmp, err := os.CreateTemp(filepath.Dir(path), ".tmp-*")
	if err != nil {
		return "", err
	}
	defer func() { _ = os.Remove(tmp.Name()) }()
	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		return "", err
	}
	if err := tmp.Close(); err != nil {
		return "", err
	}
	return hash, os.Rename(tmp.Name(), path)
}

// Get returns stored content.
func (b *BlobStore) Get(hash string) ([]byte, error) { return os.ReadFile(b.pathFor(hash)) }

// Checkpointer captures what file tools change so a rewind can put it back.
//
// It hangs off the tool-call seam rather than living inside the tools, because
// the tools have no business knowing about sessions, and because the seam sees
// every call in one place — including calls that a later tool would overwrite.
type Checkpointer struct {
	ws    *tools.Workspace
	blobs *BlobStore

	mu sync.Mutex
	// pending correlates the before-capture with the after-capture, keyed on
	// tool call id since a batch runs in parallel.
	pending map[string]*FileSnapshot
	// taken is every snapshot this session has recorded, in order.
	taken []FileSnapshot
}

// NewCheckpointer builds a checkpointer for a workspace.
func NewCheckpointer(ws *tools.Workspace, blobs *BlobStore) *Checkpointer {
	return &Checkpointer{ws: ws, blobs: blobs, pending: map[string]*FileSnapshot{}}
}

// Restore adds prior snapshots, which is how a resumed session keeps the
// ability to rewind past work done in an earlier run.
func (c *Checkpointer) Restore(snaps []FileSnapshot) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.taken = append(c.taken, snaps...)
}

// Snapshots returns what has been recorded.
func (c *Checkpointer) Snapshots() []FileSnapshot {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]FileSnapshot(nil), c.taken...)
}

// checkpointedTools are the tools whose effects can be captured.
//
// bash is deliberately absent and cannot be added: a shell command may touch
// anything, and guessing at what it changed would produce a restore that is
// confidently wrong. The limit is reported rather than papered over.
var checkpointedTools = map[string]bool{"write": true, "edit": true}

// Before captures a file's state ahead of a call that may change it.
func (c *Checkpointer) Before(call agent.ToolCall, at int) {
	if !checkpointedTools[call.Name] {
		return
	}
	rel := argString(call.Arguments, pathKeys...)
	if rel == "" {
		return
	}
	abs, err := c.ws.Resolve(rel)
	if err != nil {
		return // out of the workspace; the tool will refuse it anyway
	}

	snap := &FileSnapshot{At: at, Path: c.ws.Rel(abs)}
	switch data, err := readCapped(abs); {
	case err == nil:
		snap.Existed = true
		if hash, err := c.blobs.Put(data); err == nil {
			snap.Before = hash
		} else {
			snap.Skipped = "the previous content could not be stored: " + err.Error()
		}
	case os.IsNotExist(err):
		// Not an error: the restore for a file the agent created is to delete
		// it, and Existed staying false is what says so.
	default:
		snap.Existed = true
		snap.Skipped = err.Error()
	}

	c.mu.Lock()
	c.pending[call.ID] = snap
	c.mu.Unlock()
}

// After completes the capture once the call has run.
func (c *Checkpointer) After(call agent.ToolCall, failed bool) *FileSnapshot {
	c.mu.Lock()
	snap := c.pending[call.ID]
	delete(c.pending, call.ID)
	c.mu.Unlock()
	if snap == nil {
		return nil
	}

	// A failed call may still have changed the file — a partial write, or an
	// edit that applied before a later one in the same call failed — so the
	// after-state is read either way rather than assumed unchanged.
	abs, err := c.ws.Resolve(snap.Path)
	if err == nil {
		if data, err := readCapped(abs); err == nil {
			if hash, perr := c.blobs.Put(data); perr == nil {
				snap.After = hash
			}
		}
	}
	if failed && snap.After == snap.Before && snap.Existed {
		return nil // nothing happened; recording it would only add noise
	}

	c.mu.Lock()
	c.taken = append(c.taken, *snap)
	c.mu.Unlock()
	return snap
}

func readCapped(path string) ([]byte, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	if info.IsDir() {
		return nil, fmt.Errorf("%s is a directory", path)
	}
	if info.Size() > MaxSnapshotBytes {
		return nil, fmt.Errorf("the file is %s, over the %s snapshot limit",
			formatSize(info.Size()), formatSize(MaxSnapshotBytes))
	}
	return os.ReadFile(path)
}

func formatSize(n int64) string {
	switch {
	case n >= 1<<20:
		return fmt.Sprintf("%.1fMB", float64(n)/(1<<20))
	case n >= 1<<10:
		return fmt.Sprintf("%.1fKB", float64(n)/(1<<10))
	}
	return fmt.Sprintf("%dB", n)
}

// FileRestore is what happened to one path during a rewind.
type FileRestore struct {
	Path string
	// Action is "reverted", "removed", or "kept".
	Action string
	// Why explains a kept file, which is the only case a user must read.
	Why string
}

// Restored reports the paths a rewind put back.
type Restored struct {
	Changes []FileRestore
}

// Reverted counts the paths that went back.
func (r Restored) Reverted() int {
	n := 0
	for _, c := range r.Changes {
		if c.Action != "kept" {
			n++
		}
	}
	return n
}

// Kept lists paths left alone, with the reason.
func (r Restored) Kept() []FileRestore {
	var out []FileRestore
	for _, c := range r.Changes {
		if c.Action == "kept" {
			out = append(out, c)
		}
	}
	return out
}

// Forget drops snapshots at or after cut without touching the workspace. It is
// what /clear uses: clearing is about the conversation, and silently reverting
// a day of accepted edits because the context was dropped would be a shock.
func (c *Checkpointer) Forget(cut int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	keep := c.taken[:0]
	for _, s := range c.taken {
		if s.At < cut {
			keep = append(keep, s)
		}
	}
	c.taken = keep
}

// RewindFiles puts the workspace back to how it stood before message index cut,
// and forgets the snapshots it consumed.
//
// Two rules decide each path. The state to restore is the EARLIEST snapshot at
// or after the cut, because that is what the file looked like before any of the
// discarded work touched it. Whether it is safe to restore is decided by the
// LATEST one: if what is on disk now is not what the agent last left there,
// something outside the conversation changed it and this leaves it alone. A
// rewind that silently overwrites a hand edit is a worse tool than one that
// cannot rewind at all.
func (c *Checkpointer) RewindFiles(cut int) Restored {
	c.mu.Lock()
	var affected, keep []FileSnapshot
	for _, s := range c.taken {
		if s.At >= cut {
			affected = append(affected, s)
		} else {
			keep = append(keep, s)
		}
	}
	c.taken = keep
	c.mu.Unlock()

	first := map[string]FileSnapshot{}
	last := map[string]FileSnapshot{}
	var order []string
	for _, s := range affected {
		if _, seen := first[s.Path]; !seen {
			first[s.Path] = s
			order = append(order, s.Path)
		}
		last[s.Path] = s
	}
	sort.Strings(order)

	var out Restored
	for _, path := range order {
		out.Changes = append(out.Changes, c.revert(first[path], last[path]))
	}
	return out
}

func (c *Checkpointer) revert(first, last FileSnapshot) FileRestore {
	kept := func(why string) FileRestore {
		return FileRestore{Path: first.Path, Action: "kept", Why: why}
	}
	if first.Skipped != "" {
		return kept(first.Skipped)
	}
	abs, err := c.ws.Resolve(first.Path)
	if err != nil {
		return kept(err.Error())
	}

	// The safety check: on-disk content has to be what the agent last wrote.
	current, err := readCapped(abs)
	switch {
	case err == nil:
		hash, perr := c.blobs.Put(current)
		if perr != nil {
			return kept("the current content could not be read: " + perr.Error())
		}
		if hash != last.After {
			return kept("it changed outside this conversation since the agent wrote it")
		}
	case os.IsNotExist(err):
		if last.After != "" {
			return kept("it was deleted outside this conversation")
		}
	default:
		return kept(err.Error())
	}

	if !first.Existed {
		if err := os.Remove(abs); err != nil && !os.IsNotExist(err) {
			return kept(err.Error())
		}
		return FileRestore{Path: first.Path, Action: "removed"}
	}
	data, err := c.blobs.Get(first.Before)
	if err != nil {
		return kept("the stored copy is missing: " + err.Error())
	}
	if err := os.WriteFile(abs, data, 0o644); err != nil {
		return kept(err.Error())
	}
	return FileRestore{Path: first.Path, Action: "reverted"}
}

// PruneBlobs deletes stored content no session in the workspace still refers
// to. Blobs are written on every file change and nothing else removes them.
func PruneBlobs(cwd string) (removed int, freed int64, err error) {
	live := map[string]bool{}
	sessions, err := ListSessions(cwd)
	if err != nil {
		return 0, 0, err
	}
	for _, s := range sessions {
		snaps, err := loadSnapshots(s.Path)
		if err != nil {
			// A session that cannot be read might still reference blobs, so
			// deleting on the strength of a partial scan is not safe.
			return 0, 0, fmt.Errorf("reading %s: %w", s.Path, err)
		}
		for _, snap := range snaps {
			live[snap.Before], live[snap.After] = true, true
		}
	}

	root := NewBlobStore(cwd).root
	err = filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return nil // a missing blob dir just means nothing to prune
		}
		hash := filepath.Base(filepath.Dir(path)) + info.Name()
		if live[hash] {
			return nil
		}
		if rmErr := os.Remove(path); rmErr == nil {
			removed++
			freed += info.Size()
		}
		return nil
	})
	if os.IsNotExist(err) {
		err = nil
	}
	return removed, freed, err
}

// loadSnapshots reads just the snapshot entries a session still refers to.
func loadSnapshots(path string) ([]FileSnapshot, error) {
	_, info, err := LoadSession(path)
	if err != nil {
		return nil, err
	}
	return info.Snapshots, nil
}
