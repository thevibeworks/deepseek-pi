package harness

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/thevibeworks/deepseek-pi/agent"
	"github.com/thevibeworks/deepseek-pi/tools"
)

func testCheckpointer(t *testing.T) (*Checkpointer, string) {
	t.Helper()
	t.Setenv("DEEPSEEK_PI_HOME", t.TempDir())
	dir := t.TempDir()
	ws, err := tools.NewWorkspace(dir)
	if err != nil {
		t.Fatalf("workspace: %v", err)
	}
	return NewCheckpointer(ws, NewBlobStore(dir)), ws.Root
}

func writeCall(id, path string) agent.ToolCall {
	args, _ := json.Marshal(map[string]string{"path": path, "content": "x"})
	return agent.ToolCall{ID: id, Name: "write", Arguments: args}
}

// capture drives one tool call end to end: snapshot, apply, snapshot.
func capture(t *testing.T, c *Checkpointer, root, id, name, content string, at int) {
	t.Helper()
	call := writeCall(id, name)
	c.Before(call, at)
	if content == "" {
		if err := os.Remove(filepath.Join(root, name)); err != nil && !os.IsNotExist(err) {
			t.Fatal(err)
		}
	} else if err := os.WriteFile(filepath.Join(root, name), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	c.After(call, false)
}

func read(t *testing.T, path string) string {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading %s: %v", path, err)
	}
	return string(data)
}

func TestRewindRevertsAnEditedFile(t *testing.T) {
	c, root := testCheckpointer(t)
	file := filepath.Join(root, "a.go")
	if err := os.WriteFile(file, []byte("original"), 0o644); err != nil {
		t.Fatal(err)
	}

	capture(t, c, root, "w1", "a.go", "modified", 4)

	got := c.RewindFiles(4)
	if got.Reverted() != 1 {
		t.Fatalf("reverted %d files, want 1: %+v", got.Reverted(), got.Changes)
	}
	if body := read(t, file); body != "original" {
		t.Errorf("file content is %q, want the pre-turn %q", body, "original")
	}
}

func TestRewindRemovesAFileTheAgentCreated(t *testing.T) {
	// The inverse of a revert: if there was nothing there before, putting it
	// back means the file should not exist.
	c, root := testCheckpointer(t)
	capture(t, c, root, "w1", "new.go", "created", 4)

	got := c.RewindFiles(4)
	if got.Reverted() != 1 {
		t.Fatalf("reverted %d, want 1: %+v", got.Reverted(), got.Changes)
	}
	if _, err := os.Stat(filepath.Join(root, "new.go")); !os.IsNotExist(err) {
		t.Error("a file the agent created survived the rewind")
	}
	if got.Changes[0].Action != "removed" {
		t.Errorf("action = %q, want removed", got.Changes[0].Action)
	}
}

func TestRewindRestoresTheOldestStateNotTheNewest(t *testing.T) {
	// Three writes to one file inside the discarded range. Going back means
	// going to how it looked before the FIRST of them, not before the last.
	c, root := testCheckpointer(t)
	file := filepath.Join(root, "a.go")
	if err := os.WriteFile(file, []byte("v0"), 0o644); err != nil {
		t.Fatal(err)
	}
	capture(t, c, root, "w1", "a.go", "v1", 4)
	capture(t, c, root, "w2", "a.go", "v2", 6)
	capture(t, c, root, "w3", "a.go", "v3", 8)

	c.RewindFiles(4)
	if body := read(t, file); body != "v0" {
		t.Errorf("content is %q, want v0 — the state before the first discarded write", body)
	}
}

func TestRewindLeavesEarlierTurnsAlone(t *testing.T) {
	c, root := testCheckpointer(t)
	kept := filepath.Join(root, "kept.go")
	if err := os.WriteFile(kept, []byte("v0"), 0o644); err != nil {
		t.Fatal(err)
	}
	capture(t, c, root, "w1", "kept.go", "v1", 2)     // turn 1, retained
	capture(t, c, root, "w2", "dropped.go", "new", 6) // turn 2, discarded

	c.RewindFiles(4)
	if body := read(t, kept); body != "v1" {
		t.Errorf("a retained turn's change was reverted: %q", body)
	}
	if _, err := os.Stat(filepath.Join(root, "dropped.go")); !os.IsNotExist(err) {
		t.Error("the discarded turn's file survived")
	}
}

// TestRewindWillNotClobberAHandEdit is the safety property. Restoring over work
// the conversation never knew about is worse than not restoring at all.
func TestRewindWillNotClobberAHandEdit(t *testing.T) {
	c, root := testCheckpointer(t)
	file := filepath.Join(root, "a.go")
	if err := os.WriteFile(file, []byte("original"), 0o644); err != nil {
		t.Fatal(err)
	}
	capture(t, c, root, "w1", "a.go", "agent wrote this", 4)

	// The user edits it themselves afterwards.
	if err := os.WriteFile(file, []byte("I fixed it by hand"), 0o644); err != nil {
		t.Fatal(err)
	}

	got := c.RewindFiles(4)
	if got.Reverted() != 0 {
		t.Errorf("a hand-edited file was reverted anyway: %+v", got.Changes)
	}
	if body := read(t, file); body != "I fixed it by hand" {
		t.Errorf("hand edit destroyed; file is now %q", body)
	}
	kept := got.Kept()
	if len(kept) != 1 || !strings.Contains(kept[0].Why, "outside this conversation") {
		t.Errorf("the refusal does not explain itself: %+v", kept)
	}
}

func TestRewindWillNotResurrectAFileDeletedByHand(t *testing.T) {
	c, root := testCheckpointer(t)
	file := filepath.Join(root, "a.go")
	if err := os.WriteFile(file, []byte("original"), 0o644); err != nil {
		t.Fatal(err)
	}
	capture(t, c, root, "w1", "a.go", "agent wrote this", 4)
	if err := os.Remove(file); err != nil {
		t.Fatal(err)
	}

	got := c.RewindFiles(4)
	if got.Reverted() != 0 {
		t.Errorf("a file deleted outside the conversation was recreated: %+v", got.Changes)
	}
	if _, err := os.Stat(file); !os.IsNotExist(err) {
		t.Error("the deleted file came back")
	}
}

func TestSnapshotsAreConsumedByARewind(t *testing.T) {
	// A second rewind must not re-apply the first one's work, or rewinding
	// twice would revert a file to a state nobody asked for.
	c, root := testCheckpointer(t)
	capture(t, c, root, "w1", "a.go", "created", 4)

	if n := c.RewindFiles(4).Reverted(); n != 1 {
		t.Fatalf("first rewind reverted %d, want 1", n)
	}
	if n := c.RewindFiles(4).Reverted(); n != 0 {
		t.Errorf("the second rewind found %d files to revert, want 0", n)
	}
	if len(c.Snapshots()) != 0 {
		t.Errorf("consumed snapshots were kept: %+v", c.Snapshots())
	}
}

func TestOversizedFilesAreReportedNotSilentlyDropped(t *testing.T) {
	c, root := testCheckpointer(t)
	file := filepath.Join(root, "big.bin")
	if err := os.WriteFile(file, make([]byte, MaxSnapshotBytes+1), 0o644); err != nil {
		t.Fatal(err)
	}
	capture(t, c, root, "w1", "big.bin", "small now", 4)

	got := c.RewindFiles(4)
	kept := got.Kept()
	if len(kept) != 1 {
		t.Fatalf("an unrestorable file was not reported: %+v", got.Changes)
	}
	if !strings.Contains(kept[0].Why, "snapshot limit") {
		t.Errorf("the reason does not name the limit: %q", kept[0].Why)
	}
	if body := read(t, file); body != "small now" {
		t.Errorf("an unrestorable file was modified anyway: %q", body)
	}
}

func TestBlobStoreDeduplicates(t *testing.T) {
	t.Setenv("DEEPSEEK_PI_HOME", t.TempDir())
	b := NewBlobStore(t.TempDir())

	h1, err := b.Put([]byte("same bytes"))
	if err != nil {
		t.Fatal(err)
	}
	h2, err := b.Put([]byte("same bytes"))
	if err != nil {
		t.Fatal(err)
	}
	if h1 != h2 {
		t.Errorf("identical content stored under two hashes: %s / %s", h1, h2)
	}
	got, err := b.Get(h1)
	if err != nil || string(got) != "same bytes" {
		t.Errorf("round trip failed: %q, %v", got, err)
	}
}

func TestPruneKeepsWhatSessionsStillReference(t *testing.T) {
	h := testHarness(t)
	cwd := h.Workspace.Root
	file := filepath.Join(cwd, "a.go")
	if err := os.WriteFile(file, []byte("original"), 0o644); err != nil {
		t.Fatal(err)
	}

	call := writeCall("w1", "a.go")
	h.Checkpoints.Before(call, 4)
	if err := os.WriteFile(file, []byte("modified"), 0o644); err != nil {
		t.Fatal(err)
	}
	snap := h.Checkpoints.After(call, false)
	if snap == nil {
		t.Fatal("no snapshot recorded")
	}
	if err := h.Session.RecordSnapshot(*snap); err != nil {
		t.Fatalf("RecordSnapshot: %v", err)
	}

	// An orphan nothing refers to.
	orphan, err := NewBlobStore(cwd).Put([]byte("unreferenced"))
	if err != nil {
		t.Fatal(err)
	}
	if err := h.Session.Close(); err != nil {
		t.Fatal(err)
	}

	removed, _, err := PruneBlobs(cwd)
	if err != nil {
		t.Fatalf("PruneBlobs: %v", err)
	}
	if removed != 1 {
		t.Errorf("removed %d blobs, want just the orphan", removed)
	}
	if _, err := NewBlobStore(cwd).Get(orphan); !os.IsNotExist(err) {
		t.Error("the orphan survived")
	}
	// The referenced one has to survive, or a resumed session cannot rewind.
	if _, err := NewBlobStore(cwd).Get(snap.Before); err != nil {
		t.Errorf("a referenced blob was pruned: %v", err)
	}
}
