package tools

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// Workspace scopes file operations to a directory and serializes writes to the
// same file.
//
// The per-file lock is not optional decoration. Tool batches run in parallel by
// default, and a model that emits two edits to one file in a single batch is
// common, not exotic. Without a lock keyed on the RESOLVED path, those two
// edits read the same original content and the second silently discards the
// first. Keying on the resolved path (symlinks followed) is what makes two
// different spellings of the same file take the same lock.
type Workspace struct {
	// Root is the directory tool paths resolve against.
	Root string
	// AllowOutside permits absolute paths outside Root. Off by default: a
	// coding agent that can write anywhere is a different security posture than
	// one scoped to a project, and that should be a deliberate choice.
	AllowOutside bool

	mu    sync.Mutex
	locks map[string]*sync.Mutex
}

// NewWorkspace creates a workspace rooted at dir.
func NewWorkspace(dir string) (*Workspace, error) {
	abs, err := filepath.Abs(dir)
	if err != nil {
		return nil, fmt.Errorf("resolving workspace root: %w", err)
	}
	info, err := os.Stat(abs)
	if err != nil {
		return nil, fmt.Errorf("workspace root %s: %w", abs, err)
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("workspace root %s is not a directory", abs)
	}
	return &Workspace{Root: abs, locks: make(map[string]*sync.Mutex)}, nil
}

// Resolve turns a tool-supplied path into an absolute path, enforcing scope.
func (w *Workspace) Resolve(path string) (string, error) {
	if strings.TrimSpace(path) == "" {
		return "", fmt.Errorf("path must not be empty")
	}
	if strings.HasPrefix(path, "~") {
		home, err := os.UserHomeDir()
		if err == nil {
			path = filepath.Join(home, strings.TrimPrefix(path, "~"))
		}
	}
	abs := path
	if !filepath.IsAbs(abs) {
		abs = filepath.Join(w.Root, abs)
	}
	abs = filepath.Clean(abs)

	if w.AllowOutside {
		return abs, nil
	}
	// Compare against the resolved root so a symlinked workspace does not
	// reject every path inside itself.
	realRoot, err := filepath.EvalSymlinks(w.Root)
	if err != nil {
		realRoot = w.Root
	}
	probe := abs
	if resolved, err := filepath.EvalSymlinks(abs); err == nil {
		probe = resolved
	}
	rel, err := filepath.Rel(realRoot, probe)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("path %s is outside the workspace (%s)", path, w.Root)
	}
	return abs, nil
}

// Rel renders a path relative to the workspace root for display. Model-facing
// output uses relative paths so the transcript stays portable and short.
func (w *Workspace) Rel(abs string) string {
	if rel, err := filepath.Rel(w.Root, abs); err == nil && !strings.HasPrefix(rel, "..") {
		return rel
	}
	return abs
}

// lockFor returns the mutex guarding a path, keyed on its resolved identity so
// symlinks and relative spellings of one file share a lock.
func (w *Workspace) lockFor(abs string) *sync.Mutex {
	key := abs
	if resolved, err := filepath.EvalSymlinks(abs); err == nil {
		key = resolved
	} else {
		// The file may not exist yet (a write). Resolve the parent instead so
		// two creates of the same new path still serialize.
		if parent, err := filepath.EvalSymlinks(filepath.Dir(abs)); err == nil {
			key = filepath.Join(parent, filepath.Base(abs))
		}
	}

	w.mu.Lock()
	defer w.mu.Unlock()
	if w.locks == nil {
		w.locks = make(map[string]*sync.Mutex)
	}
	m, ok := w.locks[key]
	if !ok {
		m = &sync.Mutex{}
		w.locks[key] = m
	}
	return m
}

// WithFileLock runs fn holding the per-file lock.
func (w *Workspace) WithFileLock(abs string, fn func() error) error {
	m := w.lockFor(abs)
	m.Lock()
	defer m.Unlock()
	return fn()
}
