package harness

import (
	"os"
	"path/filepath"
	"strings"
)

// InstructionFile is one project instruction document.
type InstructionFile struct {
	Path    string
	Content string
}

// instructionNames are the filenames checked in each directory, in priority
// order. The first one found in a directory wins, so a project can adopt
// deepseek-pi without renaming the file its other agents already read.
var instructionNames = []string{
	"DEEPSEEK.md",
	"AGENTS.md",
	"CLAUDE.md",
}

// maxInstructionBytes bounds a single instruction file. These live in the
// cached prefix on every turn, so an accidentally huge AGENTS.md is a
// permanent per-turn tax rather than a one-off cost.
const maxInstructionBytes = 32 * 1024

// DiscoverInstructions collects project instructions for a workspace.
//
// Order is outermost directory first, ending with the workspace root, so the
// most specific instructions come last and read as refinements of the general
// ones. The global file (if any) leads.
//
// Discovery stops at a .git directory or the filesystem root, whichever comes
// first: walking past a repository boundary picks up unrelated projects that
// happen to be ancestors on disk.
func DiscoverInstructions(root string) []InstructionFile {
	var files []InstructionFile

	if global := globalInstructionPath(); global != "" {
		if f, ok := readInstruction(global); ok {
			files = append(files, f)
		}
	}

	abs, err := filepath.Abs(root)
	if err != nil {
		return files
	}

	// Collect ancestors up to the repo boundary, then reverse so the walk reads
	// root-down.
	var dirs []string
	dir := abs
	for {
		dirs = append(dirs, dir)
		if isRepoRoot(dir) {
			break
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	for i, j := 0, len(dirs)-1; i < j; i, j = i+1, j-1 {
		dirs[i], dirs[j] = dirs[j], dirs[i]
	}

	seen := make(map[string]bool, len(files))
	for _, f := range files {
		seen[f.Path] = true
	}
	for _, d := range dirs {
		for _, name := range instructionNames {
			p := filepath.Join(d, name)
			if seen[p] {
				break
			}
			if f, ok := readInstruction(p); ok {
				files = append(files, f)
				seen[p] = true
				break // first match per directory
			}
		}
	}
	return files
}

func isRepoRoot(dir string) bool {
	info, err := os.Stat(filepath.Join(dir, ".git"))
	return err == nil && (info.IsDir() || info.Mode().IsRegular())
}

func readInstruction(path string) (InstructionFile, bool) {
	data, err := os.ReadFile(path)
	if err != nil {
		return InstructionFile{}, false
	}
	content := strings.TrimSpace(string(data))
	if content == "" {
		return InstructionFile{}, false
	}
	if len(content) > maxInstructionBytes {
		content = content[:maxInstructionBytes] +
			"\n\n[truncated: instruction file exceeds " + itoa(maxInstructionBytes) + " bytes]"
	}
	return InstructionFile{Path: path, Content: content}, true
}

func globalInstructionPath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	for _, p := range []string{
		filepath.Join(home, ".deepseek-pi", "AGENTS.md"),
		filepath.Join(home, ".agents", "AGENTS.md"),
	} {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return ""
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}
