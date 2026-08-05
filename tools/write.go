package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/thevibeworks/deepseek-pi/agent"
	"github.com/thevibeworks/deepseek-pi/ai"
)

// WriteArgs are the write tool's parameters.
type WriteArgs struct {
	Path    string `json:"path"`
	Content string `json:"content"`
}

// Write builds the file-writing tool.
//
// It overwrites unconditionally and creates parent directories, because a
// write tool that refuses on an existing file just teaches the model to delete
// first. The guardrail that matters is the guideline telling it to prefer edit
// for existing files, plus the workspace scope.
func Write(w *Workspace) agent.Tool {
	return agent.Tool{
		Name:        "write",
		Label:       "Write",
		Description: "Write a file, creating parent directories as needed. Overwrites an existing file completely.",
		Parameters: ai.Object(map[string]*ai.Schema{
			"path":    ai.Str("Path to the file to write."),
			"content": ai.Str("Full content of the file."),
		}, "path", "content"),
		PromptSnippet: "write(path, content) — create or overwrite a file",
		PromptGuidelines: "Use write only for new files or a full rewrite; use edit to change part of an existing file. " +
			"Never write a file you have not read when it already exists.",
		ExecutionMode: agent.ModeSequential,
		PrepareArguments: agent.HealArguments(map[string]string{
			"file_path": "path", "filename": "path", "filepath": "path", "file": "path",
			"text": "content", "data": "content", "contents": "content",
		}),
		Execute: func(_ context.Context, call agent.ToolCall, _ agent.UpdateFunc) (agent.ToolResult, error) {
			var args WriteArgs
			if err := json.Unmarshal(call.Arguments, &args); err != nil {
				return agent.ToolResult{}, fmt.Errorf("invalid arguments: %w", err)
			}
			abs, err := w.Resolve(args.Path)
			if err != nil {
				return agent.ToolResult{}, err
			}

			var res agent.ToolResult
			err = w.WithFileLock(abs, func() error {
				existed := false
				perm := os.FileMode(0o644)
				if info, err := os.Stat(abs); err == nil {
					if info.IsDir() {
						return fmt.Errorf("%s is a directory", w.Rel(abs))
					}
					existed = true
					perm = info.Mode().Perm()
				}
				if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
					return fmt.Errorf("creating parent directory: %w", err)
				}
				if err := writeFileAtomic(abs, []byte(args.Content), perm); err != nil {
					return err
				}
				verb := "Created"
				if existed {
					verb = "Overwrote"
				}
				lines := len(splitLines(args.Content))
				res = agent.ToolResult{
					Content: []ai.Content{ai.TextContent(fmt.Sprintf(
						"%s %s (%d lines, %s)", verb, w.Rel(abs), lines, FormatSize(len(args.Content))))},
					Details: map[string]any{
						"path": w.Rel(abs), "bytes": len(args.Content),
						"lines": lines, "created": !existed,
					},
				}
				return nil
			})
			if err != nil {
				return agent.ToolResult{}, err
			}
			return res, nil
		},
	}
}

// writeFileAtomic writes via a temp file and rename, so an interrupted write
// leaves the original intact rather than a half-written file. An agent that
// crashes mid-edit should cost a retry, not a corrupted source file.
func writeFileAtomic(path string, data []byte, perm os.FileMode) error {
	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, "."+filepath.Base(path)+".tmp-*")
	if err != nil {
		// A read-only or unusual directory: fall back to a direct write rather
		// than failing an otherwise valid edit.
		return os.WriteFile(path, data, perm)
	}
	tmpName := tmp.Name()
	defer func() { _ = os.Remove(tmpName) }()

	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Chmod(perm); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpName, path)
}
