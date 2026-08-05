package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/thevibeworks/deepseek-pi/agent"
	"github.com/thevibeworks/deepseek-pi/ai"
)

// EditArgs are the edit tool's parameters.
type EditArgs struct {
	Path  string     `json:"path"`
	Edits []TextEdit `json:"edits"`
}

// Edit builds the file-editing tool.
//
// The schema is multi-edit native. One call carrying five edits is one round
// trip and one cache-friendly turn; five calls carrying one edit each are five
// of everything, and they interleave badly with parallel batches.
func Edit(w *Workspace) agent.Tool {
	return agent.Tool{
		Name:  "edit",
		Label: "Edit",
		Description: "Replace exact text in a file. Supply every edit for one file in a single call.\n" +
			"Each oldText must appear exactly once in the file; include surrounding lines to make it unique.\n" +
			"All edits are matched against the original file content, so they cannot interfere with each other.",
		Parameters: ai.Object(map[string]*ai.Schema{
			"path": ai.Str("Path to the file to edit."),
			"edits": ai.Array("The replacements to apply.", ai.Object(map[string]*ai.Schema{
				"oldText": ai.Str("Exact text to replace, unique within the file."),
				"newText": ai.Str("Replacement text. Use an empty string to delete."),
			}, "oldText", "newText")),
		}, "path", "edits"),
		PromptSnippet: "edit(path, edits[]) — replace exact text in a file",
		PromptGuidelines: "Read a file before editing it. Batch every edit for one file into a single edit call. " +
			"Keep oldText as small as possible while still unique, and preserve the file's existing indentation exactly.",
		// Sequential: this tool mutates the workspace. The per-file lock makes
		// concurrent edits to one file safe, but a batch that edits a file
		// another tool is reading still produces confusing interleavings, so
		// the whole batch serializes when an edit is present.
		ExecutionMode:    agent.ModeSequential,
		PrepareArguments: healEditArgs,
		Execute: func(_ context.Context, call agent.ToolCall, _ agent.UpdateFunc) (agent.ToolResult, error) {
			var args EditArgs
			if err := json.Unmarshal(call.Arguments, &args); err != nil {
				return agent.ToolResult{}, fmt.Errorf("invalid arguments: %w", err)
			}
			if len(args.Edits) == 0 {
				return agent.ToolResult{}, fmt.Errorf("edits must contain at least one edit")
			}
			abs, err := w.Resolve(args.Path)
			if err != nil {
				return agent.ToolResult{}, err
			}

			var result agent.ToolResult
			err = w.WithFileLock(abs, func() error {
				raw, err := os.ReadFile(abs)
				if err != nil {
					if os.IsNotExist(err) {
						return fmt.Errorf("file not found: %s. Use write to create it", w.Rel(abs))
					}
					return err
				}
				info, statErr := os.Stat(abs)
				if statErr != nil {
					return statErr
				}

				original := string(raw)
				bom, body := StripBOM(original)
				ending := DetectLineEnding(body)
				lfBody := NormalizeToLF(body)

				updated, err := ApplyEdits(lfBody, args.Edits, w.Rel(abs))
				if err != nil {
					return err
				}

				out := bom + RestoreLineEndings(updated, ending)
				if err := writeFileAtomic(abs, []byte(out), info.Mode().Perm()); err != nil {
					return err
				}

				diff, changed := renderDiff(lfBody, updated)
				result = agent.ToolResult{
					Content: []ai.Content{ai.TextContent(fmt.Sprintf(
						"Applied %d edit(s) to %s (%d line(s) changed).\n%s",
						len(args.Edits), w.Rel(abs), changed, diff))},
					Details: map[string]any{
						"path": w.Rel(abs), "edits": len(args.Edits), "linesChanged": changed,
					},
				}
				return nil
			})
			if err != nil {
				return agent.ToolResult{}, err
			}
			return result, nil
		},
	}
}

// healEditArgs repairs the two shapes models actually emit instead of the
// documented one: the whole argument object as a JSON string, and a legacy
// single-edit form with oldText/newText at the top level.
func healEditArgs(raw json.RawMessage) json.RawMessage {
	raw = agent.HealArguments(map[string]string{
		"file_path": "path", "filename": "path", "filepath": "path", "file": "path",
	})(raw)

	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		return raw
	}

	// edits delivered as a JSON string rather than an array.
	if v, ok := obj["edits"]; ok {
		var s string
		if json.Unmarshal(v, &s) == nil && json.Valid([]byte(s)) {
			obj["edits"] = json.RawMessage(s)
		}
	}

	// Legacy single-edit form: {path, oldText, newText}.
	if _, hasEdits := obj["edits"]; !hasEdits {
		oldText, hasOld := obj["oldText"]
		newText, hasNew := obj["newText"]
		if !hasOld {
			oldText, hasOld = obj["old_string"]
		}
		if !hasNew {
			newText, hasNew = obj["new_string"]
		}
		if hasOld && hasNew {
			one, err := json.Marshal([]map[string]json.RawMessage{
				{"oldText": oldText, "newText": newText},
			})
			if err == nil {
				obj["edits"] = one
				delete(obj, "oldText")
				delete(obj, "newText")
				delete(obj, "old_string")
				delete(obj, "new_string")
			}
		}
	}

	out, err := json.Marshal(obj)
	if err != nil {
		return raw
	}
	return out
}

// renderDiff produces a compact changed-region view and the number of changed
// lines. It trims the common prefix and suffix rather than running a full diff
// algorithm: edits are localized by construction, so that is enough to show
// what happened without pulling in a diff dependency.
func renderDiff(before, after string) (string, int) {
	oldLines := strings.Split(before, "\n")
	newLines := strings.Split(after, "\n")

	prefix := 0
	for prefix < len(oldLines) && prefix < len(newLines) && oldLines[prefix] == newLines[prefix] {
		prefix++
	}
	suffix := 0
	for suffix < len(oldLines)-prefix && suffix < len(newLines)-prefix &&
		oldLines[len(oldLines)-1-suffix] == newLines[len(newLines)-1-suffix] {
		suffix++
	}

	removed := oldLines[prefix : len(oldLines)-suffix]
	added := newLines[prefix : len(newLines)-suffix]

	const maxShown = 40
	var b strings.Builder
	shown := 0
	for _, l := range removed {
		if shown >= maxShown {
			break
		}
		text, _ := ClipLine(l, MaxLineLength)
		fmt.Fprintf(&b, "-%d %s\n", prefix+1, text)
		shown++
	}
	for i, l := range added {
		if shown >= maxShown {
			break
		}
		text, _ := ClipLine(l, MaxLineLength)
		fmt.Fprintf(&b, "+%d %s\n", prefix+1+i, text)
		shown++
	}
	total := len(removed) + len(added)
	if total > shown {
		fmt.Fprintf(&b, "... %d more changed line(s)\n", total-shown)
	}
	changed := len(added)
	if len(removed) > changed {
		changed = len(removed)
	}
	return strings.TrimRight(b.String(), "\n"), changed
}
