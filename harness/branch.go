package harness

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/thevibeworks/deepseek-pi/ai"
)

// Turn is one user prompt and everything the agent did in response.
//
// Turns exist because a transcript cannot be cut just anywhere. A tool call and
// its result are a matched pair the provider rejects if split, so the only
// unambiguously complete boundary is the start of a user message. That is the
// same rule findCutPoint applies to compaction; a turn is that boundary made
// addressable, so a user can name one.
type Turn struct {
	// Number is the 1-based index shown to the user.
	Number int
	// Start is the index of the user message that opened the turn.
	Start int
	// End is one past the turn's last message.
	End int
	// Prompt is the opening message, trimmed for display.
	Prompt string
	// Synthetic marks a turn the harness inserted rather than the user, which
	// today means a compaction summary. It is a real branch point, but showing
	// it as a prompt someone typed would be a lie.
	Synthetic bool
	// Wrote lists workspace paths the turn's tools modified.
	Wrote []string
	// Ran lists the turn's shell commands that were not read-only.
	Ran []string
	// Delegated lists sub-agent roles the turn spawned that could change
	// things. A child's tool calls never appear in this transcript — the parent
	// sees one task call and a report — so without this a turn that delegated
	// all its writing would read as having changed nothing.
	Delegated []string
}

// Changed reports whether the turn touched anything outside the transcript.
//
// This is the number that matters when rewinding: the conversation goes back
// and the filesystem does not, so a turn that wrote a file leaves that file
// behind after it is dropped from the context.
func (t Turn) Changed() bool {
	return len(t.Wrote) > 0 || len(t.Ran) > 0 || len(t.Delegated) > 0
}

// Turns segments a transcript into addressable branch points.
func Turns(msgs []ai.Message) []Turn {
	var turns []Turn
	for i, m := range msgs {
		if m.Role != ai.RoleUser {
			continue
		}
		if n := len(turns); n > 0 {
			turns[n-1].End = i
		}
		text := m.Text()
		synthetic := strings.HasPrefix(text, summaryPreamble)
		turns = append(turns, Turn{
			Number:    len(turns) + 1,
			Start:     i,
			End:       len(msgs),
			Prompt:    firstLine(strings.TrimPrefix(text, summaryPreamble), 68),
			Synthetic: synthetic,
		})
	}
	for i := range turns {
		changes(&turns[i], msgs[turns[i].Start:turns[i].End])
	}
	return turns
}

// pathKeys are the argument names a file tool may have been called with.
//
// The transcript stores the model's RAW arguments: the loop heals aliases into
// a local copy before executing, so what lands on disk is whatever the model
// actually emitted. Reading only "path" therefore misses every call that said
// file_path, which is common enough that tools.Write and tools.Edit carry a
// heal map for it. Keep this in step with those maps.
var pathKeys = []string{"path", "file_path", "filepath", "filename", "file"}

// changes reports what a stretch of transcript did to the machine.
//
// It reads the tool CALLS, not the results: a call records what was attempted
// even when the result was an error or the process died mid-write, and the
// point here is to warn about anything that may have landed. Shell commands go
// through the same classifier the permission gate uses, so a turn that only ran
// `rg` is not reported as having changed anything.
func changes(t *Turn, msgs []ai.Message) {
	seen := map[string]bool{}
	add := func(list *[]string, kind, value string) {
		if value == "" || seen[kind+value] {
			return
		}
		seen[kind+value] = true
		*list = append(*list, value)
	}

	for _, m := range msgs {
		if m.Role != ai.RoleAssistant {
			continue
		}
		for _, c := range m.Content {
			if c.Type != ai.ContentToolCall {
				continue
			}
			switch c.Name {
			case "write", "edit":
				add(&t.Wrote, "p:", argString(c.Arguments, pathKeys...))
			case "bash":
				if safe, _ := classifyBash(c.Arguments); safe {
					continue
				}
				add(&t.Ran, "c:", firstLine(extractCommand(c.Arguments), 60))
			case "task":
				if role := argString(c.Arguments, "role", "type"); mayMutate(Role(role)) {
					add(&t.Delegated, "d:", role)
				}
			}
		}
	}
}

// mayMutate reports whether a sub-agent role can change the workspace.
//
// An unrecognized role counts as mutating, on the same principle as the shell
// classifier: unknown means unsafe. A role added later that writes must not go
// unreported because this function predates it, and the cost of being wrong is
// one extra line of warning against a silently missed change.
func mayMutate(role Role) bool {
	preset, known := rolePresets[role]
	return !known || preset.mode != ModePlan
}

// argString returns the first key present in raw arguments.
func argString(raw json.RawMessage, keys ...string) string {
	var m map[string]any
	if json.Unmarshal(raw, &m) != nil {
		return ""
	}
	for _, k := range keys {
		if s, ok := m[k].(string); ok && s != "" {
			return s
		}
	}
	return ""
}

// BranchReport describes what a rewind or fork did.
type BranchReport struct {
	// Turn is the turn cut at, or 0 for a branch from the current end.
	Turn int
	// Retained is how many turns survive in the context.
	Retained int
	// Discarded lists the turns removed from the context, in order.
	Discarded []Turn
	// Session is the transcript now being written.
	Session string
	// Forked is set when a new session file was created.
	Forked bool
}

// Changed lists the discarded turns that modified something on disk.
func (r BranchReport) Changed() []Turn {
	var out []Turn
	for _, t := range r.Discarded {
		if t.Changed() {
			out = append(out, t)
		}
	}
	return out
}

// cutFor resolves a user-supplied turn number to a message index.
//
// Turn 0 means the end of the transcript, which is the branch-from-here case.
// Anything else must name a real turn, and the error says how many there are
// rather than only refusing.
func cutFor(turns []Turn, msgs []ai.Message, turn int) (int, error) {
	if turn == 0 {
		return len(msgs), nil
	}
	if turn < 1 || turn > len(turns) {
		if len(turns) == 0 {
			return 0, fmt.Errorf("this session has no turns yet")
		}
		return 0, fmt.Errorf("no turn %d; this session has %d (/turns lists them)", turn, len(turns))
	}
	return turns[turn-1].Start, nil
}

func branchReport(turns []Turn, turn, cut int) BranchReport {
	rep := BranchReport{Turn: turn}
	for _, t := range turns {
		if t.Start < cut {
			rep.Retained++
			continue
		}
		rep.Discarded = append(rep.Discarded, t)
	}
	return rep
}

// Rewind drops the given turn and everything after it from the conversation.
// A turn of 0 means the most recent one, which is the undo-that case.
//
// The transcript FILE is not rewritten. Storage stays append-only and gains a
// rewind marker instead, so the file still records what was tried — the whole
// point of keeping one — while the context the model sees goes back. Rewinding
// is therefore safe to do repeatedly and survives a crash mid-rewind: either
// the marker is on disk or it is not.
//
// What this cannot do is undo the turn's effects. Files written stay written,
// commands run stay run. BranchReport.Changed names them so the caller can say
// so plainly rather than letting the user assume otherwise.
func (h *Harness) Rewind(turn int) (BranchReport, error) {
	actx := h.Agent.Context()
	turns := Turns(actx.Messages)
	if turn == 0 {
		turn = len(turns)
	}
	cut, err := cutFor(turns, actx.Messages, turn)
	if err != nil {
		return BranchReport{}, err
	}
	if cut >= len(actx.Messages) {
		return BranchReport{}, fmt.Errorf("nothing to rewind")
	}

	if err := h.Session.RecordRewind(cut); err != nil {
		return BranchReport{}, err
	}
	actx.Messages = actx.Messages[:cut]
	h.Compactor.SyncSummary(actx.Messages)

	rep := branchReport(turns, turn, cut)
	rep.Session = h.Session.Path()
	return rep, nil
}

// Fork continues in a copy of the session cut at a turn, leaving the original
// file untouched. A turn of 0 branches from the current end.
//
// The retained prefix is byte-identical to the parent's, so the branch's first
// request lands on the same cached prompt. Measured live: the turn after a cut
// read 17792 of 17809 input tokens from cache, the 17 uncached being the new
// prompt itself. Branching costs a file write, not a re-read of the
// conversation — which is what makes "try it another way" a cheap move rather
// than a decision about money.
func (h *Harness) Fork(turn int) (BranchReport, error) {
	actx := h.Agent.Context()
	turns := Turns(actx.Messages)
	cut, err := cutFor(turns, actx.Messages, turn)
	if err != nil {
		return BranchReport{}, err
	}
	retained := actx.Messages[:cut]

	child, err := h.Session.Fork(retained)
	if err != nil {
		return BranchReport{}, err
	}
	// Close the parent only once the branch is safely on disk, so a failure
	// leaves the session still writable rather than orphaned between two files.
	if err := h.Session.Close(); err != nil {
		return BranchReport{}, err
	}
	h.setSession(child)
	actx.Messages = retained
	h.Compactor.SyncSummary(retained)

	rep := branchReport(turns, turn, cut)
	rep.Forked = true
	rep.Session = child.Path()
	return rep, nil
}

// Clear drops the whole conversation.
//
// It is a rewind to nothing, and it goes through the same recording path for
// the same reason: /clear used to change only the in-memory context, so
// `deepseek-pi -c` replayed the file and handed back every message the user had
// just been told was gone. Announcing a cleared conversation and then restoring
// it is worse than not offering the command.
func (h *Harness) Clear() (BranchReport, error) {
	actx := h.Agent.Context()
	turns := Turns(actx.Messages)
	if len(actx.Messages) > 0 {
		if err := h.Session.RecordRewind(0); err != nil {
			return BranchReport{}, err
		}
	}
	actx.Messages = nil
	h.Compactor.SyncSummary(nil)

	rep := branchReport(turns, 1, 0)
	rep.Session = h.Session.Path()
	return rep, nil
}
