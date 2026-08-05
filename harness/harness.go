package harness

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"os"
	"sync"

	"github.com/thevibeworks/deepseek-pi/agent"
	"github.com/thevibeworks/deepseek-pi/ai"
	"github.com/thevibeworks/deepseek-pi/tools"
)

// Options configure a harness.
type Options struct {
	// Cwd is the workspace root. Defaults to the process working directory.
	Cwd string
	// APIKey defaults to $DEEPSEEK_API_KEY.
	APIKey string
	// Model defaults to flash: it is the executor a cost-aware loop should
	// reach for, and pro is an explicit escalation rather than a default.
	Model string
	// Effort is the reasoning level. Empty disables thinking.
	Effort ai.Effort
	// MaxTokens caps output per turn. Zero uses the model maximum.
	MaxTokens int
	// AllowOutsideWorkspace lifts the path scope on file tools.
	AllowOutsideWorkspace bool
	// AppendSystemPrompt is extra system prompt text.
	AppendSystemPrompt string
	// Resume loads a prior transcript from this session file.
	Resume string

	// NoSubagents removes the task tool. Sub-agents are the right default for
	// a coding agent, but an embedder driving deepseek-pi as a component may
	// want a flat one-agent shape.
	NoSubagents bool

	// NoSkills disables skill discovery. Skills are cheap per turn once cached,
	// but a large personal collection dominates the prompt, and someone who
	// wants a minimal prefix should be able to say so.
	NoSkills bool

	// Mode is the permission posture. Empty means ModeDefault.
	Mode Mode
	// AllowTools pre-approves tool names for the session.
	AllowTools []string
	// Approve is called when the policy says Ask. It returns whether to run the
	// call, and whether to remember that answer for the rest of the session.
	//
	// Nil means NO USER IS PRESENT, and Ask becomes a denial. That is the
	// correct default for headless runs: a prompt nobody can answer must not
	// silently become permission.
	Approve func(ctx context.Context, call agent.ToolCall, reason string) (allow, remember bool)
}

// Harness is a fully assembled coding agent.
type Harness struct {
	Agent     *agent.Agent
	Session   *Session
	Client    *ai.Client
	Workspace *tools.Workspace
	Model     ai.Model

	// Policy governs tool permissions. Exposed so a UI can switch modes
	// mid-session without rebuilding the harness.
	Policy *Policy

	// Compactor keeps the session inside the context window. Exposed so a UI
	// can report compactions and force one on demand.
	Compactor *Compactor

	// Checkpoints capture what the file tools change, so a rewind can put the
	// workspace back and not merely report that it cannot.
	Checkpoints *Checkpointer

	// Cache explains prompt-cache misses. The prefix cache is what makes long
	// sessions affordable, so its health should be observable rather than
	// assumed.
	Cache *CacheTracker

	// Skills and Instructions are exposed for status output.
	Skills       []Skill
	Instructions []InstructionFile

	mu       sync.Mutex
	children []ChildReport
}

// Children returns the sub-agents this session has run.
func (h *Harness) Children() []ChildReport {
	h.mu.Lock()
	defer h.mu.Unlock()
	return append([]ChildReport(nil), h.children...)
}

// session returns the transcript currently being written.
//
// Fork replaces it, so anything running off the agent's event goroutine has to
// read it through the lock rather than capturing the pointer once — a callback
// holding the pre-fork session would keep appending to a closed file.
func (h *Harness) session() *Session {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.Session
}

// setSession swaps the transcript being written. Fork is the only caller.
func (h *Harness) setSession(s *Session) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.Session = s
}

// New assembles a harness.
//
// Assembly order matters in one place: the system prompt is built AFTER the
// tools exist, because it is derived from them. That is the co-location rule —
// tool documentation lives with the tool, and the prompt is a deterministic
// function of the registered set.
func New(ctx context.Context, opts Options) (*Harness, error) {
	cwd := opts.Cwd
	if cwd == "" {
		var err error
		cwd, err = os.Getwd()
		if err != nil {
			return nil, fmt.Errorf("resolving working directory: %w", err)
		}
	}

	apiKey := opts.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("DEEPSEEK_API_KEY")
	}
	if apiKey == "" {
		return nil, fmt.Errorf("no API key: set DEEPSEEK_API_KEY")
	}

	modelID := opts.Model
	if modelID == "" {
		modelID = ai.ModelFlash
	}
	model, ok := ai.Lookup(modelID)
	if !ok {
		return nil, fmt.Errorf("unknown model %q (known: %s, %s)", modelID, ai.ModelFlash, ai.ModelPro)
	}
	if opts.Effort != "" && !opts.Effort.Valid() {
		return nil, fmt.Errorf("unknown effort %q (known: low, high, xhigh, max)", opts.Effort)
	}

	ws, err := tools.NewWorkspace(cwd)
	if err != nil {
		return nil, err
	}
	ws.AllowOutside = opts.AllowOutsideWorkspace

	sessionID := newID()
	toolEnv := []string{
		"DEEPSEEK_PI_SESSION_ID=" + sessionID,
		"DEEPSEEK_PI_MODEL=" + model.ID,
		"DEEPSEEK_PI=1",
	}

	client := ai.NewClient(apiKey)
	client.UserAgent = "deepseek-pi/" + Version

	mode := opts.Mode
	if mode == "" {
		mode = ModeDefault
	}
	policy := NewPolicy(mode, opts.AllowTools)

	instructions := DiscoverInstructions(cwd)
	var skills []Skill
	if !opts.NoSkills {
		skills = DiscoverSkills(cwd)
	}

	toolSet := []agent.Tool{
		tools.Read(ws),
		tools.Bash(ws, toolEnv),
		tools.Edit(ws),
		tools.Write(ws),
	}
	var taskTool *taskEnv
	if !opts.NoSubagents {
		taskTool = &taskEnv{
			cwd: cwd, newStream: client.StreamFunc, parentModel: model, parentPolicy: policy,
			instructions: instructions, skills: skills, budget: DefaultBudget(),
		}
		toolSet = append(toolSet, newTaskTool(taskTool))
	}

	// Built from the FINAL tool set, task tool included, so its documentation
	// reaches the model the same way every other tool's does.
	systemPrompt := BuildSystemPrompt(PromptConfig{
		Cwd:                 cwd,
		Tools:               toolSet,
		ProjectInstructions: instructions,
		Skills:              skills,
		Append:              opts.AppendSystemPrompt,
	})

	checkpoints := NewCheckpointer(ws, NewBlobStore(cwd))

	var session *Session
	var history []ai.Message
	if opts.Resume != "" {
		var info SessionInfo
		session, history, info, err = ResumeSession(opts.Resume)
		if err != nil {
			return nil, fmt.Errorf("resuming session: %w", err)
		}
		// Without this a resumed session can rewind its conversation but not
		// the files, which is the difference between undo and a warning.
		checkpoints.Restore(info.Snapshots)
		// Usage on preserved assistant messages must not carry into the new
		// run's accounting, or a resumed session reports tokens it did not
		// spend and any budget check trips immediately.
		history = zeroUsage(history)
	} else {
		session = NewSession(sessionID, cwd, model.ID)
	}

	actx := &agent.Context{
		SystemPrompt: systemPrompt,
		Messages:     history,
		Tools:        toolSet,
	}

	// The compactor shares the parent system prompt so its summarization
	// request lands on the same cached prefix instead of paying full input
	// rate for a second one. It deliberately uses the RAW stream: attribution
	// assumes each request extends the previous one, and a summarization call
	// is a side branch that would read as a break.
	compactor := NewCompactor(model, client.StreamFunc(ctx), systemPrompt)

	tracker := NewCacheTracker(model)

	// Declared before the hooks so they can reach the harness. Fork replaces the
	// session, and a closure that captured the assembly-time pointer would go on
	// writing to a file that has since been closed.
	var h *Harness

	cfg := agent.LoopConfig{
		Model:         model.ID,
		Effort:        opts.Effort,
		MaxTokens:     opts.MaxTokens,
		SessionID:     sessionID,
		ToolExecution: agent.ModeParallel,
		// Compaction runs from the between-turns seam because that is the only
		// safe point: mid-turn the transcript holds an assistant message whose
		// tool calls are still unanswered, and rewriting there orphans them.
		PrepareNextTurn: compactor.PrepareNextTurn,
		BeforeToolCall: func(c context.Context, call agent.ToolCall, actx *agent.Context) agent.BeforeToolResult {
			decision := approve(c, policy, opts.Approve, call)
			if !decision.Block {
				// Capture only what is actually going to run. Snapshotting a
				// refused call would store bytes nothing can ever restore.
				checkpoints.Before(call, len(actx.Messages))
			}
			return decision
		},
		AfterToolCall: func(
			_ context.Context, call agent.ToolCall, _ agent.ToolResult, failed bool, _ *agent.Context,
		) *agent.AfterToolResult {
			if snap := checkpoints.After(call, failed); snap != nil {
				if err := h.session().RecordSnapshot(*snap); err != nil {
					fmt.Fprintf(os.Stderr, "warning: could not record a file snapshot: %v\n", err)
				}
			}
			return nil // the result itself is left exactly as the tool produced it
		},
	}

	a := agent.New(actx, cfg, tracker.Wrap(client.StreamFunc(ctx)))

	h = &Harness{
		Agent: a, Session: session, Client: client,
		Workspace: ws, Model: model, Policy: policy, Compactor: compactor,
		Cache: tracker, Checkpoints: checkpoints,
		Skills: skills, Instructions: instructions,
	}

	if taskTool != nil {
		taskTool.onChild = func(r ChildReport) {
			h.mu.Lock()
			h.children = append(h.children, r)
			h.mu.Unlock()
		}
	}

	compactor.OnEvent = func(ev CompactionEvent) {
		// Compaction rewrites history on purpose, so the prefix break it causes
		// is a cost to report, not a defect to chase.
		tracker.ExpectBreak("compaction rewrote the transcript")
		if err := h.session().RecordCompaction(ev); err != nil {
			fmt.Fprintf(os.Stderr, "warning: could not record compaction: %v\n", err)
		}
	}

	// Persist every finalized message. Streaming partials are skipped: only
	// message_end carries a message worth keeping.
	a.Subscribe(func(ev agent.Event) {
		if ev.Type != agent.EventMessageEnd || ev.Message == nil {
			return
		}
		if err := h.session().Append(*ev.Message); err != nil {
			fmt.Fprintf(os.Stderr, "warning: could not write session: %v\n", err)
		}
	})

	return h, nil
}

// approve turns a policy decision into a loop-level block or pass.
//
// The refusal text goes back to the model as a tool result, so it always says
// what to do next. A bare "denied" produces an identical retry, which costs a
// full round trip and breaks the prompt cache for nothing.
func approve(
	ctx context.Context,
	policy *Policy,
	ask func(context.Context, agent.ToolCall, string) (bool, bool),
	call agent.ToolCall,
) agent.BeforeToolResult {
	decision, reason := policy.Decide(call)
	switch decision {
	case Allow:
		return agent.BeforeToolResult{}
	case Deny:
		return agent.BeforeToolResult{Block: true, Reason: reason}
	}

	// Ask with nobody to ask is a denial, not an approval.
	if ask == nil {
		detail := ""
		if reason != "" {
			detail = " (" + reason + ")"
		}
		return agent.BeforeToolResult{Block: true, Reason: fmt.Sprintf(
			"Refused: %s needs approval%s and this is a non-interactive run. "+
				"Tell the user to re-run with --yolo, or with --allow %s to permit this tool.",
			call.Name, detail, call.Name)}
	}

	allow, remember := ask(ctx, call, reason)
	if !allow {
		return agent.BeforeToolResult{Block: true, Reason: fmt.Sprintf(
			"The user declined to run %s. Do not retry it; "+
				"either take a different approach or ask what they would prefer.", call.Name)}
	}
	if remember {
		policy.Remember(call.Name)
	}
	return agent.BeforeToolResult{}
}

// zeroUsage clears usage on a resumed transcript.
func zeroUsage(msgs []ai.Message) []ai.Message {
	out := make([]ai.Message, len(msgs))
	copy(out, msgs)
	for i := range out {
		out[i].Usage = ai.Usage{}
	}
	return out
}

// Close releases session resources.
func (h *Harness) Close() error { return h.Session.Close() }

// SystemPrompt returns the assembled prompt, for inspection and cache debugging.
func (h *Harness) SystemPrompt() string { return h.Agent.Context().SystemPrompt }

func newID() string {
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		return fmt.Sprintf("%d", os.Getpid())
	}
	return hex.EncodeToString(b[:])
}
