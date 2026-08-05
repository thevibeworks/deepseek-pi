package harness

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"os"

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
	// ApproveTool gates every tool call. Nil allows everything.
	ApproveTool func(context.Context, agent.ToolCall, *agent.Context) agent.BeforeToolResult
}

// Harness is a fully assembled coding agent.
type Harness struct {
	Agent     *agent.Agent
	Session   *Session
	Client    *ai.Client
	Workspace *tools.Workspace
	Model     ai.Model

	// Skills and Instructions are exposed for status output.
	Skills       []Skill
	Instructions []InstructionFile
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

	toolSet := []agent.Tool{
		tools.Read(ws),
		tools.Bash(ws, toolEnv),
		tools.Edit(ws),
		tools.Write(ws),
	}

	instructions := DiscoverInstructions(cwd)
	skills := DiscoverSkills(cwd)

	systemPrompt := BuildSystemPrompt(PromptConfig{
		Cwd:                 cwd,
		Tools:               toolSet,
		ProjectInstructions: instructions,
		Skills:              skills,
		Append:              opts.AppendSystemPrompt,
	})

	var session *Session
	var history []ai.Message
	if opts.Resume != "" {
		session, history, err = ResumeSession(opts.Resume)
		if err != nil {
			return nil, fmt.Errorf("resuming session: %w", err)
		}
		// Usage on preserved assistant messages must not carry into the new
		// run's accounting, or a resumed session reports tokens it did not
		// spend and any budget check trips immediately.
		history = zeroUsage(history)
	} else {
		session = NewSession(sessionID, cwd, model.ID)
	}

	client := ai.NewClient(apiKey)
	client.UserAgent = "deepseek-pi/" + Version

	actx := &agent.Context{
		SystemPrompt: systemPrompt,
		Messages:     history,
		Tools:        toolSet,
	}

	cfg := agent.LoopConfig{
		Model:         model.ID,
		Effort:        opts.Effort,
		MaxTokens:     opts.MaxTokens,
		SessionID:     sessionID,
		ToolExecution: agent.ModeParallel,
		BeforeToolCall: func(c context.Context, call agent.ToolCall, actx *agent.Context) agent.BeforeToolResult {
			if opts.ApproveTool == nil {
				return agent.BeforeToolResult{}
			}
			return opts.ApproveTool(c, call, actx)
		},
	}

	a := agent.New(actx, cfg, client.StreamFunc(ctx))

	h := &Harness{
		Agent: a, Session: session, Client: client,
		Workspace: ws, Model: model,
		Skills: skills, Instructions: instructions,
	}

	// Persist every finalized message. Streaming partials are skipped: only
	// message_end carries a message worth keeping.
	a.Subscribe(func(ev agent.Event) {
		if ev.Type != agent.EventMessageEnd || ev.Message == nil {
			return
		}
		if err := session.Append(*ev.Message); err != nil {
			fmt.Fprintf(os.Stderr, "warning: could not write session: %v\n", err)
		}
	})

	return h, nil
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
