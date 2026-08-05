package harness

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/thevibeworks/deepseek-pi/agent"
	"github.com/thevibeworks/deepseek-pi/ai"
	"github.com/thevibeworks/deepseek-pi/tools"
)

// Role is a sub-agent preset: a system prompt plus a permission posture.
//
// Roles rather than free-form configuration because the useful axis is what a
// sub-agent is FOR. "Read-only, flash, looking for something" and "writes
// files, needs approval semantics" are different jobs, and letting the model
// assemble arbitrary combinations produces neither.
type Role string

const (
	// RoleExplorer searches and reads. It cannot change anything, which is what
	// makes it safe to run several at once.
	RoleExplorer Role = "explorer"
	// RoleImplementer makes changes in the workspace.
	RoleImplementer Role = "implementer"
	// RoleReviewer reads and judges. It runs on pro, because judgment is the
	// one job where the cheaper model is a false economy.
	RoleReviewer Role = "reviewer"
	// RoleTester runs commands and reports what happened.
	RoleTester Role = "tester"
)

type rolePreset struct {
	prompt   string
	mode     Mode
	proModel bool
}

var rolePresets = map[Role]rolePreset{
	RoleExplorer: {
		mode: ModePlan,
		prompt: `You are an explorer sub-agent. Your job is to find things out and report back.

You cannot modify anything, so do not try. Search widely, read what matters, and
stop as soon as you can answer. Your report is the ONLY thing that reaches the
agent that spawned you — it never sees your tool output — so state findings with
concrete file paths and line references, and say plainly what you could not
determine.`,
	},
	RoleImplementer: {
		mode: ModeYolo,
		prompt: `You are an implementer sub-agent. Your job is to make a specific change and verify it.

Read before you write. Make the smallest change that fully solves the stated
problem, and verify it by running something. Your report is the only thing that
reaches the agent that spawned you: say what you changed, in which files, and
what you ran to prove it works. If you could not finish, say exactly where you
stopped and why.`,
	},
	RoleReviewer: {
		mode: ModePlan, proModel: true,
		prompt: `You are a reviewer sub-agent. Your job is to judge work, not to change it.

You cannot modify anything. Read the relevant code and form an opinion with
evidence: what is wrong, where, and why it matters. Lead with the most serious
issue. Distinguish a defect from a preference, and say when something is fine.
Your report is the only thing that reaches the agent that spawned you.`,
	},
	RoleTester: {
		mode: ModePlan,
		prompt: `You are a tester sub-agent. Your job is to run things and report what actually happened.

Run the tests, the build, whatever was asked. Report the real output, not a
summary of what you expected. If something fails, include enough of the failure
for someone else to act on it. Do not fix anything.`,
	},
}

// Roles lists the presets in a stable order for prompts and help text.
func Roles() []Role {
	return []Role{RoleExplorer, RoleImplementer, RoleReviewer, RoleTester}
}

// Budget bounds one sub-agent run.
//
// Every sub-agent gets one. A child that loops is the failure mode that matters:
// unbounded, it burns the budget of the whole session and reports nothing. With
// a budget it dies quietly and the parent still gets a partial report, which is
// strictly more useful than a stall.
type Budget struct {
	MaxTurns  int
	MaxTokens int
	MaxWall   time.Duration
}

// DefaultBudget is deliberately generous enough for real work and strict enough
// that a loop cannot run away.
func DefaultBudget() Budget {
	return Budget{MaxTurns: 30, MaxTokens: 200_000, MaxWall: 5 * time.Minute}
}

// TaskArgs are the task tool's parameters.
type TaskArgs struct {
	Role   string `json:"role"`
	Prompt string `json:"prompt"`
}

// taskEnv is what the task tool needs to spawn children.
type taskEnv struct {
	cwd string
	// newStream builds the provider call for a child. A function rather than a
	// client so a test can run a whole sub-agent without a network.
	newStream    func(context.Context) ai.StreamFunc
	parentModel  ai.Model
	parentPolicy *Policy
	instructions []InstructionFile
	skills       []Skill
	budget       Budget

	mu      sync.Mutex
	spawned int
	// OnChild reports a finished child, for session records and UI.
	onChild func(ChildReport)
}

// ChildReport summarizes one completed sub-agent.
type ChildReport struct {
	Role      Role
	Prompt    string
	Model     string
	Turns     int
	Usage     ai.Usage
	Duration  time.Duration
	Truncated string // non-empty when a budget stopped the child early
	SessionID string
}

// MaxConcurrentChildren bounds how many sub-agents one parent may have running.
//
// Not a resource limit so much as a sanity limit: a model that decides to spawn
// twenty explorers is confused, and the useful response is to make that fail
// visibly rather than to serve it.
const MaxConcurrentChildren = 8

// newTaskTool builds the sub-agent tool.
//
// Recursion is prevented STRUCTURALLY: a child's tool set simply does not
// include this tool. No depth counter to get wrong, no limit to tune — a
// sub-agent cannot spawn a sub-agent because it has no way to express it.
func newTaskTool(env *taskEnv) agent.Tool {
	roleNames := make([]string, 0, len(Roles()))
	for _, r := range Roles() {
		roleNames = append(roleNames, string(r))
	}

	return agent.Tool{
		Name:  "task",
		Label: "Task",
		Description: fmt.Sprintf(
			"Delegate a self-contained piece of work to a sub-agent with its own context.\n"+
				"Roles: %s. explorer and tester and reviewer cannot modify anything; "+
				"implementer can. reviewer runs on the stronger model.\n"+
				"The sub-agent starts with NO knowledge of this conversation, so the prompt must "+
				"be self-contained. You receive only its final report, never its tool output.\n"+
				"Use this to keep a large search or an independent subtask out of your own context. "+
				"Do not use it for work you could do in one or two tool calls.",
			strings.Join(roleNames, ", ")),
		Parameters: ai.Object(map[string]*ai.Schema{
			"role": {
				Type: "string", Enum: roleNames,
				Description: "Which kind of sub-agent to run.",
			},
			"prompt": ai.Str(
				"The complete instruction for the sub-agent. It sees nothing of this " +
					"conversation, so include every detail it needs."),
		}, "role", "prompt"),
		PromptSnippet: "task(role, prompt) — delegate self-contained work to a sub-agent with its own context",
		PromptGuidelines: "Delegate a broad search or an independent subtask with task rather than doing it inline, " +
			"so its tool output stays out of your context. Write the sub-agent prompt as if to a stranger: it cannot see this conversation. " +
			"Run independent sub-agents in one batch so they execute concurrently.",
		// Parallel by design: running several explorers at once is the reason
		// this tool exists.
		ExecutionMode:    agent.ModeParallel,
		PrepareArguments: agent.HealArguments(map[string]string{"instruction": "prompt", "task": "prompt", "type": "role"}),
		Execute: func(ctx context.Context, call agent.ToolCall, onUpdate agent.UpdateFunc) (agent.ToolResult, error) {
			var args TaskArgs
			if err := json.Unmarshal(call.Arguments, &args); err != nil {
				return agent.ToolResult{}, fmt.Errorf("invalid arguments: %w", err)
			}
			role := Role(strings.ToLower(strings.TrimSpace(args.Role)))
			preset, ok := rolePresets[role]
			if !ok {
				return agent.ToolResult{}, fmt.Errorf(
					"unknown role %q; choose one of %s", args.Role, strings.Join(roleNames, ", "))
			}
			if strings.TrimSpace(args.Prompt) == "" {
				return agent.ToolResult{}, fmt.Errorf("prompt must not be empty")
			}

			env.mu.Lock()
			if env.spawned >= MaxConcurrentChildren {
				env.mu.Unlock()
				return agent.ToolResult{}, fmt.Errorf(
					"too many sub-agents in flight (limit %d); do some of this work directly",
					MaxConcurrentChildren)
			}
			env.spawned++
			env.mu.Unlock()
			defer func() {
				env.mu.Lock()
				env.spawned--
				env.mu.Unlock()
			}()

			return runChild(ctx, env, role, preset, args.Prompt, onUpdate)
		},
	}
}

// runChild executes one sub-agent to completion and compresses it to a report.
func runChild(
	ctx context.Context, env *taskEnv, role Role, preset rolePreset,
	prompt string, onUpdate agent.UpdateFunc,
) (agent.ToolResult, error) {
	started := time.Now()

	runCtx, cancel := context.WithTimeout(ctx, env.budget.MaxWall)
	defer cancel()

	model := env.parentModel
	if preset.proModel {
		model = ai.MustLookup(ai.ModelPro)
	}

	// The child is bounded by the STRICTER of its role and its parent. A plan-
	// mode parent must not be able to write through an implementer child, or
	// the mode is decorative.
	mode := stricterMode(preset.mode, env.parentPolicy.Mode)

	ws, err := tools.NewWorkspace(env.cwd)
	if err != nil {
		return agent.ToolResult{}, err
	}
	sessionID := newID()
	childTools := []agent.Tool{
		tools.Read(ws),
		tools.Bash(ws, []string{
			"DEEPSEEK_PI_SESSION_ID=" + sessionID,
			"DEEPSEEK_PI_ROLE=" + string(role),
			"DEEPSEEK_PI=1",
		}),
		tools.Edit(ws),
		tools.Write(ws),
		// No task tool. This is the whole recursion story.
	}

	systemPrompt := BuildSystemPrompt(PromptConfig{
		Cwd:                 env.cwd,
		Tools:               childTools,
		ProjectInstructions: env.instructions,
		Skills:              env.skills,
		Append:              preset.prompt,
	})

	policy := NewPolicy(mode, nil)
	actx := &agent.Context{SystemPrompt: systemPrompt, Tools: childTools}

	var turns int
	var stoppedBy string
	cfg := agent.LoopConfig{
		Model:         model.ID,
		SessionID:     sessionID,
		ToolExecution: agent.ModeParallel,
		BeforeToolCall: func(c context.Context, tc agent.ToolCall, _ *agent.Context) agent.BeforeToolResult {
			// No approver: a sub-agent has no user attached, so anything its
			// role does not already permit is refused.
			return approve(c, policy, nil, tc)
		},
		ShouldStopAfterTurn: func(tc agent.TurnContext) bool {
			turns++
			if turns >= env.budget.MaxTurns {
				stoppedBy = fmt.Sprintf("turn budget (%d turns)", env.budget.MaxTurns)
				return true
			}
			if used := usageOf(tc.NewMessages); used >= env.budget.MaxTokens {
				stoppedBy = fmt.Sprintf("token budget (%d tokens)", env.budget.MaxTokens)
				return true
			}
			return false
		},
	}

	child := agent.New(actx, cfg, env.newStream(runCtx))
	// A sub-agent gets its own transcript on disk: inspectable and resumable
	// afterwards, and excluded from the parent's context except for the report.
	session := NewSession(sessionID, env.cwd, model.ID)
	defer func() { _ = session.Close() }()

	child.Subscribe(func(ev agent.Event) {
		if ev.Type == agent.EventMessageEnd && ev.Message != nil {
			_ = session.Append(*ev.Message)
		}
		if ev.Type == agent.EventToolStart && onUpdate != nil {
			onUpdate(agent.Text(fmt.Sprintf("%s: %s", role, ev.ToolName)))
		}
	})

	msgs, err := child.Prompt(runCtx, prompt)
	if err != nil {
		return agent.ToolResult{}, fmt.Errorf("%s sub-agent: %w", role, err)
	}

	if runCtx.Err() != nil && stoppedBy == "" {
		stoppedBy = fmt.Sprintf("time budget (%s)", env.budget.MaxWall)
	}

	report := ChildReport{
		Role: role, Prompt: prompt, Model: model.ID,
		Turns: turns, Usage: session.Usage,
		Duration: time.Since(started), Truncated: stoppedBy,
		SessionID: sessionID,
	}
	if env.onChild != nil {
		env.onChild(report)
	}

	return agent.ToolResult{
		Content: []ai.Content{ai.TextContent(renderReport(report, msgs, session.Path()))},
		Details: map[string]any{
			"role": string(role), "turns": turns, "session": session.Path(),
			"costUsd": session.Usage.Cost.Total, "stoppedBy": stoppedBy,
		},
	}, nil
}

// renderReport compresses a child run into what the parent sees.
//
// The parent gets the child's final answer and nothing else. Passing the tool
// output through would defeat the entire purpose: the reason to delegate is to
// keep that traffic out of the parent's context.
func renderReport(r ChildReport, msgs []ai.Message, sessionPath string) string {
	answer := ""
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == ai.RoleAssistant {
			if text := strings.TrimSpace(msgs[i].Text()); text != "" {
				answer = text
				break
			}
		}
	}

	var b strings.Builder
	if r.Truncated != "" {
		// Say so first. A partial report that reads as complete is worse than
		// no report, because the parent will act on it as if it were finished.
		fmt.Fprintf(&b, "[PARTIAL: the %s sub-agent was stopped by its %s. "+
			"What follows is as far as it got.]\n\n", r.Role, r.Truncated)
	}
	if answer == "" {
		fmt.Fprintf(&b, "The %s sub-agent finished without producing a report.", r.Role)
	} else {
		b.WriteString(answer)
	}
	fmt.Fprintf(&b, "\n\n---\n%s sub-agent: %d turns, %d in / %d out, $%.4f, %s. Transcript: %s",
		r.Role, r.Turns, r.Usage.Input, r.Usage.Output, r.Usage.Cost.Total,
		r.Duration.Round(time.Millisecond), sessionPath)
	return b.String()
}

// usageOf sums provider-reported tokens across a child's messages.
func usageOf(msgs []ai.Message) int {
	total := 0
	for _, m := range msgs {
		total += m.Usage.Input + m.Usage.Output
	}
	return total
}

// stricterMode returns whichever posture permits less.
//
// Ordering: plan < default < yolo. A child can never be granted more than its
// parent holds, whatever its role would normally allow.
func stricterMode(a, b Mode) Mode {
	rank := map[Mode]int{ModePlan: 0, ModeDefault: 1, ModeYolo: 2}
	if rank[a] <= rank[b] {
		return a
	}
	return b
}
