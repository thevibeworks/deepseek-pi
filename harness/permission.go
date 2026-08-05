package harness

import (
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	"github.com/thevibeworks/deepseek-pi/agent"
)

// Decision is what a policy says about one tool call.
type Decision int

const (
	// Allow runs the call without asking.
	Allow Decision = iota
	// Ask defers to the user. With no user present it becomes a denial.
	Ask
	// Deny refuses the call and tells the model why.
	Deny
)

// Mode is the permission posture for a session.
type Mode string

const (
	// ModeDefault reads freely, asks before anything that can change the
	// machine. This is the posture for a human watching a terminal.
	ModeDefault Mode = "default"
	// ModePlan forbids mutation entirely. Read and safe shell still work, so
	// the agent can investigate and propose without touching anything.
	ModePlan Mode = "plan"
	// ModeYolo allows everything. For sandboxes, disposable checkouts and CI
	// that has already accepted the blast radius.
	ModeYolo Mode = "yolo"
)

// ParseMode validates a mode name.
func ParseMode(s string) (Mode, error) {
	switch Mode(strings.ToLower(strings.TrimSpace(s))) {
	case "", ModeDefault:
		return ModeDefault, nil
	case ModePlan:
		return ModePlan, nil
	case ModeYolo:
		return ModeYolo, nil
	}
	return "", fmt.Errorf("unknown mode %q (default, plan, yolo)", s)
}

// mutatingTools can change the machine. Everything not listed is treated as
// read-only, so adding a tool without classifying it fails open on safety only
// if it genuinely cannot mutate — which is why new mutating tools must be
// added here in the same commit that adds the tool.
var mutatingTools = map[string]bool{
	"edit":  true,
	"write": true,
	"bash":  true, // conditionally: see classifyBash
}

// Policy decides whether a tool call may run.
//
// It is deliberately not a general rules engine. Three modes and one
// conservative shell classifier cover the real cases; anything finer grained
// ends up being configuration nobody audits, which is worse than a prompt.
type Policy struct {
	Mode Mode

	// AllowTools are tool names pre-approved for the session, from --allow.
	AllowTools map[string]bool

	mu sync.Mutex
	// remembered holds "allow this tool for the rest of the session" answers.
	remembered map[string]bool
}

// NewPolicy builds a policy.
func NewPolicy(mode Mode, allow []string) *Policy {
	set := make(map[string]bool, len(allow))
	for _, name := range allow {
		name = strings.TrimSpace(name)
		if name != "" {
			set[name] = true
		}
	}
	return &Policy{Mode: mode, AllowTools: set, remembered: map[string]bool{}}
}

// Remember records a session-scoped approval for a tool.
func (p *Policy) Remember(tool string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.remembered[tool] = true
}

// Decide classifies one tool call and, for Ask and Deny, explains why.
//
// The reason string is written for the MODEL, not the user: it goes back as a
// tool result, so it must say what was refused and what the model can do
// instead. "Permission denied" teaches it nothing and it retries the same call.
// Order matters here and is the whole security property. Plan mode is checked
// BEFORE the allow-list and before remembered approvals, so neither a stale
// session-scoped "yes" nor a --allow flag can erode it. A mode that some other
// setting can quietly override is not a mode, it is a suggestion.
func (p *Policy) Decide(call agent.ToolCall) (Decision, string) {
	if p.Mode == ModeYolo {
		return Allow, ""
	}

	// Read-only shell is how this agent searches and navigates, since
	// grep/find/ls are folded into bash. Prompting for `rg foo` would make
	// every mode unusable, so classify before anything else.
	bashSafe, bashWhy := true, ""
	if call.Name == "bash" {
		bashSafe, bashWhy = classifyBash(call.Arguments)
	}
	mutates := mutatingTools[call.Name] && (call.Name != "bash" || !bashSafe)

	if !mutates {
		return Allow, ""
	}

	if p.Mode == ModePlan {
		// The tool stays REGISTERED and only the permission changes. Removing a
		// tool would change the tool schemas, which are part of the cached
		// prompt prefix, so toggling plan mode mid-session would re-bill the
		// entire prefix. Modes change permissions, never schemas.
		if call.Name == "bash" {
			return Deny, fmt.Sprintf(
				"Refused: plan mode does not run commands that can change the machine (%s). "+
					"Investigate with read-only commands, then describe the change you would make.", bashWhy)
		}
		return Deny, fmt.Sprintf(
			"Refused: plan mode is read-only, so %s cannot run. "+
				"Describe the change you would make and let the user leave plan mode to apply it.", call.Name)
	}

	if p.AllowTools[call.Name] {
		return Allow, ""
	}
	p.mu.Lock()
	remembered := p.remembered[call.Name]
	p.mu.Unlock()
	if remembered {
		return Allow, ""
	}

	return Ask, bashWhy
}

// safeCommands are shell commands that only read.
//
// The list is short on purpose. Everything absent from it prompts, so the cost
// of omitting something is one keystroke while the cost of wrongly including
// something is a mutated workspace. sed and awk are excluded despite being
// common readers: `sed -i` edits in place and awk can redirect to a file, and a
// classifier that has to reason about their flags is a classifier that will be
// wrong eventually.
var safeCommands = map[string]bool{
	"ls": true, "cat": true, "head": true, "tail": true, "wc": true,
	"rg": true, "grep": true, "egrep": true, "fgrep": true, "ag": true,
	"find": true, "fd": true, "file": true, "stat": true, "tree": true,
	"pwd": true, "echo": true, "which": true, "whereis": true, "type": true,
	"basename": true, "dirname": true, "realpath": true, "readlink": true,
	"sort": true, "uniq": true, "cut": true, "tr": true, "column": true,
	"diff": true, "cmp": true, "du": true, "df": true, "date": true,
	"jq": true, "yq": true, "true": true, "false": true, "test": true,
	"printf": true, "seq": true, "nl": true, "cd": true, "env": true,
	"uname": true, "hostname": true, "id": true, "ps": true,
}

// safeSubcommands covers tools whose safety depends on the verb.
var safeSubcommands = map[string]map[string]bool{
	"git": {
		"status": true, "log": true, "diff": true, "show": true,
		"blame": true, "ls-files": true, "rev-parse": true, "describe": true,
		"shortlog": true, "grep": true, "cat-file": true, "for-each-ref": true,
	},
	"go": {
		"doc": true, "list": true, "version": true, "env": true,
	},
	"cargo": {"tree": true, "--version": true},
	"npm":   {"ls": true, "view": true, "outdated": true},
	"docker": {
		"ps": true, "images": true, "logs": true, "inspect": true, "version": true,
	},
	"kubectl": {"get": true, "describe": true, "logs": true, "version": true},
}

// dangerousFragments force a prompt wherever they appear, because each one
// either writes to the filesystem or executes text this classifier never sees.
//
// They are checked BEFORE the command is split, so no amount of chaining hides
// them. `&&` and `||` are deliberately absent: chaining two read-only commands
// is ordinary and safe, and splitting handles it. A bare `&` is checked per
// segment afterwards, where it unambiguously means "background this".
var dangerousFragments = []struct {
	frag, why string
}{
	{">", "redirects output to a file"},
	{"$(", "runs a command substitution"},
	{"`", "runs a command substitution"},
	{"<(", "runs a process substitution"},
	{"\n", "spans multiple lines"},
}

// classifyBash reports whether a shell command only reads.
//
// The rule is conservative by construction: the command is split on the
// operators that chain commands, EVERY segment must lead with a known-safe
// program, and any construct that writes or evaluates text disqualifies the
// whole command. Unknown means unsafe. This classifier decides whether to
// prompt, so its failure mode should be an unnecessary question, never an
// unintended write.
func classifyBash(rawArgs []byte) (safe bool, why string) {
	command := extractCommand(rawArgs)
	if strings.TrimSpace(command) == "" {
		return false, "the command could not be read"
	}

	for _, d := range dangerousFragments {
		if strings.Contains(command, d.frag) {
			return false, "the command " + d.why
		}
	}

	for _, segment := range splitPipeline(command) {
		if strings.Contains(segment, "&") {
			return false, "the command backgrounds a process"
		}
		fields := strings.Fields(segment)
		if len(fields) == 0 {
			continue
		}
		head := fields[0]
		// A leading VAR=value assignment hides the real command behind it.
		if strings.Contains(head, "=") {
			return false, "the command sets environment variables inline"
		}
		// Absolute or relative paths bypass the name check entirely.
		if strings.ContainsRune(head, '/') {
			return false, fmt.Sprintf("%q runs a program by path", head)
		}

		if subs, ok := safeSubcommands[head]; ok {
			if len(fields) < 2 {
				return false, fmt.Sprintf("%q was given no subcommand", head)
			}
			verb := fields[1]
			if !subs[verb] {
				return false, fmt.Sprintf("%s %s is not a read-only subcommand", head, verb)
			}
			continue
		}
		if !safeCommands[head] {
			return false, fmt.Sprintf("%q is not on the read-only command list", head)
		}
	}
	return true, ""
}

// extractCommand pulls the command field out of raw bash arguments. It reads
// the JSON directly rather than importing the tools package, which would make
// harness -> tools -> agent into a cycle.
func extractCommand(rawArgs []byte) string {
	var args struct {
		Command string `json:"command"`
	}
	if err := json.Unmarshal(rawArgs, &args); err != nil {
		return ""
	}
	return args.Command
}

// splitPipeline breaks a command on the operators that separate commands.
func splitPipeline(command string) []string {
	replacer := strings.NewReplacer("&&", "\x00", "||", "\x00", "|", "\x00", ";", "\x00")
	return strings.Split(replacer.Replace(command), "\x00")
}
