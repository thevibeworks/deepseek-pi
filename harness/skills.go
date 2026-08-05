package harness

import (
	"os"
	"path/filepath"
	"strings"
)

// Skill is one discovered SKILL.md, reduced to what the prompt index needs.
type Skill struct {
	Name        string
	Description string
	// Path is the absolute path of SKILL.md. The model reads it on demand.
	Path string
	// Hidden keeps a skill out of the prompt while leaving it readable.
	Hidden bool
}

// skillDirs are searched in order. Project skills come first so a project can
// override a personal skill of the same name.
//
// The .agents convention is shared with other harnesses on purpose: a skill is
// a markdown file with frontmatter, and there is no reason for every agent to
// demand its own copy.
func skillDirs(root string) []string {
	var dirs []string
	dirs = append(dirs,
		filepath.Join(root, ".deepseek-pi", "skills"),
		filepath.Join(root, ".agents", "skills"),
	)
	if home, err := os.UserHomeDir(); err == nil {
		dirs = append(dirs,
			filepath.Join(home, ".deepseek-pi", "skills"),
			filepath.Join(home, ".agents", "skills"),
		)
	}
	return dirs
}

// DiscoverSkills finds skills for a workspace.
//
// A skill is a directory containing SKILL.md. The first definition of a given
// name wins, so project skills shadow personal ones. A skill missing a name or
// description is skipped rather than guessed at: those two fields are the
// entire basis on which the model decides to read the body.
func DiscoverSkills(root string) []Skill {
	var out []Skill
	seen := make(map[string]bool)

	for _, dir := range skillDirs(root) {
		entries, err := os.ReadDir(dir)
		if err != nil {
			continue
		}
		for _, e := range entries {
			if !e.IsDir() {
				continue
			}
			path := filepath.Join(dir, e.Name(), "SKILL.md")
			s, ok := parseSkill(path)
			if !ok || seen[s.Name] {
				continue
			}
			seen[s.Name] = true
			out = append(out, s)
		}
	}
	return out
}

// parseSkill reads the YAML-ish frontmatter of a SKILL.md.
//
// Only the handful of scalar keys that matter are parsed, so there is no YAML
// dependency. Anything more elaborate belongs in the skill body, which the
// model reads directly.
func parseSkill(path string) (Skill, bool) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Skill{}, false
	}
	content := string(data)
	if !strings.HasPrefix(content, "---") {
		return Skill{}, false
	}
	rest := content[3:]
	rest = strings.TrimPrefix(rest, "\r")
	rest = strings.TrimPrefix(rest, "\n")
	end := strings.Index(rest, "\n---")
	if end == -1 {
		return Skill{}, false
	}

	s := Skill{Path: path}
	lines := strings.Split(rest[:end], "\n")
	for i := 0; i < len(lines); i++ {
		key, value, found := strings.Cut(lines[i], ":")
		if !found || strings.HasPrefix(lines[i], " ") || strings.HasPrefix(lines[i], "\t") {
			continue
		}
		value = strings.TrimSpace(value)

		// YAML block scalars (`>`, `>-`, `|`, `|-`) put the value on the
		// following indented lines. Skills in the wild use these constantly for
		// long descriptions; reading only the marker would put a literal ">-"
		// in the prompt where the description belongs.
		if isBlockScalar(value) {
			folded := value == "" || value[0] == '>'
			var parts []string
			for j := i + 1; j < len(lines); j++ {
				if strings.TrimSpace(lines[j]) == "" {
					parts = append(parts, "")
					i = j
					continue
				}
				if !strings.HasPrefix(lines[j], " ") && !strings.HasPrefix(lines[j], "\t") {
					break
				}
				parts = append(parts, strings.TrimSpace(lines[j]))
				i = j
			}
			sep := "\n"
			if folded {
				sep = " "
			}
			value = strings.TrimSpace(strings.Join(parts, sep))
		}
		value = strings.Trim(value, `"'`)

		switch strings.TrimSpace(strings.ToLower(key)) {
		case "name":
			s.Name = value
		case "description":
			s.Description = value
		case "hidden", "disable-model-invocation":
			s.Hidden = value == "true"
		}
	}
	if s.Name == "" {
		// Fall back to the directory name so a skill with a description but no
		// explicit name is still usable.
		s.Name = filepath.Base(filepath.Dir(path))
	}
	if s.Description == "" {
		return Skill{}, false
	}
	return s, true
}

// isBlockScalar reports whether a YAML value marker defers the value to the
// following indented lines.
func isBlockScalar(v string) bool {
	if v == "" {
		return false
	}
	if v[0] != '>' && v[0] != '|' {
		return false
	}
	// Only the marker plus optional chomping/indent indicators; anything else
	// is ordinary text that happens to start with > or |.
	for _, r := range v[1:] {
		if r != '-' && r != '+' && (r < '0' || r > '9') {
			return false
		}
	}
	return true
}
