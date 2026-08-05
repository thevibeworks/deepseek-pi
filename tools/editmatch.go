package tools

import (
	"fmt"
	"sort"
	"strings"

	"golang.org/x/text/unicode/norm"
)

// TextEdit is one text replacement.
type TextEdit struct {
	OldText string `json:"oldText"`
	NewText string `json:"newText"`
}

// DetectLineEnding reports the dominant line ending, judged by whichever style
// appears first. Files are edited in LF space and restored on write.
func DetectLineEnding(content string) string {
	crlf := strings.Index(content, "\r\n")
	lf := strings.Index(content, "\n")
	if lf == -1 || crlf == -1 {
		return "\n"
	}
	if crlf < lf {
		return "\r\n"
	}
	return "\n"
}

// NormalizeToLF collapses CRLF and lone CR to LF.
func NormalizeToLF(s string) string {
	s = strings.ReplaceAll(s, "\r\n", "\n")
	return strings.ReplaceAll(s, "\r", "\n")
}

// RestoreLineEndings converts LF back to the file's original ending.
func RestoreLineEndings(s, ending string) string {
	if ending == "\r\n" {
		return strings.ReplaceAll(s, "\n", "\r\n")
	}
	return s
}

// StripBOM splits a leading UTF-8 BOM off content so it can be restored on
// write. An edit must not silently add or remove a BOM.
func StripBOM(content string) (bom, text string) {
	const b = "\uFEFF"
	if strings.HasPrefix(content, b) {
		return b, strings.TrimPrefix(content, b)
	}
	return "", content
}

var fuzzyReplacer = strings.NewReplacer(
	// Smart single quotes.
	"‘", "'", "’", "'", "‚", "'", "‛", "'",
	// Smart double quotes.
	"“", `"`, "”", `"`, "„", `"`, "‟", `"`,
	// Dashes and minus.
	"‐", "-", "‑", "-", "‒", "-", "–", "-",
	"—", "-", "―", "-", "−", "-",
	// Exotic spaces.
	" ", " ", " ", " ", " ", " ", " ", " ", " ", " ",
	" ", " ", " ", " ", " ", " ", " ", " ", " ", " ",
	" ", " ", " ", " ", "　", " ",
)

// NormalizeForFuzzyMatch maps text into the space where matching is forgiving.
//
// These are the transforms that recover REAL model mistakes: it retypes a
// smart quote as ASCII, drops trailing whitespace, or swaps an em-dash for a
// hyphen. NFKC additionally folds full-width and compatibility forms, which
// matters in CJK-adjacent source. Nothing here changes semantics, which is why
// it is safe to match in this space and then write back original bytes.
func NormalizeForFuzzyMatch(text string) string {
	text = norm.NFKC.String(text)
	lines := strings.Split(text, "\n")
	for i, line := range lines {
		lines[i] = strings.TrimRight(line, " \t")
	}
	return fuzzyReplacer.Replace(strings.Join(lines, "\n"))
}

type matched struct {
	editIndex int
	index     int
	length    int
	newText   string
}

// ApplyEdits applies every edit to LF-normalized content.
//
// The rules, in the order they matter:
//
//   - Exact match first, fuzzy only as a fallback. Exact is what preserves
//     intent; fuzzy is what rescues a near-miss.
//   - Every edit matches against the SAME original content, and replacements
//     are applied in reverse offset order, so earlier edits cannot shift the
//     offsets of later ones.
//   - A non-unique oldText is an error, not a first-match-wins guess. Guessing
//     which of three occurrences the model meant is how an agent corrupts a
//     file while reporting success.
//   - When any edit needed fuzzy matching, untouched lines are copied back
//     BYTE FOR BYTE from the original. Normalizing a whole file as a side
//     effect of one edit would produce an enormous, unreviewable diff.
func ApplyEdits(content string, edits []TextEdit, path string) (string, error) {
	normalized := make([]TextEdit, len(edits))
	for i, e := range edits {
		normalized[i] = TextEdit{OldText: NormalizeToLF(e.OldText), NewText: NormalizeToLF(e.NewText)}
		if normalized[i].OldText == "" {
			return "", emptyOldTextError(path, i, len(edits))
		}
	}

	usedFuzzy := false
	for _, e := range normalized {
		if !strings.Contains(content, e.OldText) {
			usedFuzzy = true
			break
		}
	}

	base := content
	if usedFuzzy {
		base = NormalizeForFuzzyMatch(content)
	}

	found := make([]matched, 0, len(normalized))
	for i, e := range normalized {
		needle := e.OldText
		if usedFuzzy {
			needle = NormalizeForFuzzyMatch(needle)
		}
		idx := strings.Index(base, needle)
		if idx == -1 {
			return "", notFoundError(path, i, len(normalized))
		}
		if n := strings.Count(base, needle); n > 1 {
			return "", duplicateError(path, i, len(normalized), n)
		}
		found = append(found, matched{
			editIndex: i, index: idx, length: len(needle), newText: e.NewText,
		})
	}

	sort.Slice(found, func(a, b int) bool { return found[a].index < found[b].index })
	for i := 1; i < len(found); i++ {
		prev, cur := found[i-1], found[i]
		if prev.index+prev.length > cur.index {
			return "", fmt.Errorf(
				"edits[%d] and edits[%d] overlap in %s. Merge them into one edit or target disjoint regions",
				prev.editIndex, cur.editIndex, path)
		}
	}

	var out string
	if usedFuzzy {
		var err error
		out, err = applyPreservingUnchangedLines(content, base, found)
		if err != nil {
			return "", err
		}
	} else {
		out = applyReplacements(base, found, 0)
	}

	if out == content {
		return "", noChangeError(path, len(normalized))
	}
	return out, nil
}

// applyReplacements rewrites in reverse order so offsets stay valid.
func applyReplacements(content string, reps []matched, offset int) string {
	out := content
	for i := len(reps) - 1; i >= 0; i-- {
		at := reps[i].index - offset
		out = out[:at] + reps[i].newText + out[at+reps[i].length:]
	}
	return out
}

type lineSpan struct{ start, end int }

// splitLinesKeepEnds splits into lines that retain their terminators, so
// rejoining is lossless.
func splitLinesKeepEnds(s string) []string {
	if s == "" {
		return nil
	}
	var out []string
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			out = append(out, s[start:i+1])
			start = i + 1
		}
	}
	if start < len(s) {
		out = append(out, s[start:])
	}
	return out
}

func lineSpans(s string) []lineSpan {
	spans := make([]lineSpan, 0, 64)
	offset := 0
	for _, line := range splitLinesKeepEnds(s) {
		spans = append(spans, lineSpan{start: offset, end: offset + len(line)})
		offset += len(line)
	}
	return spans
}

// applyPreservingUnchangedLines overlays fuzzy-space replacements onto the
// original content, widening each replacement to the lines it touches and
// copying every other line back verbatim.
//
// The line-count invariant is what makes this sound: NormalizeForFuzzyMatch
// never adds or removes a newline, so line N in the normalized view is line N
// in the original. If that ever stops holding, this returns an error rather
// than writing a file it cannot justify.
func applyPreservingUnchangedLines(original, base string, reps []matched) (string, error) {
	originalLines := splitLinesKeepEnds(original)
	baseLines := lineSpans(base)
	if len(originalLines) != len(baseLines) {
		return "", fmt.Errorf(
			"internal: cannot preserve unchanged lines (%d original lines vs %d normalized)",
			len(originalLines), len(baseLines))
	}

	type group struct {
		startLine, endLine int
		reps               []matched
	}
	var groups []group

	sorted := append([]matched(nil), reps...)
	sort.Slice(sorted, func(a, b int) bool { return sorted[a].index < sorted[b].index })

	for _, r := range sorted {
		startLine, endLine, err := replacementLineRange(baseLines, r)
		if err != nil {
			return "", err
		}
		if n := len(groups); n > 0 && startLine < groups[n-1].endLine {
			if endLine > groups[n-1].endLine {
				groups[n-1].endLine = endLine
			}
			groups[n-1].reps = append(groups[n-1].reps, r)
			continue
		}
		groups = append(groups, group{startLine: startLine, endLine: endLine, reps: []matched{r}})
	}

	var b strings.Builder
	next := 0
	for _, g := range groups {
		for i := next; i < g.startLine; i++ {
			b.WriteString(originalLines[i])
		}
		from := baseLines[g.startLine].start
		to := baseLines[g.endLine-1].end
		b.WriteString(applyReplacements(base[from:to], g.reps, from))
		next = g.endLine
	}
	for i := next; i < len(originalLines); i++ {
		b.WriteString(originalLines[i])
	}
	return b.String(), nil
}

func replacementLineRange(lines []lineSpan, r matched) (int, int, error) {
	end := r.index + r.length
	startLine := -1
	for i, l := range lines {
		if r.index >= l.start && r.index < l.end {
			startLine = i
			break
		}
	}
	if startLine == -1 {
		// A match at the very end of content with no trailing newline.
		if len(lines) > 0 && r.index == lines[len(lines)-1].end {
			startLine = len(lines) - 1
		} else {
			return 0, 0, fmt.Errorf("internal: replacement range outside content")
		}
	}
	endLine := startLine
	for endLine < len(lines) && lines[endLine].end < end {
		endLine++
	}
	if endLine >= len(lines) {
		endLine = len(lines) - 1
	}
	return startLine, endLine + 1, nil
}

// The error strings below are pedagogical on purpose. A model reads them and
// retries; "edit failed" produces another identical failing call, while naming
// the cause and the fix produces a correct one.

func notFoundError(path string, i, total int) error {
	if total == 1 {
		return fmt.Errorf(
			"could not find the text in %s. oldText must match the file exactly, "+
				"including all whitespace and indentation. Read the file and copy the exact text", path)
	}
	return fmt.Errorf(
		"could not find edits[%d] in %s. oldText must match the file exactly, "+
			"including all whitespace and indentation", i, path)
}

func duplicateError(path string, i, total, n int) error {
	if total == 1 {
		return fmt.Errorf(
			"found %d occurrences of the text in %s. oldText must be unique: "+
				"include surrounding lines until it identifies exactly one location", n, path)
	}
	return fmt.Errorf(
		"found %d occurrences of edits[%d] in %s. Each oldText must be unique: "+
			"include surrounding lines until it identifies exactly one location", n, i, path)
}

func emptyOldTextError(path string, i, total int) error {
	if total == 1 {
		return fmt.Errorf("oldText must not be empty in %s. To create a file, use write", path)
	}
	return fmt.Errorf("edits[%d].oldText must not be empty in %s", i, path)
}

func noChangeError(path string, total int) error {
	if total == 1 {
		return fmt.Errorf(
			"no change made to %s: the replacement is identical to the original text", path)
	}
	return fmt.Errorf("no change made to %s: the replacements are identical to the original text", path)
}
