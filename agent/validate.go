package agent

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/thevibeworks/deepseek-pi/ai"
)

// ValidateArguments checks raw tool arguments against a schema.
//
// It reports EVERY problem in one error rather than stopping at the first.
// A rejected tool call costs a full round trip and breaks the prompt cache, so
// the model gets one message listing everything it must fix instead of
// discovering the problems one turn at a time.
func ValidateArguments(schema *ai.Schema, raw json.RawMessage) error {
	if schema == nil {
		return nil
	}
	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		return fmt.Errorf("arguments are not valid JSON: %v\nReceived: %s", err, clip(string(raw), 400))
	}
	var problems []string
	checkValue(schema, value, "", &problems)
	if len(problems) == 0 {
		return nil
	}
	sort.Strings(problems)
	return fmt.Errorf("invalid arguments:\n  - %s", strings.Join(problems, "\n  - "))
}

func checkValue(schema *ai.Schema, value any, path string, problems *[]string) {
	if schema == nil {
		return
	}
	switch schema.Type {
	case "object":
		obj, ok := value.(map[string]any)
		if !ok {
			*problems = append(*problems, fmt.Sprintf("%s: expected an object, got %s", label(path), kindOf(value)))
			return
		}
		for _, req := range schema.Required {
			if _, present := obj[req]; !present {
				*problems = append(*problems, fmt.Sprintf("%s: required property %q is missing", label(path), req))
			}
		}
		for key, sub := range schema.Properties {
			if v, present := obj[key]; present {
				checkValue(sub, v, join(path, key), problems)
			}
		}
	case "array":
		items, ok := value.([]any)
		if !ok {
			*problems = append(*problems, fmt.Sprintf("%s: expected an array, got %s", label(path), kindOf(value)))
			return
		}
		if schema.Items != nil {
			for i, item := range items {
				checkValue(schema.Items, item, fmt.Sprintf("%s[%d]", label(path), i), problems)
			}
		}
	case "string":
		s, ok := value.(string)
		if !ok {
			*problems = append(*problems, fmt.Sprintf("%s: expected a string, got %s", label(path), kindOf(value)))
			return
		}
		if len(schema.Enum) > 0 && !contains(schema.Enum, s) {
			*problems = append(*problems, fmt.Sprintf("%s: %q is not one of %s", label(path), s, strings.Join(schema.Enum, ", ")))
		}
	case "integer", "number":
		// JSON numbers decode to float64. Accept an integral float for
		// "integer" because that is how every model emits one.
		f, ok := value.(float64)
		if !ok {
			*problems = append(*problems, fmt.Sprintf("%s: expected a number, got %s", label(path), kindOf(value)))
			return
		}
		if schema.Type == "integer" && f != float64(int64(f)) {
			*problems = append(*problems, fmt.Sprintf("%s: expected an integer, got %v", label(path), f))
		}
	case "boolean":
		if _, ok := value.(bool); !ok {
			*problems = append(*problems, fmt.Sprintf("%s: expected a boolean, got %s", label(path), kindOf(value)))
		}
	}
}

// HealArguments fixes the argument shapes models actually get wrong.
//
// Two failures dominate in practice and both are mechanical:
//
//  1. The whole argument object arrives as a JSON *string* ("{\"path\":...}")
//     instead of an object.
//  2. A property name is a near-miss synonym: file_path for path, and friends.
//
// Coercing beats rejecting. Every rejected call is wasted tokens plus a
// cache-unfriendly retry turn, and neither failure carries information about
// the user's intent that a retry would recover.
func HealArguments(aliases map[string]string) func(json.RawMessage) json.RawMessage {
	return func(raw json.RawMessage) json.RawMessage {
		healed := unwrapJSONString(raw)

		if len(aliases) == 0 {
			return healed
		}
		var obj map[string]json.RawMessage
		if err := json.Unmarshal(healed, &obj); err != nil {
			return healed
		}
		changed := false
		for alias, canonical := range aliases {
			v, hasAlias := obj[alias]
			if !hasAlias {
				continue
			}
			if _, hasCanonical := obj[canonical]; !hasCanonical {
				obj[canonical] = v
			}
			delete(obj, alias)
			changed = true
		}
		if !changed {
			return healed
		}
		out, err := json.Marshal(obj)
		if err != nil {
			return healed
		}
		return out
	}
}

// unwrapJSONString turns a JSON string that contains a JSON object into that
// object. Applied repeatedly for the double-encoded case.
func unwrapJSONString(raw json.RawMessage) json.RawMessage {
	current := raw
	for range 3 {
		trimmed := strings.TrimSpace(string(current))
		if !strings.HasPrefix(trimmed, `"`) {
			return current
		}
		var s string
		if err := json.Unmarshal(current, &s); err != nil {
			return current
		}
		inner := strings.TrimSpace(s)
		if !strings.HasPrefix(inner, "{") && !strings.HasPrefix(inner, "[") {
			return current
		}
		if !json.Valid([]byte(inner)) {
			return current
		}
		current = json.RawMessage(inner)
	}
	return current
}

func label(path string) string {
	if path == "" {
		return "arguments"
	}
	return path
}

func join(path, key string) string {
	if path == "" {
		return key
	}
	return path + "." + key
}

func kindOf(v any) string {
	switch v.(type) {
	case nil:
		return "null"
	case bool:
		return "a boolean"
	case float64:
		return "a number"
	case string:
		return "a string"
	case []any:
		return "an array"
	case map[string]any:
		return "an object"
	}
	return "an unknown value"
}

func contains(list []string, s string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}
	return false
}

func clip(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + "... [clipped]"
}
