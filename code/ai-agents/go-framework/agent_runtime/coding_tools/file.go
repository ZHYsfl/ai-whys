package tools

import (
	"context"
	"fmt"
	"os"
	"strings"
)

const defaultMaxResponseLen = 10000

func EditFile(ctx context.Context, path string, oldString string, newString string) (string, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return fmt.Sprintf("failed to read file: %v", err), fmt.Errorf("failed to read file: %w", err)
	}
	replaced := strings.Replace(string(content), oldString, newString, 1)
	if replaced == string(content) {
		return "old_string not found in file", fmt.Errorf("old_string not found in file")
	}
	if err := os.WriteFile(path, []byte(replaced), 0644); err != nil {
		return fmt.Sprintf("failed to write file: %v", err), fmt.Errorf("failed to write file: %w", err)
	}
	return "successfully edited file", nil
}

func ReadFile(ctx context.Context, path string, startLine int, endLine int, maxResponseLen int) (string, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("failed to read file: %w", err)
	}

	selected := string(content)
	if startLine > 0 || endLine > 0 {
		lines := strings.Split(selected, "\n")
		if startLine < 0 {
			startLine = 0
		}
		if endLine <= 0 || endLine > len(lines) {
			endLine = len(lines)
		}
		if startLine > endLine {
			startLine = endLine // start past EOF: empty range, not a panic
		}
		selected = strings.Join(lines[startLine:endLine], "\n")
	}

	if maxResponseLen <= 0 {
		maxResponseLen = defaultMaxResponseLen
	}
	runes := []rune(selected)
	if len(runes) > maxResponseLen {
		selected = string(runes[:maxResponseLen]) + "\n[truncated by max_response_len]"
	}

	return selected, nil
}

func WriteFile(ctx context.Context, path string, content string) (string, error) {
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		return fmt.Sprintf("failed to write file: %v", err), fmt.Errorf("failed to write file: %w", err)
	}
	return "successfully wrote file", nil
}

func AppendFile(ctx context.Context, path string, content string) (string, error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		return fmt.Sprintf("failed to open file: %v", err), fmt.Errorf("failed to open file: %w", err)
	}
	defer f.Close()
	if _, err := f.WriteString(content); err != nil {
		return fmt.Sprintf("failed to append: %v", err), fmt.Errorf("failed to append: %w", err)
	}
	return "successfully appended to file", nil
}
