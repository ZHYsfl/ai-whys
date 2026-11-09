package tools

import (
	"bytes"
	"context"
	"fmt"
	"os/exec"
)

func ExecuteCommand(ctx context.Context, command string, workdir string) (string, error) {
	cmd := exec.CommandContext(ctx, "sh", "-c", command)
	if workdir != "" {
		cmd.Dir = workdir
	}

	var stdoutBuf, stderrBuf bytes.Buffer
	cmd.Stdout = &stdoutBuf
	cmd.Stderr = &stderrBuf

	err := cmd.Run()
	stdoutStr := stdoutBuf.String()
	stderrStr := stderrBuf.String()

	var result string
	if stdoutStr != "" {
		result = "stdout:" + stdoutStr
	}
	if stderrStr != "" {
		if result != "" {
			result += "\n"
		}
		result += "stderr:" + stderrStr
	}

	if err != nil {
		if ctx.Err() != nil {
			return result, fmt.Errorf("command cancelled: %w", ctx.Err())
		}
		return result, fmt.Errorf("command failed: %w", err)
	}
	return result, nil
}
