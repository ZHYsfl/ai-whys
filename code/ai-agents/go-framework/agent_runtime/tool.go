package agent_runtime

import (
	"context"

	"github.com/openai/openai-go/v3"
)

type ToolFunc func(ctx context.Context, args map[string]any) (string, error)

type Tool struct {
	Name        string         // the name of the tool
	Description string         // the description of the tool
	Func        ToolFunc       // the function to execute the tool
	Parameters  map[string]any // the parameters of the tool
}

// ToolCallInput is one tool call ready for execution, assembled after a
// streaming accumulation completed.
type ToolCallInput struct {
	ID            string
	Name          string
	ArgumentsJSON string
}

type ToolResponse struct {
	message   openai.ChatCompletionMessageParamUnion
	content   string  // the content of the response
	status    string  // "success" or "error"
	errorType *string // "parse_error", "not_found", "exec_error", "arg_error", "panic"
}
