package agent_runtime

import (
	"context"
	"sync"

	"github.com/openai/openai-go/v3"
)

// ExecuteToolCalls runs every call concurrently and returns the tool-role
// messages in call order. Parse/argument/execution failures still become tool
// messages (the model sees the error); nothing is returned as error. A tool
// that panics is caught the same way — see executeToolCallFunc.
func (a *Agent) ExecuteToolCalls(ctx context.Context, calls []ToolCallInput) []openai.ChatCompletionMessageParamUnion {
	availableTools := make(map[string]*Tool, len(a.tools))
	for _, t := range a.tools {
		availableTools[t.Name] = t
	}

	results := make([]ToolResponse, len(calls))
	var wg sync.WaitGroup
	for i, c := range calls {
		wg.Add(1)
		go func(idx int, call ToolCallInput) {
			defer wg.Done()
			results[idx] = a.executeToolCallFunc(ctx, call.ID, call.Name, call.ArgumentsJSON, availableTools)
		}(i, c)
	}
	wg.Wait()

	msgs := make([]openai.ChatCompletionMessageParamUnion, len(results))
	for i, r := range results {
		msgs[i] = r.message
	}
	return msgs
}
