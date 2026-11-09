package agent_runtime

import (
	"context"

	"github.com/openai/openai-go/v3"
)

type MemoryMode int

const (
	MemoryModeNone MemoryMode = iota
	MemoryModeSlideWindow
	MemoryModeCompress
)

type Summarizer func(ctx context.Context, msgs []openai.ChatCompletionMessageParamUnion) (string, error)

type MemoryConfig struct {
	Mode       MemoryMode
	Max        int
	KeepRecent int
	Summarizer Summarizer
}

func WithMemory(cfg MemoryConfig) AgentOption {
	return func(a *Agent) { a.memory = newMemory(cfg) }
}

func newMemory(cfg MemoryConfig) MemoryManager {
	switch cfg.Mode {
	case MemoryModeSlideWindow:
		return &slideWindowMemory{maxMessages: cfg.Max}
	case MemoryModeCompress:
		keep := cfg.KeepRecent
		if keep <= 0 {
			keep = compressKeepRecentMessages
		}
		return &compressMemory{maxInputTokens: cfg.Max, keepRecent: keep, summarize: cfg.Summarizer}
	default:
		return &defaultMemory{}
	}
}

type MemoryManager interface {
	Prepare(messages []openai.ChatCompletionMessageParamUnion) []openai.ChatCompletionMessageParamUnion
}

type usageRecorder interface{ RecordUsage(promptTokens int64) }

type defaultMemory struct{}

func (m *defaultMemory) Prepare(messages []openai.ChatCompletionMessageParamUnion) []openai.ChatCompletionMessageParamUnion {
	return messages
}

func splitSystem(messages []openai.ChatCompletionMessageParamUnion) (
	system openai.ChatCompletionMessageParamUnion,
	nonSystem []openai.ChatCompletionMessageParamUnion,
) {
	for _, msg := range messages {
		if msg.OfSystem != nil {
			system = msg
			continue
		}
		nonSystem = append(nonSystem, msg)
	}
	return system, nonSystem
}

func joinSystem(
	system openai.ChatCompletionMessageParamUnion,
	msgs []openai.ChatCompletionMessageParamUnion,
) []openai.ChatCompletionMessageParamUnion {
	out := make([]openai.ChatCompletionMessageParamUnion, 0, len(msgs)+1)
	if system.OfSystem != nil {
		out = append(out, system)
	}
	return append(out, msgs...)
}

func DropLeadingOrphanTools(msgs []openai.ChatCompletionMessageParamUnion) []openai.ChatCompletionMessageParamUnion {
	i := 0
	for i < len(msgs) && msgs[i].OfTool != nil {
		i++
	}
	return msgs[i:]
}
