package agent_runtime

import (
	"github.com/openai/openai-go/v3"
)

type slideWindowMemory struct {
	maxMessages int
}

func (m *slideWindowMemory) Prepare(messages []openai.ChatCompletionMessageParamUnion) []openai.ChatCompletionMessageParamUnion {
	if m.maxMessages <= 0 {
		return messages
	}
	system, nonSystem := splitSystem(messages)
	if len(nonSystem) > m.maxMessages {
		nonSystem = nonSystem[len(nonSystem)-m.maxMessages:]
	}
	return joinSystem(system, DropLeadingOrphanTools(nonSystem))
}
