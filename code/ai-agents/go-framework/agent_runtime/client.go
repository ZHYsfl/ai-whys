package agent_runtime

import (
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

func NewOpenAIClient(config *LLMConfig) openai.Client {
	clientOpts := []option.RequestOption{
		option.WithAPIKey(config.APIKey),
		option.WithBaseURL(config.BaseURL),
	}

	return openai.NewClient(clientOpts...)
}
