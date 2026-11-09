package agent_runtime

import (
	"context"
	"fmt"
	"log"

	"github.com/openai/openai-go/v3"
)

func (a *Agent) BuildChatCompletionParams(messages []openai.ChatCompletionMessageParamUnion, withTool bool) openai.ChatCompletionNewParams {
	prepared := a.memory.Prepare(messages)
	params := openai.ChatCompletionNewParams{
		Model:    openai.ChatModel(a.config.Model),
		Messages: prepared,
	}

	if withTool {
		params.Tools = a.GetTools()
		params.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{
			OfAuto: openai.String("auto"),
		}
	}

	if a.config.Temperature != nil {
		params.Temperature = openai.Float(*a.config.Temperature)
	}
	if a.config.ExtraBody != nil {
		params.SetExtraFields(a.config.ExtraBody)
	}

	return params
}

func (a *Agent) ChatWithoutToolCall(
	ctx context.Context,
	messages []openai.ChatCompletionMessageParamUnion,
) ([]openai.ChatCompletionMessageParamUnion, string, error) {
	params := a.BuildChatCompletionParams(messages, false)

	resp, err := a.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return nil, "", fmt.Errorf("chat text: %w", err)
	}
	if len(resp.Choices) == 0 {
		return nil, "", fmt.Errorf("chat text: no choices returned")
	}

	next := make([]openai.ChatCompletionMessageParamUnion, len(messages))
	copy(next, messages)
	next = append(next, AssistantMsgToParam(resp.Choices[0].Message))

	return next, resp.Choices[0].Message.Content, nil
}

func (a *Agent) ChatWithoutToolCallStream(
	ctx context.Context,
	messages []openai.ChatCompletionMessageParamUnion,
) <-chan string {
	ch := make(chan string, 200)

	go func() {
		defer close(ch)

		params := a.BuildChatCompletionParams(messages, false)

		stream := a.client.Chat.Completions.NewStreaming(ctx, params)

		for stream.Next() {
			chunk := stream.Current()
			for _, choice := range chunk.Choices {
				if choice.Delta.Content != "" {
					select {
					case ch <- choice.Delta.Content:
					case <-ctx.Done():
						return
					}
				}
			}
		}

		if err := stream.Err(); err != nil {
			if ctx.Err() == nil {
				log.Printf("stream chat error: %v", err)
			}
		}
	}()

	return ch
}
