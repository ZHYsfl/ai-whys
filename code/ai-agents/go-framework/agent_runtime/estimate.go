package agent_runtime

import (
	"strings"

	"github.com/openai/openai-go/v3"
)

func estimateMessages(msgs []openai.ChatCompletionMessageParamUnion) int {
	const overhead = 8 // per-message role/template tokens
	total := 0
	for _, msg := range msgs {
		total += overhead + estimateTokens(messageText(msg))
	}
	return total
}

func messageText(msg openai.ChatCompletionMessageParamUnion) string {
	var b strings.Builder
	switch {
	case msg.OfSystem != nil:
		if v := msg.OfSystem.Content.OfString; v.Valid() {
			b.WriteString(v.Value)
		}
		for _, p := range msg.OfSystem.Content.OfArrayOfContentParts {
			b.WriteString(p.Text)
		}
	case msg.OfUser != nil:
		if v := msg.OfUser.Content.OfString; v.Valid() {
			b.WriteString(v.Value)
		}
		for _, p := range msg.OfUser.Content.OfArrayOfContentParts {
			if p.OfText != nil {
				b.WriteString(p.OfText.Text)
			}
		}
	case msg.OfAssistant != nil:
		if v := msg.OfAssistant.Content.OfString; v.Valid() {
			b.WriteString(v.Value)
		}
		for _, p := range msg.OfAssistant.Content.OfArrayOfContentParts {
			if p.OfText != nil {
				b.WriteString(p.OfText.Text)
			}
		}
		for _, tc := range msg.OfAssistant.ToolCalls {
			if tc.OfFunction != nil {
				b.WriteString(tc.OfFunction.Function.Name)
				b.WriteString(tc.OfFunction.Function.Arguments)
			}
		}
	case msg.OfTool != nil:
		if v := msg.OfTool.Content.OfString; v.Valid() {
			b.WriteString(v.Value)
		}
		for _, p := range msg.OfTool.Content.OfArrayOfContentParts {
			b.WriteString(p.Text)
		}
	}
	return b.String()
}

func estimateTokens(s string) int {
	cjk, other := 0, 0
	for _, r := range s {
		if isCJKHeavy(r) {
			cjk++
		} else {
			other++
		}
	}
	return cjk + (other+3)/4
}

func isCJKHeavy(r rune) bool {
	return (r >= 0x4E00 && r <= 0x9FFF) || // CJK unified ideographs
		(r >= 0x3000 && r <= 0x303F) || // CJK punctuation
		(r >= 0xFF00 && r <= 0xFFEF) // fullwidth forms
}
