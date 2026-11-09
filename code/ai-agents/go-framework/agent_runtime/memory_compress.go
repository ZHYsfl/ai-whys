package agent_runtime

import (
	"context"
	"log"
	"strings"
	"sync/atomic"
	"time"

	"github.com/openai/openai-go/v3"
)

const (
	compressTriggerRatio       = 0.8
	summaryFoldAt              = 5
	compressKeepRecentMessages = 40
	summarizeAttempts          = 3
	summarizeAttemptTimeout    = 60 * time.Second
	summarizeRetryGap          = 500 * time.Millisecond
)

const (
	summaryOpenTag  = "<conversation_summary>"
	summaryCloseTag = "</conversation_summary>"
)

type compressMemory struct {
	maxInputTokens   int
	keepRecent       int
	summarize        Summarizer
	lastPromptTokens atomic.Int64
}

func (m *compressMemory) RecordUsage(promptTokens int64) {
	if promptTokens > 0 {
		m.lastPromptTokens.Store(promptTokens)
	}
}

func (m *compressMemory) Prepare(messages []openai.ChatCompletionMessageParamUnion) []openai.ChatCompletionMessageParamUnion {
	if m.maxInputTokens <= 0 || !m.overThreshold(messages) {
		return messages
	}
	return m.compress(messages)
}

func (m *compressMemory) overThreshold(messages []openai.ChatCompletionMessageParamUnion) bool {
	threshold := int64(float64(m.maxInputTokens) * compressTriggerRatio)
	return m.lastPromptTokens.Load() >= threshold ||
		int64(estimateMessages(messages)) >= threshold
}

func (m *compressMemory) compress(messages []openai.ChatCompletionMessageParamUnion) []openai.ChatCompletionMessageParamUnion {
	system, nonSystem := splitSystem(messages)
	keep := m.keepRecent
	if keep <= 0 {
		keep = compressKeepRecentMessages
	}

	nSum := countLeadingSummaries(nonSystem)

	regionStart := nSum
	if nSum >= summaryFoldAt {
		regionStart = 0
	}

	cut := len(nonSystem) - keep
	cut = toolGroupSafeCut(nonSystem, regionStart, cut)

	region := nonSystem[regionStart:cut]
	if len(region) == 0 {
		return messages
	}

	tail := DropLeadingOrphanTools(nonSystem[cut:])

	m.lastPromptTokens.Store(0)

	if summary, ok := m.summarizeRegion(system, region); ok {
		out := make([]openai.ChatCompletionMessageParamUnion, 0, regionStart+2+len(tail))
		if system.OfSystem != nil {
			out = append(out, system)
		}
		out = append(out, nonSystem[:regionStart]...)
		out = append(out, summaryMessage(summary))
		return append(out, tail...)
	}

	out := make([]openai.ChatCompletionMessageParamUnion, 0, 1+nSum+len(tail))
	if system.OfSystem != nil {
		out = append(out, system)
	}
	out = append(out, nonSystem[:nSum]...)
	return append(out, tail...)
}

func toolGroupSafeCut(msgs []openai.ChatCompletionMessageParamUnion, lo, cut int) int {
	if cut <= lo {
		return lo
	}
	j := cut - 1
	for j > lo && msgs[j].OfTool != nil {
		j--
	}
	if msgs[j].OfAssistant != nil && len(msgs[j].OfAssistant.ToolCalls) > 0 {
		return j
	}
	if j < cut-1 {
		return j + 1
	}
	return cut
}

func (m *compressMemory) summarizeRegion(
	system openai.ChatCompletionMessageParamUnion,
	region []openai.ChatCompletionMessageParamUnion,
) (string, bool) {
	if m.summarize == nil || system.OfSystem == nil || !system.OfSystem.Content.OfString.Valid() {
		return "", false
	}
	req := make([]openai.ChatCompletionMessageParamUnion, 0, len(region)+1)
	req = append(req, system)
	req = append(req, region...)

	var got string
	var err error
	for attempt := 1; attempt <= summarizeAttempts; attempt++ {
		if attempt > 1 {
			time.Sleep(summarizeRetryGap)
		}
		ctx, cancel := context.WithTimeout(context.Background(), summarizeAttemptTimeout)
		got, err = m.summarize(ctx, req)
		cancel()
		if err == nil && got != "" {
			return got, true
		}
	}
	log.Printf("memory: summarize failed %d times (%v), falling back to slide window", summarizeAttempts, err)
	return "", false
}

func summaryMessage(text string) openai.ChatCompletionMessageParamUnion {
	return openai.UserMessage(summaryOpenTag + "\n" + text + "\n" + summaryCloseTag)
}

func IsSummaryMessage(msg openai.ChatCompletionMessageParamUnion) bool {
	return msg.OfUser != nil &&
		msg.OfUser.Content.OfString.Valid() &&
		strings.HasPrefix(msg.OfUser.Content.OfString.Value, summaryOpenTag)
}

func countLeadingSummaries(msgs []openai.ChatCompletionMessageParamUnion) int {
	i := 0
	for i < len(msgs) && IsSummaryMessage(msgs[i]) {
		i++
	}
	return i
}
