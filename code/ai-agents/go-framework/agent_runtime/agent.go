package agent_runtime

import (
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/shared"
)

type Agent struct {
	client         openai.Client
	config         *LLMConfig
	tools          []*Tool
	debug          bool
	maxToolRetries int
	// beforeRoundHook 非 nil 时，Loop 在每次 LLM 调用前追加它返回的消息
	// （时间条等"此刻"信息），让长回合内的每轮调用都看得见当下时刻。
	beforeRoundHook func() []openai.ChatCompletionMessageParamUnion
	// producedHook 非 nil 时，Loop 每把一条本轮新产生的消息（assistant
	// 回复、工具结果、重试提示）append 上序列就同步回调一次。事中记账：
	// 记忆压缩会改写序列基座、把产出卷进摘要，调用方事后拿输入长度截
	// "增量"必然截错；事中攒下的产出清单不受压缩影响。
	producedHook func(openai.ChatCompletionMessageParamUnion)
	// memory 是唯一的记忆出口：策略、预算、摘要器都收在它自己的配置里。
	memory MemoryManager
}

type AgentOption func(*Agent)

func WithDebug(debug bool) AgentOption {
	return func(a *Agent) {
		a.debug = debug
	}
}

// WithBeforeRoundHook 注入"每次 LLM 调用前的附加消息"（每次调用都触发，
// 包括工具轮）。调用方返回的消息会被原样追加到输入末尾。
func WithBeforeRoundHook(f func() []openai.ChatCompletionMessageParamUnion) AgentOption {
	return func(a *Agent) { a.beforeRoundHook = f }
}

// WithProducedCallback 注册"产出"回调：Loop 每产生一条新消息（assistant
// 回复、工具结果、重试提示）就同步调用。注入物（hook 时间戳等）不触发。
func WithProducedCallback(f func(openai.ChatCompletionMessageParamUnion)) AgentOption {
	return func(a *Agent) { a.producedHook = f }
}

func WithMaxToolRetries(maxToolRetries int) AgentOption {
	return func(a *Agent) {
		a.maxToolRetries = maxToolRetries
	}
}

func NewAgent(config *LLMConfig, tools []*Tool, opts ...AgentOption) *Agent {
	client := NewOpenAIClient(config)

	agent := &Agent{
		client:         client,
		tools:          tools,
		config:         config,
		debug:          false,
		maxToolRetries: 3,
		memory:         &defaultMemory{},
	}

	for _, opt := range opts {
		opt(agent)
	}

	if agent.memory == nil {
		agent.memory = &defaultMemory{}
	}

	return agent
}

func (a *Agent) AddTool(tool *Tool) {
	a.tools = append(a.tools, tool)
}

func (a *Agent) RemoveTool(name string) {
	filtered := make([]*Tool, 0, len(a.tools))
	for _, t := range a.tools {
		if t.Name != name {
			filtered = append(filtered, t)
		}
	}
	a.tools = filtered
}

func (a *Agent) GetTools() []openai.ChatCompletionToolUnionParam {
	out := make([]openai.ChatCompletionToolUnionParam, len(a.tools))
	for i, t := range a.tools {
		out[i] = openai.ChatCompletionFunctionTool(shared.FunctionDefinitionParam{
			Name:        t.Name,
			Description: openai.String(t.Description),
			Parameters:  shared.FunctionParameters(t.Parameters),
		})
	}
	return out
}

// Prepare applies the configured memory manager (e.g. slide window) to a
// message list; used by streaming paths that bypass BuildChatCompletionParams.
func (a *Agent) Prepare(messages []openai.ChatCompletionMessageParamUnion) []openai.ChatCompletionMessageParamUnion {
	return a.memory.Prepare(messages)
}

// recordUsage feeds the real token usage of the last response back into the
// memory: budget-based memories anchor their next estimate on it instead of
// re-estimating the whole history.
func (a *Agent) recordUsage(promptTokens int64) {
	if r, ok := a.memory.(usageRecorder); ok {
		r.RecordUsage(promptTokens)
	}
}

// RecordUsage feeds an externally observed prompt token count into the memory policy.
func (a *Agent) RecordUsage(promptTokens int64) {
	a.recordUsage(promptTokens)
}
