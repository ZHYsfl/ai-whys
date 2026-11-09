我们先介绍一下agent_runtime模块的设计。

今天的agent的policy是由LLM驱动的，关于LLM相关的配置，我们有如下定义：

```go
type LLMConfig struct{
    APIKey string
    BaseURL string
    Model string
    Temperature *float64
    Timeout *time.Duration
    ExtraBody map[string]any
}
```

有了这个配置，我们就可以创造遵循openai协议的客户端了：

```go
func NewOpenAIClient(config *LLMConfig) openai.Client
```

今天的agent的动作空间是各种工具，于是我们定义工具如下：

```go
type ToolFunc func(ctx context.Context,args map[string]any)(string,error)

type Tool struct{
    Name string
    Description string
    Func ToolFunc
    Parameters map[string]any
}
```

每一个工具被调用执行前需要的信息定义为：

```go
type ToolCallInput struct{
    ID string
    Name string
    ArgumentsJSON string
}
```

每一个工具被调用执行后，会返回调用执行的状态、信息和情况，把这些定义为：

```go
type ToolResponse struct{
    message openai.ChatCompletionMessageParamUnion
    content string
    status string // "success" or "error"
    errorType *string // "parse_error", "not_found", "exec_error", "arg_error", "panic"
}
```

今天Agent的观测空间，就是LLM的上下文。我们需要设计记忆机制，去管理这些上下文。记忆机制的策略有很多种，我们实现了两种，算上不设置特殊的记忆机制，一共三种情况：

```go
type MemoryMode int

const (
    MemoryModeNone MemoryMode = iota
    MemoryModeSlideWindow
    MemoryModeCompress
)
```

不同的记忆机制对应不同的上下文管理者，我们统一抽象为：

```go
type MemoryManager interface{
    Prepare(messages []openai.ChatCompletionMessageParamUnion) []openai.ChatCompletionMessageParamUnion
}
```

对于MemoryModeNone策略，其记忆机制管理者为：

```go
type defaultMemory struct{}

func (m *defaulMemory) Prepare(messages []openai.ChatCompletionMessageParamUnion) []openai.ChatCompletionMessageParamUnion{
    return messages
}
```

对于MemoryModeSlideWindow策略，其记忆机制管理者为：

```go
type slideWindowMemory struct{
    maxMessages int
}

func (m *slideWindowMemory) Prepare(messages []openai.ChatCompletionParamUnion) []openai.ChatCompletionMessageParamUnion{
    if m.maxMessages <= 0 {
        return messages
    }
    system, nonSystem := splitSystem(messages)
    if len(nonSystem) > m.maxMessages {
        nonSystem = nonSystem[len(nonSystem)-m.maxMessages:]
    }
    return joinSystem(system,DropLeadingOrphanTools(nonSystem))
}
```

对于MemoryModeCompress策略，其记忆机制管理者为：

```go
const(
    compressTriggerRatio = 0.8
    summaryFoldAt = 5
    compressKeepRecentMessages = 40
    summarizeAttempts = 3
    summarizeAttemptTimeout = 60 * time.Second
    summarizeRetryGap = 500 * time.Millisecond
    summaryOpenTag = "<conversation_summary>"
    summaryCloseTag = "</conversation_summary>"
)

type Summarizer func(ctx context.Context, msgs []openai.ChatCompletionMessageParamUnion) (string, error)

type compressMemory struct{
    maxInputTokens int
    keepRecent int
    summarize Summarizer
    lastPromptTokens atomic.Int64
}

func (m *compressMemory) Prepare(messages []openai.ChatCompletionParamUnion)[]openai.ChatCompletionMessageParamUnion{
    if m.maxInputTokens <= 0 || !m.overThreshold(messages){
        return messages
    }
    return m.compress(messages)
}

func (m* compressMemory) overThreshold(messages []openai.ChatCompletionMessageParamUnion) bool{
    threshold := int64(float64(m.maxInputTokens)*compressTriggerRatio)
    return m.lastPromptTokens.Load() >= threshold ||
        int64(estimateMessages(messages)) >= threshold
}

func (m *compressMemory) compress(messages []openai.ChatCompletionMessageParamUnion) []openai.ChatCompletionMessageParamUnion

type usageRecorder interface{ RecordUsage(promptTokens int64) }

func (m *compressMemory) RecordUsage(promptTokens int64) {
	if promptTokens > 0 {
		m.lastPromptTokens.Store(promptTokens)
	}
}
```

这种策略是：

当这次LLM CALL的输入总token量>=int64(float64(m.maxInputTokens)*compressTriggerRatio)时，触发压缩，压缩保护最近compressKeepRecentMessages条message，以及最远的最多summaryFoldAt-1条曾经的摘要（如果有），如果摘要已经达到summaryFoldAt条，那么不保护这些摘要。系统提示词一定受保护。把不在保护范围内的messages区间进行总结压缩。

把上面的记忆策略、机制相关信息定义为配置：

```go
type MemoryConfig struct{
    Mode MemoryMode
    Max int
    KeepRecent int
    Summarizer Summarizer
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
```

Agent有了上述的LLM配置信息、工具定义和记忆机制，我们终于可以定义Agent：

```go
type Agent struct{
    client openai.Client
    config *LLMConfig

    tools []*Tool
    maxToolRetries int

    memory MemoryManager

    beforeRoundHook func() []openai.ChatCompletionMessageParamUnion

    debug bool
}

func (a *Agent) Prepare(messages []openai.ChatCompletionMessageParamUnion) []openai.ChatCompletionMessageParamUnion {
	return a.memory.Prepare(messages)
}

func (a *Agent) recordUsage(promptTokens int64) {
	if r, ok := a.memory.(usageRecorder); ok {
		r.RecordUsage(promptTokens)
	}
}
```

现在让Agent进行不考虑工具调用的推理：

```go
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
```

考虑工具调用，Agent的一次LLM CALL可能会调用多个工具，所以必须支持并行执行工具的功能（Template 1：fan-out）：

```go
func (a *Agent) ExecuteToolCalls(ctx context.Context,calls []ToolCallInput)[]openai.ChatCompletionMessageParamUnion{
    availableTools := make(map[string]*Tool,len(a.tools))
    for _, t := range a.tools{
        availableTools[t.Name] = t
    }

    results := make([]ToolResponse,len(calls))
    var wg sync.WaitGroup
    for i, c := range calls{
        wg.Add(1)
        go func(idx int, call ToolCallInput){
            defer wg.Done()
            results[idx] = a.executeToolCallFunc(ctx,call.ID,call.Name,call.ArgumentsJSON,availableTools)
        }(i, c)
    }
    wg.Wait()

    msgs := make([]openai.ChatCompletionMessageParamUnion, len(results))
    for i, r := range results {
		msgs[i] = r.message
	}
	return msgs
}
```

Agent的每次Loop是一个状态机：

![agent_runtime](pics/agent_runtime.png)

因此每次Loop为：

```go
// appendProduced 把一条本轮新产生的消息挂上序列，并同步知会 producedHook
// （若注册）。事中记账的意义：记忆压缩会改写序列基座、把产出卷进摘要，
// 调用方事后拿输入长度截"增量"必然截错；事中攒下的产出不受压缩影响。
func (a *Agent) appendProduced(
	next []openai.ChatCompletionMessageParamUnion,
	msg openai.ChatCompletionMessageParamUnion,
) []openai.ChatCompletionMessageParamUnion {
	if a.producedHook != nil {
		a.producedHook(msg)
	}
	return append(next, msg)
}

func (a *Agent) Loop(
	ctx context.Context,
	messages []openai.ChatCompletionMessageParamUnion,
) ([]openai.ChatCompletionMessageParamUnion, error) {
	if a.beforeRoundHook != nil {
		messages = append(messages, a.beforeRoundHook()...)
	}
	params := a.BuildChatCompletionParams(messages, true)

	roundStart := time.Now()
	resp, err := a.roundWithRetry(ctx, params, 1)
	if err != nil {
		// 首次调用就失败：没有产生任何新内容，返回输入本身占位，
		// 调用方按"无增量"处理（len(out) == len(msgs)）。
		return messages, fmt.Errorf("failed to get chat completion: %w", err)
	}
	a.recordUsage(resp.Usage.PromptTokens)
	logRound(roundStart, 1, resp)

	// params.Messages 即 Prepare 后的基座：记忆策略的裁剪/压缩随它永久生效
	// （调用方把 Loop 的返回值原样落账），工具轮在这个基座上继续累积。
	next := make([]openai.ChatCompletionMessageParamUnion, len(params.Messages))
	copy(next, params.Messages)

	retryCount := 0
	round := 1

	for resp.Choices[0].FinishReason == "tool_calls" {
		if ctx.Err() != nil {
			return next, ctx.Err()
		}

		next = a.appendProduced(next, AssistantMsgToParam(resp.Choices[0].Message))

		toolResults := a.getToolResponses(ctx, resp.Choices[0].Message.ToolCalls)
		for _, tr := range toolResults {
			next = a.appendProduced(next, tr.message)
		}

		if hasToolErrors(toolResults) && retryCount < a.maxToolRetries {
			retryCount++
			summary := getErrorSummary(toolResults)
			if a.debug {
				fmt.Printf("[RETRY %d/%d] Tool errors:\n%s\n", retryCount, a.maxToolRetries, summary)
			}
			next = a.appendProduced(next, openai.UserMessage(
				fmt.Sprintf(
					"Tool execution errors detected:\n\n%s\n\nPlease fix and retry. Retries left: %d",
					summary, a.maxToolRetries-retryCount,
				),
			))
		}

		if a.debug {
			jsonMsgs, _ := json.MarshalIndent(next, "", "  ")
			fmt.Printf("[DEBUG] Sending %d messages to LLM:\n%s\n", len(next), string(jsonMsgs))
		}

		if a.beforeRoundHook != nil {
			next = append(next, a.beforeRoundHook()...)
		}
		params.Messages = a.Prepare(next)
		next = params.Messages // 巨型工具结果再次顶爆预算时，二次压缩同样落账
		round++
		roundStart = time.Now()
		resp, err = a.roundWithRetry(ctx, params, round)
		if err != nil {
			return next, fmt.Errorf("chat completion (tool loop): %w", err)
		}
		a.recordUsage(resp.Usage.PromptTokens)
		logRound(roundStart, round, resp)
	}

	next = a.appendProduced(next, AssistantMsgToParam(resp.Choices[0].Message))
	return next, nil
}
```

Loop 的每一次 LLM 调用走的是流式通道而非一次性响应。用流式不是图逐 token——Loop 要的仍是整段应答——而是给"无声卡死"装探测器。动机是一次真实的生产事故（2026-10-04）：API 静默约 4 分钟，非流式调用从外观上无法区分"服务端挂了"与"正常长推理"，worker turn 一路冻结到被用户打断。流式之后，首个 chunk（TTFT）与相邻 chunk 间隔（idle）共用一份时限预算（默认 15s，`LLMConfig.Timeout` 可改），超时即哨兵错误 `errStreamTimeout`；看门狗先于 NewStreaming 上岗，"服务端连响应头都不给"的死法也在覆盖范围内：

```go
func (a *Agent) streamRound(ctx context.Context, params openai.ChatCompletionNewParams) (*openai.ChatCompletion, error) {
	callCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	timeout := a.streamTimeout()
	var lastBeat atomic.Int64 // 最近一帧时间戳（nanos），0 = 一帧都没来
	var timedOut atomic.Bool
	timer := time.NewTimer(timeout)
	disarm := make(chan struct{})
	go func() {
		defer timer.Stop()
		for {
			select {
			case <-timer.C:
				// 到点瞬间刚有帧（迟发竞态）：自首重整，不误报。
				if lastBeat.Load() != 0 && time.Since(time.Unix(0, lastBeat.Load())) < timeout {
					timer.Reset(timeout)
					continue
				}
				timedOut.Store(true)
				cancel() // 掐断底层读取，NewStreaming/Next 脱阻
				return
			case <-disarm:
				return
			case <-callCtx.Done():
				return
			}
		}
	}()

	// 流尾带 usage（include_usage 的真实线形是空 choices 的末帧），记账不断。
	params.StreamOptions.IncludeUsage = openai.Bool(true)
	stream := a.client.Chat.Completions.NewStreaming(callCtx, params)
	defer stream.Close()

	var acc openai.ChatCompletionAccumulator
	for stream.Next() {
		lastBeat.Store(time.Now().UnixNano())
		timer.Reset(timeout) // 每帧重整：首帧超时只是空闲超时的特例
		acc.AddChunk(stream.Current())
	}
	close(disarm)
	// …超时→errStreamTimeout；stream.Err() 原样上抛（ctx 取消也走这里，
	// 调用方按 ctx.Err() 分辨打断与失败）；零 choices 报错。
}
```

只有看门狗超时会触发一次重试（`roundWithRetry`），ctx 取消（用户打断）永远不重试——打断的即时性是全双工的命根子；API 返回了错误响应是确定性失败，同样不重试。重试代价低（prefill 命中 prompt cache），收益高：挂死从"冻结到被打断"变成"超时后自愈"。

流式化带出一个 SDK 深坑：累加器（ChatCompletionAccumulator）只逐字段拼帧，从来不写 union 的 `JSON.raw`，而 `ToParam()`/`AsFunction()` 搬的正是这份 raw——对流式拼出的 tool_calls 调它们等于抱着空 RawMessage 炸 Marshal。所以 `AssistantMsgToParam` 按字段重建 function 变体，`executeToolCall` 直接读 union 的导出字段（ID/Type/Function.Name/Function.Arguments），两处都不碰 raw。本系统只注册 function 工具，字段直读对两条路径（流式/非流式）语义一致。

---

介绍一下tools模块的设计，这个包里提供了许多coding agent需要依赖的tools：

```go
// ExecuteCommand runs a shell command in the given working directory.
// It returns a single string formatted as:
//
//	stdout:<stdout content>
//	stderr:<stderr content>
//
// If the command succeeds and stderr is empty, only the stdout line is returned.
func ExecuteCommand(ctx context.Context, command string, workdir string) (string, error)

// EditFile edits a file by replacing the first occurrence of old_string with new_string.
func EditFile(ctx context.Context, path string, oldString string, newString string) (string, error)



// ReadFile reads a file, optionally restricted to a [start_line, end_line) range (0-based, end-exclusive). end_line <= 0 means "through the last line", start_line < 0 is treated as 0. If max_response_len is 0 or negative, a default cap of 10000 runes is applied to avoid overflowing the LLM context.
func ReadFile(ctx context.Context, path string, startLine int, endLine int, maxResponseLen int) (string, error)

// WriteFile overwrites path with content. The path must already be resolved by the caller (e.g. jailed into a workdir) — this package applies no path policy of its own.
func WriteFile(ctx context.Context, path string, content string) (string, error)


// AppendFile appends content to path, creating the file if absent. Same path-policy contract as WriteFile.
func AppendFile(ctx context.Context, path string, content string) (string, error)
```
