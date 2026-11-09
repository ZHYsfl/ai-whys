package agent_runtime

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"runtime/debug"
	"strings"
	"sync"
	"time"

	"github.com/openai/openai-go/v3"
)

// logRound 记一轮 LLM 调用的账：耗时、prompt/completion token 数、结束原因、
// 点名的工具（带首个关键参数的短摘要）。PPT agent 的"改稿为什么慢、刚才在
// 跑什么命令"从这几行直接读——首包延迟 ≈ prompt 长度 × prefill 速度，历史
// 越大每轮越慢。
func logRound(start time.Time, round int, resp *openai.ChatCompletion) {
	msg := resp.Choices[0].Message
	names := make([]string, 0, len(msg.ToolCalls))
	for _, tc := range msg.ToolCalls {
		names = append(names, tc.Function.Name+digestArgs(tc.Function.Arguments))
	}
	log.Printf("llm round %d: %v prompt=%d completion=%d finish=%s tools=%v",
		round, time.Since(start).Round(time.Millisecond),
		resp.Usage.PromptTokens, resp.Usage.CompletionTokens,
		resp.Choices[0].FinishReason, names)
}

// digestArgs 把工具参数压成一截短摘要，只取优先级最高的第一个字符串参数的
// 前 24 个 rune，长参数不刷屏（runes 截断，不在多字节字符中间劈开）。
func digestArgs(raw string) string {
	if raw == "" {
		return ""
	}
	var m map[string]any
	if json.Unmarshal([]byte(raw), &m) != nil {
		return ""
	}
	for _, k := range []string{"command", "path", "message", "content", "old_string", "input_token"} {
		if v, ok := m[k].(string); ok && v != "" {
			v = strings.Join(strings.Fields(v), " ")
			if r := []rune(v); len(r) > 24 {
				v = string(r[:24]) + "…"
			}
			return "(" + v + ")"
		}
	}
	return ""
}

func hasToolErrors(results []ToolResponse) bool {
	for _, r := range results {
		if r.status == "error" {
			return true
		}
	}
	return false
}

func getErrorSummary(results []ToolResponse) string {
	var parts []string
	for _, r := range results {
		if r.status == "error" {
			c := r.content
			if len(c) > 200 {
				c = c[:200]
			}
			errType := "unknown"
			if r.errorType != nil {
				errType = *r.errorType
			}
			parts = append(parts, fmt.Sprintf("- %s: %s", errType, c))
		}
	}
	if len(parts) == 0 {
		return "unknown error"
	}
	return strings.Join(parts, "\n")
}

// AssistantMsgToParam converts a response ChatCompletionMessage into a
// request ChatCompletionMessageParamUnion so it can be appended back to the
// conversation history. This is the Go equivalent of Python's
// response.choices[0].message.model_dump().
//
// tool_calls 按字段重建 function 变体，不走 tc.ToParam()：ToParam 搬的是
// union 反序列化时存下的 JSON.raw，而流式累加器（streamRound 的
// ChatCompletionAccumulator）只逐字段拼、从来不写 raw——对它 ToParam
// 等于抱着空 RawMessage 炸 Marshal（unexpected end of JSON input）。
// 本系统只注册 function 工具，字段重建对两条路径（流式/非流式）语义一致。
func AssistantMsgToParam(msg openai.ChatCompletionMessage) openai.ChatCompletionMessageParamUnion {
	var tcParams []openai.ChatCompletionMessageToolCallUnionParam
	for _, tc := range msg.ToolCalls {
		if tc.Function.Name == "" {
			continue
		}
		tcParams = append(tcParams, openai.ChatCompletionMessageToolCallUnionParam{
			OfFunction: &openai.ChatCompletionMessageFunctionToolCallParam{
				ID: tc.ID,
				Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{
					Name:      tc.Function.Name,
					Arguments: tc.Function.Arguments,
				},
			},
		})
	}

	asst := &openai.ChatCompletionAssistantMessageParam{
		ToolCalls: tcParams,
	}
	if msg.Content != "" {
		asst.Content = openai.ChatCompletionAssistantMessageParamContentUnion{
			OfString: openai.String(msg.Content),
		}
	}
	return openai.ChatCompletionMessageParamUnion{OfAssistant: asst}
}

// panicToolResponse 把一次工具 panic 打包成模型能读懂的 error 工具消息。栈只
// 进服务端日志、不进内容：工具消息会永久留在历史里，每轮重发都要付费。
// 它只负责构造，不含 recover —— recover 必须由 deferred 闭包自己直接调用，
// 隔一层栈帧调它就失效（返回 nil，panic 照旧展开），所以这层包不进函数里。
func panicToolResponse(toolCallID, funcName string, r any) ToolResponse {
	content := fmt.Sprintf("[PANIC] Tool '%s' panicked: %v", funcName, r)
	log.Printf("tool %s panicked (id=%s): %v\n%s", funcName, toolCallID, r, debug.Stack())
	errType := "panic"
	return ToolResponse{
		message:   openai.ToolMessage(content, toolCallID),
		status:    "error",
		errorType: &errType,
		content:   content,
	}
}

func (a *Agent) executeToolCall(
	ctx context.Context,
	toolCall openai.ChatCompletionMessageToolCallUnion,
	availableTools map[string]*Tool,
) (resp ToolResponse) {
	// 与 executeToolCallFunc 同源的守卫，这一层也得挡：下面直接读 union 字段，
	// 不碰 AsFunction()——它只在 API 回来的完整 JSON（非流式）里才有原料，
	// 流式累加器逐字段拼帧、从不写 JSON.raw，AsFunction 只能解出零值。worker
	// agent 的工具轮正是走这条路（host 的 turn → Agent.Loop）。
	defer func() {
		if r := recover(); r != nil {
			resp = panicToolResponse(toolCall.ID, toolCall.Function.Name, r)
		}
	}()

	if toolCall.Type != "function" || toolCall.ID == "" {
		content := "[NOT_FOUND] Unsupported tool call type (only function tool calls are supported)"
		if a.debug {
			fmt.Printf("[ERROR] %s\n", content)
		}
		errType := "not_found"
		return ToolResponse{
			message:   openai.ToolMessage(content, "unsupported-tool-call"),
			status:    "error",
			errorType: &errType,
			content:   content,
		}
	}

	return a.executeToolCallFunc(ctx, toolCall.ID, toolCall.Function.Name, toolCall.Function.Arguments, availableTools)
}

func (a *Agent) executeToolCallFunc(
	ctx context.Context,
	toolCallID string,
	funcName string,
	rawArgs string,
	availableTools map[string]*Tool,
) (resp ToolResponse) {
	// 工具是外部代码（shell、HTTP、MCP、浏览器），它 panic 一次就会带走整个
	// 进程——调用它的 worker goroutine 里的 panic 没有任何别人能 recover。
	// 这里把它收口成一条 error 工具消息：模型看到"这个工具炸了"，正在跑的
	// 语音回合并存活。ExecuteToolCalls 与 getToolResponses 两条并发路径都从
	// 这个函数过，守卫落在这里一处顶两处；上面 executeToolCall 里那层是补它
	// 自己 AsFunction() 的解析动作。
	defer func() {
		if r := recover(); r != nil {
			resp = panicToolResponse(toolCallID, funcName, r)
		}
	}()

	var args map[string]any
	if rawArgs != "" && strings.TrimSpace(rawArgs) != "" && rawArgs != "{}" {
		if err := json.Unmarshal([]byte(rawArgs), &args); err != nil {
			content := fmt.Sprintf("[PARSE_ERROR] JSON parse failed: %v. Raw: '%s'", err, rawArgs)
			if a.debug {
				fmt.Printf("[ERROR] %s\n", content)
			}
			errType := "parse_error"
			return ToolResponse{
				message:   openai.ToolMessage(content, toolCallID),
				status:    "error",
				errorType: &errType,
				content:   content,
			}
		}
	}
	if args == nil {
		args = make(map[string]any)
	}

	tool, ok := availableTools[funcName]
	if !ok {
		content := fmt.Sprintf("[NOT_FOUND] Function '%s' not found", funcName)
		if a.debug {
			fmt.Printf("[ERROR] %s\n", content)
		}
		errType := "not_found"
		return ToolResponse{
			message:   openai.ToolMessage(content, toolCallID),
			status:    "error",
			errorType: &errType,
			content:   content,
		}
	}

	// Validate required arguments against the tool's JSON schema.
	if missing := validateRequiredArgs(tool.Parameters, args); len(missing) > 0 {
		content := fmt.Sprintf("[ARG_ERROR] Missing required arguments for '%s': %v", funcName, missing)
		if a.debug {
			fmt.Printf("[ERROR] %s\n", content)
		}
		errType := "arg_error"
		return ToolResponse{
			message:   openai.ToolMessage(content, toolCallID),
			status:    "error",
			errorType: &errType,
			content:   content,
		}
	}

	if a.debug {
		keys := make([]string, 0, len(args))
		for k := range args {
			keys = append(keys, k)
		}
		fmt.Printf("[DEBUG] Executing %s with args: %v\n", funcName, keys)
	}

	result, err := tool.Func(ctx, args)
	if err != nil {
		content := fmt.Sprintf("[EXEC_ERROR] Execution failed: %v", err)
		if a.debug {
			fmt.Printf("[ERROR] %s\n", content)
		}
		errType := "exec_error"
		return ToolResponse{
			message:   openai.ToolMessage(content, toolCallID),
			status:    "error",
			errorType: &errType,
			content:   content,
		}
	}

	if a.debug {
		fmt.Printf("[DEBUG] Tool %s returned (id=%s): %s\n", funcName, toolCallID, result)
	}

	return ToolResponse{
		message: openai.ToolMessage(result, toolCallID),
		status:  "success",
		content: result,
	}
}

// validateRequiredArgs returns the list of required argument names that are
// missing from the supplied args map. It inspects the "required" field of the
// JSON Schema style parameter map.
func validateRequiredArgs(parameters map[string]any, args map[string]any) []string {
	if parameters == nil {
		return nil
	}
	requiredAny, ok := parameters["required"]
	if !ok {
		return nil
	}
	requiredList, ok := requiredAny.([]any)
	if !ok {
		return nil
	}

	var missing []string
	for _, r := range requiredList {
		name, ok := r.(string)
		if !ok {
			continue
		}
		if _, present := args[name]; !present {
			missing = append(missing, name)
		}
	}
	return missing
}

func (a *Agent) getToolResponses(
	ctx context.Context,
	toolCalls []openai.ChatCompletionMessageToolCallUnion,
) []ToolResponse {
	availableTools := make(map[string]*Tool, len(a.tools))
	for _, t := range a.tools {
		availableTools[t.Name] = t
	}

	results := make([]ToolResponse, len(toolCalls))
	var wg sync.WaitGroup
	for i, tc := range toolCalls {
		wg.Add(1)
		go func(idx int, call openai.ChatCompletionMessageToolCallUnion) {
			defer wg.Done()
			results[idx] = a.executeToolCall(ctx, call, availableTools)
		}(i, tc)
	}
	wg.Wait()
	return results
}

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

// Loop feeds messages through the chat completion, executing tool calls and
// appending their results until the model stops calling tools. On error
// (including ctx cancellation) it still returns the messages accumulated so
// far — input plus any completed tool rounds — so the caller can keep the
// partial turn instead of losing it. beforeRoundHook fires before every LLM
// call (first one included), letting callers stamp each call with "now".
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
