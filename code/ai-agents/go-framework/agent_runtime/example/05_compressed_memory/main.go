package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strconv"
	"time"

	"agent_runtime"
	"github.com/joho/godotenv"
	"github.com/openai/openai-go/v3"
)

func mustEnv(name string) string {
	value := os.Getenv(name)
	if value == "" {
		log.Fatalf("missing environment variable: %s", name)
	}
	return value
}

func loadConfig() *agent_runtime.LLMConfig {
	_ = godotenv.Load()
	temperature, err := strconv.ParseFloat(mustEnv("DEEPSEEK_TEMPERATURE"), 64)
	if err != nil {
		log.Fatal(err)
	}
	timeout, err := strconv.ParseFloat(mustEnv("DEEPSEEK_TIMEOUT"), 64)
	if err != nil {
		log.Fatal(err)
	}
	duration := time.Duration(timeout * float64(time.Second))
	return &agent_runtime.LLMConfig{
		APIKey:      mustEnv("DEEPSEEK_API_KEY"),
		BaseURL:     mustEnv("DEEPSEEK_BASEURL"),
		Model:       mustEnv("DEEPSEEK_MODEL"),
		Temperature: &temperature,
		Timeout:     &duration,
	}
}

func show(title string, messages []openai.ChatCompletionMessageParamUnion) {
	fmt.Printf("\n%s\n", title)
	encoded, _ := json.MarshalIndent(messages, "", "  ")
	fmt.Println(string(encoded))
}

func summaryMessage(text string) openai.ChatCompletionMessageParamUnion {
	return openai.UserMessage("<conversation_summary>\n" + text + "\n</conversation_summary>")
}

func newCompressAgent(
	config *agent_runtime.LLMConfig,
	maxTokens int,
	keepRecent int,
	summarizer agent_runtime.Summarizer,
) *agent_runtime.Agent {
	return agent_runtime.NewAgent(
		config,
		nil,
		agent_runtime.WithMemory(agent_runtime.MemoryConfig{
			Mode: agent_runtime.MemoryModeCompress, Max: maxTokens, KeepRecent: keepRecent, Summarizer: summarizer,
		}),
	)
}

func main() {
	config := loadConfig()
	client := agent_runtime.NewOpenAIClient(config)
	summarize := agent_runtime.Summarizer(func(
		ctx context.Context,
		messages []openai.ChatCompletionMessageParamUnion,
	) (string, error) {
		encoded, _ := json.Marshal(messages)
		response, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
			Model:       openai.ChatModel(config.Model),
			Temperature: openai.Float(0),
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.SystemMessage("你是中文对话摘要器，保留目标、事实、工具结果和待办事项。"),
				openai.UserMessage("请简洁总结以下消息：\n" + string(encoded)),
			},
		})
		if err != nil {
			return "", err
		}
		if len(response.Choices) == 0 || response.Choices[0].Message.Content == "" {
			return "", fmt.Errorf("模型返回了空摘要")
		}
		return response.Choices[0].Message.Content, nil
	})

	fmt.Println("触发比例=0.8，摘要折叠点=5，摘要最多尝试=3 次，单次超时=60s，重试间隔=500ms")
	system := openai.SystemMessage("你是中文技术学习助手。")
	small := []openai.ChatCompletionMessageParamUnion{system, openai.UserMessage("短消息")}
	disabled := newCompressAgent(config, 0, 0, summarize).Prepare(small)
	belowThreshold := newCompressAgent(config, 10000, 0, summarize).Prepare(small)
	fmt.Printf("max<=0 时禁用：%t\n", len(disabled) == len(small))
	fmt.Printf("低于 80%% 阈值时不压缩：%t\n", len(belowThreshold) == len(small))

	assistantCall := openai.ChatCompletionMessageParamUnion{
		OfAssistant: &openai.ChatCompletionAssistantMessageParam{
			ToolCalls: []openai.ChatCompletionMessageToolCallUnionParam{
				{
					OfFunction: &openai.ChatCompletionMessageFunctionToolCallParam{
						ID: "call-1",
						Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{
							Name: "get_weather", Arguments: "{\"city\":\"北京\"}",
						},
					},
				},
			},
		},
	}
	toolResult := openai.ToolMessage("{\"weather\":\"晴\",\"temperature\":\"22°C\"}", "call-1")
	recentOne := openai.UserMessage("最近消息一：保留工具调用和结果的对应关系。")
	recentTwo := openai.UserMessage("最近消息二：明白。")
	messages := []openai.ChatCompletionMessageParamUnion{
		system,
		openai.UserMessage("我的学习目标是掌握 Agent 的工具调用与上下文压缩。"),
		openai.UserMessage("我会保留这个目标，并观察旧消息如何被摘要。"),
		openai.UserMessage("请查询北京天气。"),
		assistantCall,
		toolResult,
		openai.UserMessage("这条消息位于工具组之后，也需要按安全切点处理。"),
		recentOne,
		recentTwo,
	}
	keepRecent := 3
	rawCut := len(messages[1:]) - keepRecent
	fmt.Printf("\n基础压缩：阈值=80，原始切点=%d，工具组安全切点=3\n", rawCut)
	basicAgent := newCompressAgent(config, 100, keepRecent, summarize)
	basicResult := basicAgent.Prepare(messages)
	show("system 保留，旧区间变成摘要，工具组完整留在尾部：", basicResult)
	fmt.Printf("新消息是摘要：%t\n", agent_runtime.IsSummaryMessage(basicResult[1]))

	twoSummaries := []openai.ChatCompletionMessageParamUnion{
		summaryMessage("旧摘要一"), summaryMessage("旧摘要二"),
	}
	protectedMessages := []openai.ChatCompletionMessageParamUnion{
		system,
		twoSummaries[0], twoSummaries[1],
		openai.UserMessage("需要并入新摘要的旧消息。"),
		openai.UserMessage("旧回复。"),
		openai.UserMessage("最新消息。"),
	}
	protectedResult := newCompressAgent(config, 100, 1, summarize).Prepare(protectedMessages)
	show("少于五个旧摘要时，旧摘要继续受保护：", protectedResult)
	fmt.Printf("摘要数量：%d\n", countSummaries(protectedResult))

	foldingMessages := []openai.ChatCompletionMessageParamUnion{system}
	for index := 1; index <= 5; index++ {
		foldingMessages = append(foldingMessages, summaryMessage(fmt.Sprintf("第 %d 个旧摘要", index)))
	}
	foldingMessages = append(
		foldingMessages,
		openai.UserMessage("这一阶段学习摘要折叠。"),
		openai.UserMessage("五个旧摘要不再受保护。"),
		openai.UserMessage("保留这条最新消息。"),
	)
	foldingResult := newCompressAgent(config, 100, 1, summarize).Prepare(foldingMessages)
	show("达到五个旧摘要时，它们全部参与折叠：", foldingResult)
	fmt.Printf("折叠后摘要数：%d\n", countSummaries(foldingResult))

	usageMessages := []openai.ChatCompletionMessageParamUnion{
		system,
		openai.UserMessage("短旧消息"),
		openai.UserMessage("短旧回复"),
		openai.UserMessage("最新消息"),
	}
	usageAgent := newCompressAgent(config, 1000, 1, summarize)
	usageAgent.RecordUsage(800)
	usageResult := usageAgent.Prepare(usageMessages)
	fmt.Printf("\nrecord_usage(800) 可以独立触发压缩：%t\n", agent_runtime.IsSummaryMessage(usageResult[1]))

	noRegionAgent := newCompressAgent(config, 1000, 40, summarize)
	noRegionAgent.RecordUsage(800)
	noRegionResult := noRegionAgent.Prepare(small)
	fmt.Printf("超过阈值但没有可压缩区间时原样返回：%t\n", len(noRegionResult) == len(small))

	attempts := 0
	retrySummarizer := agent_runtime.Summarizer(func(
		ctx context.Context,
		messages []openai.ChatCompletionMessageParamUnion,
	) (string, error) {
		summary, err := summarize(ctx, messages)
		attempts++
		fmt.Printf("真实摘要调用第 %d 次\n", attempts)
		if err != nil {
			return "", err
		}
		if attempts <= 5 {
			return "", fmt.Errorf("演示主动丢弃第 %d 次摘要结果", attempts)
		}
		return summary, nil
	})
	retryMessages := []openai.ChatCompletionMessageParamUnion{
		system,
		openai.UserMessage("旧消息一：需要摘要。"),
		openai.UserMessage("旧消息二：包含重要上下文。"),
		openai.UserMessage("最新消息：必须保留。"),
	}
	retryAgent := newCompressAgent(config, 80, 1, retrySummarizer)
	fallbackResult := retryAgent.Prepare(retryMessages)
	show("前三次摘要失败，降级为滑动窗口（system + 最新消息）：", fallbackResult)

	retryAgent.RecordUsage(80)
	retryResult := retryAgent.Prepare(retryMessages)
	show("再次触发后，第六次真实摘要成功，生成摘要消息：", retryResult)
	fmt.Printf("总尝试次数=%d，摘要成功=%t\n", attempts, agent_runtime.IsSummaryMessage(retryResult[1]))

	noSummarizerAgent := newCompressAgent(config, 100, 1, nil)
	noSummarizerAgent.RecordUsage(80)
	show("未配置 summarizer 时也使用同样的降级策略：", noSummarizerAgent.Prepare(retryMessages))

	noSystemMessages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("旧消息"), openai.UserMessage("旧回复"), openai.UserMessage("最新消息"),
	}
	noSystemResult := newCompressAgent(config, 20, 1, summarize).Prepare(noSystemMessages)
	show("没有字符串 system 提示时不调用摘要器，直接降级：", noSystemResult)
}

func countSummaries(messages []openai.ChatCompletionMessageParamUnion) int {
	count := 0
	for _, message := range messages {
		if agent_runtime.IsSummaryMessage(message) {
			count++
		}
	}
	return count
}
