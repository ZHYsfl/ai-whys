package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
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

func printWindow(
	agent *agent_runtime.Agent,
	messages []openai.ChatCompletionMessageParamUnion,
	title string,
) []openai.ChatCompletionMessageParamUnion {
	prepared := agent.Prepare(messages)
	fmt.Printf("\n%s：system + 最近 %d 条非 system 消息\n", title, len(prepared)-1)
	encoded, _ := json.MarshalIndent(prepared, "", "  ")
	fmt.Println(string(encoded))
	return prepared
}

func main() {
	config := loadConfig()
	agent := agent_runtime.NewAgent(
		config,
		nil,
		agent_runtime.WithMemory(agent_runtime.MemoryConfig{
			Mode: agent_runtime.MemoryModeSlideWindow,
			Max:  4,
		}),
	)
	history := []openai.ChatCompletionMessageParamUnion{
		openai.SystemMessage("你是记忆演示助手。只能根据当前请求里实际存在的对话回答，不知道就明确说不知道。"),
	}
	questions := []string{
		"请记住：我的幸运数字是 31415。只回复“已经记住”。",
		"我们换一个话题：用一句话解释滑动窗口。",
		"再换一个话题：用一句话解释工具调用。",
		"我的幸运数字是多少？不要猜测。",
	}

	for index, question := range questions {
		outgoing := make([]openai.ChatCompletionMessageParamUnion, len(history), len(history)+1)
		copy(outgoing, history)
		outgoing = append(outgoing, openai.UserMessage(question))
		prepared := printWindow(agent, outgoing, fmt.Sprintf("第 %d 轮实际发送窗口", index+1))
		if index == len(questions)-1 {
			encoded, _ := json.Marshal(prepared)
			if strings.Contains(string(encoded), "31415") {
				log.Fatal("幸运数字仍在窗口中")
			}
			fmt.Println("最早包含幸运数字的消息已经滑出窗口。")
		}
		next, answer, err := agent.ChatWithoutToolCall(context.Background(), outgoing)
		if err != nil {
			log.Fatal(err)
		}
		history = next
		fmt.Printf("模型回答：%s\n", answer)
	}

	assistantCall := openai.ChatCompletionMessageParamUnion{
		OfAssistant: &openai.ChatCompletionAssistantMessageParam{
			ToolCalls: []openai.ChatCompletionMessageToolCallUnionParam{
				{
					OfFunction: &openai.ChatCompletionMessageFunctionToolCallParam{
						ID: "call-1",
						Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{
							Name: "get_weather", Arguments: "{}",
						},
					},
				},
			},
		},
	}
	orphanDemo := []openai.ChatCompletionMessageParamUnion{
		history[0], assistantCall, openai.ToolMessage("旧结果", "call-1"), openai.UserMessage("最新消息"),
	}
	orphanAgent := agent_runtime.NewAgent(
		config,
		nil,
		agent_runtime.WithMemory(agent_runtime.MemoryConfig{
			Mode: agent_runtime.MemoryModeSlideWindow,
			Max:  2,
		}),
	)
	fmt.Println("\n窗口若从孤立 tool 消息开始，会继续丢弃它：")
	encoded, _ := json.MarshalIndent(orphanAgent.Prepare(orphanDemo), "", "  ")
	fmt.Println(string(encoded))
}
