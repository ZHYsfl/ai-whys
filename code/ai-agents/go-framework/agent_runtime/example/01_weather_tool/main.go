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

func main() {
	getWeather := &agent_runtime.Tool{
		Name:        "get_weather",
		Description: "查询指定城市的当前天气",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"city": map[string]any{"type": "string"},
			},
			"required": []any{"city"},
		},
		Func: func(ctx context.Context, args map[string]any) (string, error) {
			result := map[string]any{
				"city": args["city"], "weather": "晴", "temperature": "22°C",
			}
			fmt.Printf("工具确实执行了：%v\n", result)
			encoded, err := json.Marshal(result)
			return string(encoded), err
		},
	}
	agent := agent_runtime.NewAgent(loadConfig(), []*agent_runtime.Tool{getWeather})
	history, err := agent.Loop(context.Background(), []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("请调用 get_weather 查询北京天气，再告诉我结果。"),
	})
	if err != nil {
		log.Fatal(err)
	}

	for _, message := range history {
		if message.OfTool != nil {
			fmt.Printf("工具返回消息：%s\n", message.OfTool.Content.OfString.Value)
		}
	}
	last := history[len(history)-1]
	if last.OfAssistant != nil {
		fmt.Printf("模型最终回答：%s\n", last.OfAssistant.Content.OfString.Value)
	}
}
