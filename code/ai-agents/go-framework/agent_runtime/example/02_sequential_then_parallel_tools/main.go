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
	timestampTool := &agent_runtime.Tool{
		Name:        "get_current_timestamp",
		Description: "获取今天的 Unix 时间戳，必须先于天气查询调用",
		Parameters:  map[string]any{"type": "object", "properties": map[string]any{}, "required": []any{}},
		Func: func(ctx context.Context, args map[string]any) (string, error) {
			timestamp := time.Now().Unix()
			fmt.Printf("第一阶段，获取时间戳：%d\n", timestamp)
			return strconv.FormatInt(timestamp, 10), nil
		},
	}
	weatherTool := &agent_runtime.Tool{
		Name:        "get_weather",
		Description: "根据城市和时间戳查询天气；不同城市可以并行调用",
		Parameters: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"city":      map[string]any{"type": "string", "enum": []any{"杭州", "上海", "北京"}},
				"timestamp": map[string]any{"type": "integer"},
			},
			"required": []any{"city", "timestamp"},
		},
		Func: func(ctx context.Context, args map[string]any) (string, error) {
			city, ok := args["city"].(string)
			if !ok {
				return "", fmt.Errorf("city must be a string")
			}
			timestamp, ok := args["timestamp"].(float64)
			if !ok {
				return "", fmt.Errorf("timestamp must be an integer")
			}
			delays := map[string]time.Duration{"杭州": 500, "上海": 300, "北京": 400}
			weather := map[string][2]string{
				"杭州": {"小雨", "18°C"}, "上海": {"多云", "20°C"}, "北京": {"晴", "22°C"},
			}
			fmt.Printf("并行天气任务开始：%s\n", city)
			time.Sleep(delays[city] * time.Millisecond)
			condition := weather[city]
			result := map[string]any{
				"city":    city,
				"date":    time.Unix(int64(timestamp), 0).Format("2006-01-02"),
				"weather": condition[0], "temperature": condition[1],
			}
			encoded, err := json.Marshal(result)
			fmt.Printf("并行天气任务结束：%s\n", city)
			return string(encoded), err
		},
	}
	agent := agent_runtime.NewAgent(
		loadConfig(),
		[]*agent_runtime.Tool{timestampTool, weatherTool},
	)
	history, err := agent.Loop(context.Background(), []openai.ChatCompletionMessageParamUnion{
		openai.SystemMessage("严格遵守工具编排：第一轮只调用一次 get_current_timestamp。收到时间戳后，第二轮必须在同一条助手消息中同时发起三次 get_weather，分别查询杭州、上海、北京，禁止逐个查询。三个结果都返回后再汇总。"),
		openai.UserMessage("今天杭州、上海、北京的天气各自如何？"),
	})
	if err != nil {
		log.Fatal(err)
	}

	var toolRounds [][]string
	for _, message := range history {
		if message.OfAssistant == nil || len(message.OfAssistant.ToolCalls) == 0 {
			continue
		}
		var names []string
		for _, call := range message.OfAssistant.ToolCalls {
			name := call.OfFunction.Function.Name
			names = append(names, name)
		}
		toolRounds = append(toolRounds, names)
	}
	for index, names := range toolRounds {
		fmt.Printf("第 %d 个工具轮次：%v\n", index+1, names)
	}
	if len(toolRounds) != 2 || len(toolRounds[0]) != 1 || len(toolRounds[1]) != 3 {
		log.Fatal("模型没有遵循先串联、再三路并联的两轮工具编排，请重新运行")
	}
	last := history[len(history)-1]
	fmt.Printf("模型最终回答：\n%s\n", last.OfAssistant.Content.OfString.Value)
}
