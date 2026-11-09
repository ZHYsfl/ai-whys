package agent_runtime

import "time"

type LLMConfig struct {
	APIKey      string
	BaseURL     string
	Model       string
	Temperature *float64
	Timeout     *time.Duration
	ExtraBody   map[string]any
}
