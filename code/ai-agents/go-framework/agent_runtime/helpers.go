package agent_runtime

import "time"

func float64Ptr(f float64) *float64 {
	return &f
}

func intPtr(i int) *int {
	return &i
}

func durationPtr(d time.Duration) *time.Duration {
	return &d
}
