package agent_runtime

import (
	"context"
	"errors"
	"fmt"
	"log"
	"sync/atomic"
	"time"

	"github.com/openai/openai-go/v3"
)

// defaultStreamTimeout 是流式看门狗的默认时限：首个 chunk（TTFT）与相邻
// chunk 间隔（idle）共用这一个预算。它不是"慢一点"的容忍度，是"连接死了"
// 的探测器——2026-10-04 生产事故：API 静默约 4 分钟，非流式调用从外观上
// 无法区分"服务端挂了"与"正常长推理"，worker turn 一路冻结到被用户打断。
const defaultStreamTimeout = 15 * time.Second

// errStreamTimeout 是看门狗超时的哨兵：只有它能触发重试；ctx 取消（用户
// 打断）永远不重试——打断的即时性是全双工的命根子。
var errStreamTimeout = errors.New("llm stream: watchdog timeout")

// streamTimeout 读配置：LLMConfig.Timeout 设了用设的，否则默认 15s。
func (a *Agent) streamTimeout() time.Duration {
	if a.config.Timeout != nil && *a.config.Timeout > 0 {
		return *a.config.Timeout
	}
	return defaultStreamTimeout
}

// roundWithRetry 跑一次流式 round；看门狗超时（且不是用户取消）时重试一次。
// 重试代价低（prefill 命中 prompt cache），收益高：挂死从"冻结到被打断"
// 变成"超时后自愈"。API 返回了错误响应是确定性失败，不在重试之列。
func (a *Agent) roundWithRetry(ctx context.Context, params openai.ChatCompletionNewParams, round int) (*openai.ChatCompletion, error) {
	resp, err := a.streamRound(ctx, params)
	if errors.Is(err, errStreamTimeout) && ctx.Err() == nil {
		log.Printf("llm round %d: %v, retrying once", round, err)
		resp, err = a.streamRound(ctx, params)
	}
	return resp, err
}

// streamRound 发起一次流式调用并把 chunk 拼回完整响应。用流式不是图逐
// token——Loop 要的仍是整段应答——而是给"无声卡死"装探测器：首个 chunk
// 或相邻 chunk 间隔超过时限就掐断流、报 errStreamTimeout。看门狗先于
// NewStreaming 上岗：服务端连响应头都不给的死法也在覆盖范围内。
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

	if timedOut.Load() {
		return nil, errStreamTimeout
	}
	if err := stream.Err(); err != nil {
		return nil, err // ctx 取消也走这里；调用方按 ctx.Err() 分辨打断与失败
	}
	if len(acc.Choices) == 0 {
		return nil, fmt.Errorf("llm stream: no choices returned")
	}
	return &acc.ChatCompletion, nil
}
