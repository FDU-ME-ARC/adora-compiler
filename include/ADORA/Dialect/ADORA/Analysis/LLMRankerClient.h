//===----------------------------------------------------------------------===//
// LLMRankerClient.h — 通用 LLM ranker 调用层
//
// 只做一件事：把一个 JSON 请求字符串发给外部 ranker 子进程，收回原始响应字符串。
// 不认识任何决策字段（dep_type / tile_set 等），由调用方自己解析 raw.response。
//
// 使用方式：
//   RankerRawResult raw = callRanker(cmd, requestJson, timeoutMs);
//   if (!raw.ok) { /* 用 raw.error 判断失败原因，回退保守默认 */ }
//   // 自己解析 raw.response 里的决策字段
//   // raw.scratchpad 是 LLM 的推理日志，可记录到 log
//
// 内部实现：fork + exec + 双向 pipe + poll 超时（镜像 online_ranker.cpp::rankOnce）。
// 凭据（API key / endpoint）走子进程继承的环境变量，不入此文件。
//===----------------------------------------------------------------------===//
#ifndef ADORA_DIALECT_ADORA_ANALYSIS_LLMRANKERCLIENT_H_
#define ADORA_DIALECT_ADORA_ANALYSIS_LLMRANKERCLIENT_H_

#include "llvm/ADT/StringRef.h"
#include <string>

namespace mlir {
namespace ADORA {

/// callRanker 的返回值。
/// ok=true 时 response 是 LLM 的原始输出字符串，scratchpad 是推理日志（可能为空）。
/// ok=false 时 error 描述失败原因（超时 / exec 失败 / 管道错误）。
struct RankerRawResult {
  bool        ok        = false;
  std::string response;    // LLM 原始响应（JSON 字符串）
  std::string scratchpad;  // LLM 推理日志（{"scratchpad":"..."} 字段，通用）
  std::string error;       // 失败原因（ok=false 时有意义）
};

/// 调用外部 LLM ranker 子进程。
/// @param cmd        ranker 命令行（如 "python3 task_schedule_ranker.py --backend openai ..."）
/// @param requestJson 要写入子进程 stdin 的完整 JSON 请求字符串
/// @param timeoutMs  等待子进程响应的超时毫秒数（超时后 SIGKILL 子进程）
/// @return RankerRawResult，调用方自行解析 response 里的决策字段
RankerRawResult callRanker(llvm::StringRef cmd,
                           const std::string &requestJson,
                           int timeoutMs);

} // namespace ADORA
} // namespace mlir

#endif // ADORA_DIALECT_ADORA_ANALYSIS_LLMRANKERCLIENT_H_
