//===------------------ pipeline_scheduler.h ----------------------===//
//
// runtime-online-v0 §2 "排流水" (pipeline scheduling) decision API.
//
// This is the single entry point every lowering pass uses when it wants to
// consult the online ranker about a pipeline-schedule knob. It packages
// three orthogonal decisions -- stationary kind, double-buffering, and
// prefetch depth -- into a single typed request so call sites do not have
// to build JSON by hand, and so AgentTrace records are emitted in a
// uniform shape across op kinds (Gemm, Conv, future fused kernels).
//
// Design rationale: see Agent-Compiler-notes/MainLine/pipeline_schedule_api.md
//
//===-------------------------------------------------------------===//
#ifndef __PIPELINE_SCHEDULER_H__
#define __PIPELINE_SCHEDULER_H__

#include <cstdint>
#include <string>
#include <vector>

namespace mlir {
namespace ADORA {

// One candidate pipeline schedule. The compiler builds a vector of these
// with index 0 pinned to the default (= whatever attributes the op already
// carries); alternates follow in a deterministic order.
struct PipelineCandidate {
    std::string stationary_kind;        // "WeightStationary" / ...
    bool        double_buffer = true;   // drives ADORA::setPingpongAttr
    int         prefetch_depth = 1;     // advisory in v0
};

struct PipelineScheduleRequest {
    std::string              op_kind;     // "gemm" / "conv_direct" / "conv_im2col"
    std::vector<int64_t>     tile_size;   // compiler-selected tile, for context
    std::vector<std::string> loop_order;  // optional context, not selected here
    std::vector<PipelineCandidate> candidates; // [0] = default, size >= 1
};

// Outcome of a scheduling call. When the ranker is disabled, or fails, the
// decision still resolves to candidate 0 so lowering can proceed safely.
struct PipelineScheduleDecision {
    int  applied_index    = 0;
    bool ranker_used      = false;
    bool ranker_succeeded = false;
    bool used_fallback    = false;
    int  selected_index_raw = 0;   // as returned by ranker, may be out of range
    std::string rationale;
    std::string error;

    // Candidate that should actually be applied by the call site.
    const PipelineCandidate* applied = nullptr;
};

// Build the JSON request, call OnlineRanker::rank(), emit the AgentTrace
// record, and return a typed decision. If the request carries fewer than
// two candidates the helper short-circuits to candidate 0 without
// consulting the ranker (there is nothing to choose).
PipelineScheduleDecision
schedulePipeline(const PipelineScheduleRequest& req);

}  // namespace ADORA
}  // namespace mlir

#endif  // __PIPELINE_SCHEDULER_H__
