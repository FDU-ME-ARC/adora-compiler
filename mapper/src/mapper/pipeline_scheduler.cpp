//===------------------ pipeline_scheduler.cpp ------------------===//
//
// Implementation of the runtime-online-v0 §2 pipeline-schedule decision
// API. See mapper/include/mapper/pipeline_scheduler.h and
// Agent-Compiler-notes/MainLine/pipeline_schedule_api.md.
//
//===------------------------------------------------------------===//
#include "mapper/pipeline_scheduler.h"

#include "mapper/agent_trace.h"
#include "mapper/online_ranker.h"

#include <sstream>
#include <string>

namespace mlir {
namespace ADORA {

namespace {

// Return true if every candidate past index 0 differs ONLY in
// stationary_kind (i.e. double_buffer / prefetch_depth match the default).
// Used to preserve legacy phase-name "pipeline_policy_select" so old
// replay artefacts stay readable.
bool isLegacyPolicyOnly(const PipelineScheduleRequest& req) {
    if (req.candidates.empty()) return true;
    const auto& def = req.candidates.front();
    for (size_t i = 1; i < req.candidates.size(); ++i) {
        const auto& c = req.candidates[i];
        if (c.double_buffer  != def.double_buffer)  return false;
        if (c.prefetch_depth != def.prefetch_depth) return false;
    }
    return true;
}

std::string buildRequestJson(const PipelineScheduleRequest& req,
                             const std::string& phase) {
    std::ostringstream os;
    os << "{\"schema_version\":\"runtime-online-v0\","
       << "\"phase\":\"" << phase << "\","
       << "\"request_id\":\"" << agentTraceJsonEscape(AgentTrace::runId())
       << ":" << phase << ":" << agentTraceJsonEscape(req.op_kind)
       << ":" << OnlineRanker::budgetUsed() << "\","
       << "\"op_kind\":\"" << agentTraceJsonEscape(req.op_kind) << "\","
       << "\"tile_size\":[";
    for (size_t i = 0; i < req.tile_size.size(); ++i) {
        if (i) os << ",";
        os << req.tile_size[i];
    }
    os << "]";
    if (!req.loop_order.empty()) {
        os << ",\"loop_order\":[";
        for (size_t i = 0; i < req.loop_order.size(); ++i) {
            if (i) os << ",";
            os << "\"" << agentTraceJsonEscape(req.loop_order[i]) << "\"";
        }
        os << "]";
    }
    const bool legacy = (phase == "pipeline_policy_select");
    const char* actionType =
        legacy ? "select_pipeline_policy" : "select_pipeline_schedule";
    os << ",\"legal_candidates\":[";
    for (size_t i = 0; i < req.candidates.size(); ++i) {
        const auto& c = req.candidates[i];
        if (i) os << ",";
        os << "{\"action_type\":\"" << actionType << "\",\"payload\":{"
           << "\"candidate_index\":" << i
           << ",\"stationary_kind\":\""
           << agentTraceJsonEscape(c.stationary_kind) << "\"";
        if (!legacy) {
            os << ",\"double_buffer\":"
               << (c.double_buffer ? "true" : "false")
               << ",\"prefetch_depth\":" << c.prefetch_depth;
        }
        os << ",\"is_default\":" << (i == 0 ? "true" : "false")
           << "}}";
    }
    os << "]";
    auto adgCtx = OnlineRanker::adgContext();
    if (!adgCtx.adg_hash.empty()) {
        os << ",\"adg_ref\":{\"adg_hash\":\""
           << agentTraceJsonEscape(adgCtx.adg_hash) << "\"}";
    }
    os << "}";
    return os.str();
}

void emitTrace(const PipelineScheduleRequest& req,
               const std::string& phase,
               const PipelineScheduleDecision& d) {
    if (!AgentTrace::enabled()) return;
    const auto& applied = req.candidates[d.applied_index];
    std::ostringstream os;
    os << "{\"op_kind\":\"" << agentTraceJsonEscape(req.op_kind) << "\""
       << ",\"used\":"      << (d.ranker_used      ? "true" : "false")
       << ",\"succeeded\":" << (d.ranker_succeeded ? "true" : "false")
       << ",\"selected_index\":" << d.selected_index_raw
       << ",\"applied_index\":"  << d.applied_index
       << ",\"default_policy\":\""
       << agentTraceJsonEscape(req.candidates.front().stationary_kind) << "\""
       << ",\"applied_policy\":\""
       << agentTraceJsonEscape(applied.stationary_kind) << "\""
       << ",\"applied_double_buffer\":"
       << (applied.double_buffer ? "true" : "false")
       << ",\"applied_prefetch_depth\":" << applied.prefetch_depth
       << ",\"used_fallback\":" << (d.used_fallback ? "true" : "false")
       << ",\"advisory_only\":false"
       << ",\"rationale\":\"" << agentTraceJsonEscape(d.rationale) << "\""
       << ",\"error\":\""     << agentTraceJsonEscape(d.error)     << "\"}";
    AgentTrace::emit(phase, "online_decision", os.str());
}

}  // namespace

PipelineScheduleDecision
schedulePipeline(const PipelineScheduleRequest& req) {
    PipelineScheduleDecision d;
    if (req.candidates.empty()) {
        d.error = "no_candidates";
        return d;
    }
    d.applied = &req.candidates.front();

    // Nothing to choose: short-circuit to the default.
    if (req.candidates.size() < 2 || !OnlineRanker::enabled()) {
        return d;
    }

    const std::string phase =
        isLegacyPolicyOnly(req) ? "pipeline_policy_select"
                                : "pipeline_schedule_select";

    const std::string body = buildRequestJson(req, phase);
    auto rankerDecision =
        OnlineRanker::rank(body, static_cast<int>(req.candidates.size()));

    d.ranker_used        = rankerDecision.used;
    d.ranker_succeeded   = rankerDecision.succeeded;
    d.used_fallback      = rankerDecision.used_fallback;
    d.selected_index_raw = rankerDecision.selected_index;
    d.rationale          = rankerDecision.rationale;
    d.error              = rankerDecision.error;

    if (rankerDecision.succeeded
        && rankerDecision.selected_index >= 0
        && rankerDecision.selected_index
               < static_cast<int>(req.candidates.size())) {
        d.applied_index = rankerDecision.selected_index;
    }
    d.applied = &req.candidates[d.applied_index];

    emitTrace(req, phase, d);
    return d;
}

}  // namespace ADORA
}  // namespace mlir
