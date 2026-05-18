//===------------- partition_budget_decision.cpp ----------------------===//
//
// runtime-online-v0 §1 — per-task partition / tile-budget selection.
//
// Follows the same pattern as pipeline_scheduler.cpp: compiler builds a
// bounded legal candidate set, OnlineRanker ranks when >1 candidate exists,
// applied_index collapses to 0 on any violation (Silent Removal guarantee).
//
//===----------------------------------------------------------------------===//
#include "mapper/partition_budget_decision.h"

#include "mapper/agent_trace.h"
#include "mapper/online_ranker.h"

#include <sstream>
#include <string>

namespace mlir {
namespace ADORA {

namespace {

std::string buildRequestJson(const PartitionBudgetRequest& req,
                              const std::string& phase) {
    std::ostringstream os;
    os << "{\"schema_version\":\"runtime-online-v0\","
       << "\"phase\":\"" << agentTraceJsonEscape(phase) << "\","
       << "\"request_id\":\"" << agentTraceJsonEscape(AgentTrace::runId())
       << ":partition_budget:" << agentTraceJsonEscape(req.obs.func_symbol)
       << ":" << OnlineRanker::budgetUsed() << "\","
       << "\"decision_kind\":\"partition_budget\","
       << "\"func_symbol\":\"" << agentTraceJsonEscape(req.obs.func_symbol) << "\","
       << "\"op_kind\":\"" << agentTraceJsonEscape(req.obs.op_kind) << "\"";
    if (!req.obs.context_json.empty())
        os << ",\"context\":" << req.obs.context_json;

    os << ",\"legal_candidates\":[";
    for (size_t i = 0; i < req.candidates.size(); ++i) {
        const auto& c = req.candidates[i];
        if (i) os << ",";
        os << "{\"index\":" << i
           << ",\"label\":\"" << agentTraceJsonEscape(c.label) << "\""
           << ",\"tile_m\":" << c.tile_m
           << ",\"tile_n\":" << c.tile_n
           << ",\"tile_k\":" << c.tile_k
           << ",\"spm_bytes_needed\":" << c.spm_bytes_needed
           << ",\"banks_needed\":" << c.banks_needed
           << ",\"fits_default_budget\":"
           << (c.fits_default_budget ? "true" : "false")
           << "}";
    }
    os << "]}";
    return os.str();
}

void emitTrace(const std::string& phase,
               const PartitionBudgetRequest& req,
               const PartitionBudgetDecision& d) {
    if (!AgentTrace::enabled()) return;
    const auto& applied = req.candidates[d.applied_index];
    std::ostringstream os;
    os << "{\"decision_kind\":\"partition_budget\""
       << ",\"func_symbol\":\"" << agentTraceJsonEscape(req.obs.func_symbol) << "\""
       << ",\"op_kind\":\"" << agentTraceJsonEscape(req.obs.op_kind) << "\""
       << ",\"applied_index\":"   << d.applied_index
       << ",\"applied_label\":\"" << agentTraceJsonEscape(applied.label) << "\""
       << ",\"tile_m\":"  << applied.tile_m
       << ",\"tile_n\":"  << applied.tile_n
       << ",\"tile_k\":"  << applied.tile_k
       << ",\"ranker_used\":"    << (d.ranker_used    ? "true" : "false")
       << ",\"used_fallback\":"  << (d.used_fallback  ? "true" : "false")
       << ",\"rationale\":\"" << agentTraceJsonEscape(d.rationale) << "\""
       << ",\"error\":\""     << agentTraceJsonEscape(d.error)     << "\"}";
    AgentTrace::emit(phase, "online_decision", os.str());
}

}  // namespace

PartitionBudgetDecision
decidePartitionBudget(const PartitionBudgetRequest& req) {
    PartitionBudgetDecision d;
    const std::string phase = "partition_budget_select";

    if (!req.valid()) {
        d.applied_index = 0;
        d.used_fallback = true;
        d.error         = "empty candidate set";
        return d;
    }

    // Compiler contract: candidates[0] must be the safe default.
    if (!req.candidates[0].fits_default_budget) {
        d.applied_index = 0;
        d.used_fallback = true;
        d.error         = "default budget illegal (pre-filter gap)";
        emitTrace(phase, req, d);
        return d;
    }

    // Single-candidate or ranker disabled: fast path, no LLM call.
    if (req.candidates.size() < 2 || !OnlineRanker::enabled()) {
        d.applied_index      = 0;
        d.selected_index_raw = 0;
        d.ranker_used        = false;
        d.ranker_succeeded   = false;
        d.used_fallback      = false;
        d.rationale          = "single-candidate default";
        emitTrace(phase, req, d);
        return d;
    }

    // Multi-candidate: ask OnlineRanker.
    const std::string body = buildRequestJson(req, phase);
    auto dec = OnlineRanker::rank(body, static_cast<int>(req.candidates.size()));

    d.ranker_used        = dec.used;
    d.ranker_succeeded   = dec.succeeded;
    d.used_fallback      = dec.used_fallback;
    d.selected_index_raw = dec.selected_index;
    d.rationale          = dec.rationale;
    d.error              = dec.error;

    // Silent Removal: only apply if index is valid AND candidate is legal.
    if (dec.succeeded
        && dec.selected_index >= 0
        && dec.selected_index < static_cast<int>(req.candidates.size())
        && req.candidates[dec.selected_index].fits_default_budget) {
        d.applied_index = dec.selected_index;
    } else {
        d.applied_index = 0;
        d.used_fallback = true;
    }

    emitTrace(phase, req, d);
    return d;
}

}  // namespace ADORA
}  // namespace mlir
