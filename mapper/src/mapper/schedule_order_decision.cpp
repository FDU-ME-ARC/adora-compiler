//===----------------- schedule_order_decision.cpp -----------------------===//
//
// runtime-online-v0 §3 — dataflow task schedule-order selection.
//
// Candidates are orderings of DFG tasks: topo_default (compiler topo sort),
// lc_greedy (minimize loop-carried dependence stalls), critical_path_first.
// Only candidates produced by DepSummaryView analysis are legal.
// OnlineRanker ranks when >1 candidate exists; collapses to 0 otherwise.
//
//===----------------------------------------------------------------------===//
#include "mapper/schedule_order_decision.h"

#include "mapper/agent_trace.h"
#include "mapper/online_ranker.h"

#include <sstream>
#include <string>

namespace mlir {
namespace ADORA {

namespace {

std::string buildRequestJson(const ScheduleOrderRequest& req,
                              const std::string& phase) {
    std::ostringstream os;
    os << "{\"schema_version\":\"runtime-online-v0\","
       << "\"phase\":\"" << agentTraceJsonEscape(phase) << "\","
       << "\"request_id\":\"" << agentTraceJsonEscape(AgentTrace::runId())
       << ":schedule_order:" << agentTraceJsonEscape(req.obs.func_symbol)
       << ":" << OnlineRanker::budgetUsed() << "\","
       << "\"decision_kind\":\"schedule_order\","
       << "\"func_symbol\":\"" << agentTraceJsonEscape(req.obs.func_symbol) << "\"";
    if (!req.obs.context_json.empty())
        os << ",\"context\":" << req.obs.context_json;

    os << ",\"legal_candidates\":[";
    for (size_t i = 0; i < req.candidates.size(); ++i) {
        const auto& c = req.candidates[i];
        if (i) os << ",";
        os << "{\"index\":" << i
           << ",\"label\":\"" << agentTraceJsonEscape(c.label) << "\""
           << ",\"estimated_overlap_hint\":" << c.estimated_overlap_hint
           << ",\"task_count\":" << c.task_order.size()
           << "}";
    }
    os << "]}";
    return os.str();
}

void emitTrace(const std::string& phase,
               const ScheduleOrderRequest& req,
               const ScheduleOrderDecision& d) {
    if (!AgentTrace::enabled()) return;
    const auto& applied = req.candidates[d.applied_index];
    std::ostringstream os;
    os << "{\"decision_kind\":\"schedule_order\""
       << ",\"func_symbol\":\"" << agentTraceJsonEscape(req.obs.func_symbol) << "\""
       << ",\"task_count\":" << applied.task_order.size()
       << ",\"applied_index\":"   << d.applied_index
       << ",\"applied_label\":\"" << agentTraceJsonEscape(applied.label) << "\""
       << ",\"ranker_used\":"   << (d.ranker_used   ? "true" : "false")
       << ",\"used_fallback\":" << (d.used_fallback ? "true" : "false")
       << ",\"rationale\":\"" << agentTraceJsonEscape(d.rationale) << "\""
       << ",\"error\":\""     << agentTraceJsonEscape(d.error)     << "\"}";
    AgentTrace::emit(phase, "online_decision", os.str());
}

}  // namespace

ScheduleOrderDecision
decideScheduleOrder(const ScheduleOrderRequest& req) {
    ScheduleOrderDecision d;
    const std::string phase = "schedule_order_select";

    if (!req.valid()) {
        d.applied_index = 0;
        d.used_fallback = true;
        d.error         = "empty candidate set";
        return d;
    }

    // Single-candidate or ranker disabled: fast path.
    if (req.candidates.size() < 2 || !OnlineRanker::enabled()) {
        d.applied_index      = 0;
        d.selected_index_raw = 0;
        d.ranker_used        = false;
        d.ranker_succeeded   = false;
        d.used_fallback      = false;
        d.rationale          = "single-candidate topo_default";
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

    // Silent Removal: collapse to 0 on any invalid response.
    if (dec.succeeded
        && dec.selected_index >= 0
        && dec.selected_index < static_cast<int>(req.candidates.size())) {
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
