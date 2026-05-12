//===--------------------- agent_decision_api.h --------------------------===//
//
// Unified contract for agent-in-the-loop compilation decisions.
//
// Per paper_plan.md §statement-of-method, every agent-participating decision
// in the compiler follows the same 4-tuple:
//
//     compiler owns: φ, Γ, T, a0
//     policy owns:   π_θ (acts only on Γ(s))
//     reliability:   Legality Preservation + Silent Removal
//     quality:       Quality-Correctness Decoupling
//
// Concretely, a decision site produces a Request<Candidate> whose
// candidates[0] is the compiler-pinned fallback (a0). An OnlineRanker may
// reorder / select within the candidate set; any out-of-range or malformed
// selection collapses back to applied_index=0 and sets used_fallback=true.
//
// This header is intentionally dependency-light (no MLIR / JSON / Ranker
// includes) so that dialect-side passes (compiler-online-v0) and mapper-side
// lowerings (mapper-online-v0 / runtime-online-v0) can both target it.
//
// Per-decision headers (pipeline_scheduler.h, partition_budget_decision.h,
// schedule_order_decision.h, UnrollFactorDecision.h) specialize Candidate
// and add decision-kind-specific context fields.
//
//===----------------------------------------------------------------------===//
#ifndef ADORA_MAPPER_AGENT_DECISION_API_H
#define ADORA_MAPPER_AGENT_DECISION_API_H

#include <string>
#include <vector>

namespace mlir {
namespace ADORA {
namespace AgentAPI {

// Lightweight observation summary attached to every Request. Per-decision
// context goes in context_json as a compact JSON object ready for
// AgentTrace emission.
struct StateObs {
  std::string decision_kind;   // e.g. "pipeline" / "partition_budget"
  std::string func_symbol;     // FuncOp symbol name (AgentTrace grouping key)
  std::string op_kind;         // e.g. "gemm" / "conv_direct" / "loop_nest"
  std::string context_json;    // pre-serialized JSON payload (may be empty)
};

// Request<Candidate>: candidates[0] MUST be the compiler-pinned default.
// candidates.empty() is a contract violation (callers assert).
template <class Candidate>
struct Request {
  StateObs               obs;
  std::vector<Candidate> candidates;

  bool valid() const { return !candidates.empty(); }
};

// Decision: shape shared across every agent hook. Populated by
// decide<Kind>(req) implementations after consulting OnlineRanker.
struct Decision {
  int         applied_index      = 0;     // index actually used by compiler
  int         selected_index_raw = 0;     // raw value returned by ranker
  bool        ranker_used        = false; // did we consult OnlineRanker?
  bool        ranker_succeeded   = false; // did it return a legal index?
  bool        used_fallback      = false; // fell back to candidates[0]?
  std::string rationale;
  std::string error;
};

// Canonical AgentTrace event name for every hook. AgentTrace JSONL readers
// should switch on payload.decision_kind rather than on event name.
inline const char* kAgentDecisionEvent() { return "agent_decision"; }

}  // namespace AgentAPI
}  // namespace ADORA
}  // namespace mlir

#endif  // ADORA_MAPPER_AGENT_DECISION_API_H
