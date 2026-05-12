//===------------- partition_budget_decision.cpp ----------------------===//
//
// Phase 0 PR-β minimal implementation: single-candidate short-circuit.
//
// Full enumeration + OnlineRanker integration lands in a follow-up once
// ADG budget fields are wired through. For now the function exists so
// call sites (mapGemm/mapConv) can route through the unified AgentAPI
// contract and emit a structured "agent_decision" event per task.
//
//===----------------------------------------------------------------------===//
#include "mapper/partition_budget_decision.h"

namespace mlir {
namespace ADORA {

PartitionBudgetDecision
decidePartitionBudget(const PartitionBudgetRequest& req) {
  PartitionBudgetDecision d;
  if (!req.valid()) {
    d.applied_index   = 0;
    d.used_fallback   = true;
    d.error           = "empty candidate set";
    return d;
  }
  // Single-candidate fast path. When callers start enumerating legal budgets
  // and populate req.candidates.size()>1, this function will consult the
  // OnlineRanker. For now we simply confirm candidate[0] is legal.
  if (!req.candidates[0].fits_default_budget) {
    d.applied_index   = 0;
    d.used_fallback   = true;
    d.error           = "default budget illegal (pre-filter gap)";
    return d;
  }
  d.applied_index       = 0;
  d.selected_index_raw  = 0;
  d.ranker_used         = false;
  d.ranker_succeeded    = false;
  d.used_fallback       = false;
  d.rationale           = "single-candidate default";
  return d;
}

}  // namespace ADORA
}  // namespace mlir
