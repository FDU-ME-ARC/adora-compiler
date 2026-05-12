//===----------------- schedule_order_decision.cpp -----------------------===//
//
// Phase 0 PR-γ minimal implementation: single-candidate short-circuit.
// Identical shape to decidePartitionBudget; follow-up PR adds lc_greedy /
// critical_path_first candidates driven by DepSummaryView and
// LoopCarriedDepResult.
//
//===----------------------------------------------------------------------===//
#include "mapper/schedule_order_decision.h"

namespace mlir {
namespace ADORA {

ScheduleOrderDecision
decideScheduleOrder(const ScheduleOrderRequest& req) {
  ScheduleOrderDecision d;
  if (!req.valid()) {
    d.applied_index = 0;
    d.used_fallback = true;
    d.error         = "empty candidate set";
    return d;
  }
  d.applied_index      = 0;
  d.selected_index_raw = 0;
  d.ranker_used        = false;
  d.ranker_succeeded   = false;
  d.used_fallback      = false;
  d.rationale          = "single-candidate topo_default";
  return d;
}

}  // namespace ADORA
}  // namespace mlir
