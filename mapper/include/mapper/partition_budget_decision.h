//===----------------- partition_budget_decision.h ---------------------===//
//
// runtime-online-v0 §1 — per-task partition / tile-budget selection.
//
// Decision site: before decidePipeline in mapGemm / mapConv, the compiler
// asks the policy to rank alternative (tile_m, tile_n, tile_k) budgets. The
// compiler owns legality: only candidates for which fits_default_budget is
// true reach the ranker. Per PR-β (paper_plan Phase 0 scaffold) this file
// ships the shape; the first caller will pin candidates.size()==1 (default
// tile) so behavior is bit-identical to pre-PR-β. Candidate enumeration
// moves into a follow-up PR once ADG.spm_bytes_avail / ADG.banks_available
// are wired through SystolicImplInterface.
//
//===--------------------------------------------------------------------===//
#ifndef ADORA_MAPPER_PARTITION_BUDGET_DECISION_H
#define ADORA_MAPPER_PARTITION_BUDGET_DECISION_H

#include "mapper/agent_decision_api.h"

namespace mlir {
namespace ADORA {

struct PartitionBudgetCandidate {
  int  tile_m           = 0;
  int  tile_n           = 0;
  int  tile_k           = 0;   // set to 0 for non-GEMM (Conv repurposes as tile_oc)
  int  spm_bytes_needed = 0;
  int  banks_needed     = 0;
  bool fits_default_budget = true;  // compiler-owned legality flag
  std::string label;
};

using PartitionBudgetRequest  = AgentAPI::Request<PartitionBudgetCandidate>;
using PartitionBudgetDecision = AgentAPI::Decision;

// decidePartitionBudget(req)
//   Precondition : req.candidates[0] is the compiler-pinned default budget.
//   Postcondition: returned decision.applied_index is in [0, size) and points
//                  at a candidate with fits_default_budget==true. Violations
//                  collapse to 0 with used_fallback=true.
PartitionBudgetDecision decidePartitionBudget(const PartitionBudgetRequest& req);

}  // namespace ADORA
}  // namespace mlir

#endif  // ADORA_MAPPER_PARTITION_BUDGET_DECISION_H
