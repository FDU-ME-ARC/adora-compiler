//===------------------ schedule_order_decision.h -----------------------===//
//
// runtime-online-v0 §3 — linearization-order selection for a task graph.
//
// Given a topologically-legal task graph (populated from adora.dep_summary
// and adora.lc_dep_summary), multiple emission orders may satisfy all
// dependencies. The compiler constructs a small bounded set of legal
// orderings (topo_default, lc_greedy, critical_path_first, ...) and asks
// the policy to pick one. The compiler is the legality owner; the policy
// never sees a violating order.
//
// Phase 0 PR-γ scope: ship the shape + a single-candidate short-circuit
// implementation so call sites can route through AgentAPI without changing
// mapping output. Candidate enumeration lands in a follow-up.
//
//===----------------------------------------------------------------------===//
#ifndef ADORA_MAPPER_SCHEDULE_ORDER_DECISION_H
#define ADORA_MAPPER_SCHEDULE_ORDER_DECISION_H

#include <cstdint>
#include "mapper/agent_decision_api.h"

namespace mlir {
namespace ADORA {

struct ScheduleOrderCandidate {
  std::vector<int64_t> task_order;   // TaskNode ids in emission order
  std::string          label;        // "topo_default" / "lc_greedy" / ...
  int                  estimated_overlap_hint = 0;  // compiler heuristic
};

using ScheduleOrderRequest  = AgentAPI::Request<ScheduleOrderCandidate>;
using ScheduleOrderDecision = AgentAPI::Decision;

ScheduleOrderDecision
decideScheduleOrder(const ScheduleOrderRequest& req);

}  // namespace ADORA
}  // namespace mlir

#endif  // ADORA_MAPPER_SCHEDULE_ORDER_DECISION_H
