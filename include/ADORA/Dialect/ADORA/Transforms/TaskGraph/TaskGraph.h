//===----------------------------------------------------------------------===//
// For Task Graph
//===----------------------------------------------------------------------===//
#ifndef ADORA_TASK_GRAPH_H
#define ADORA_TASK_GRAPH_H
#include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "TaskNode.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#include <unordered_map>
#include <vector>

#include "ADORA/Dialect/ADORA/Transforms/TaskGraph/DepKind.h"

namespace mlir {
namespace ADORA {

/////////////////////////
/// P4.0 — DataBlock-level dependency edge record.
/// Produced by analyzeDependencyInGraph, consumed by the serializer that
/// emits the `adora.dep_summary` DictionaryAttr onto the host FuncOp, and by
/// the mapper-side DepSummaryView parser (see DepSummaryView.{h,cpp}).
/// DataBlockDepKind is defined in DepKind.h so consumers and producers
/// share a single source of truth.
/////////////////////////

struct DataBlockDepEdge {
  TaskNode *src;           // producer / first accessor in IR order
  TaskNode *dst;           // consumer / second accessor in IR order
  DataBlockDepKind kind;
  bool mustOverlap;        // true iff exact-same-block; false means conservative overlap
};

/////////////////////////
/// base graph class
/////////////////////////
class TaskGraph{
private:
  std::unordered_map<TaskNode *, int> _nodes;
  std::vector<DataBlockDepEdge> _depEdges;   // P4.0 — datablock-level edges

public:
  void JustAddNode(TaskNode* node);

  void AddNodeAndAnalyzeDefaultDependency(TaskNode* node);
  template <typename T> void AddNodeAndAnalyzeDefaultDependency(T* node){
    AddNodeAndAnalyzeDefaultDependency(dyn_cast<TaskNode>(node));
  };

  std::vector<TaskNode *> getAllNodes();
  TaskNode* getNode(mlir::Operation* op);

  // P4.0 — datablock edge accessors.
  void addDepEdge(const DataBlockDepEdge& e) { _depEdges.push_back(e); }
  const std::vector<DataBlockDepEdge>& depEdges() const { return _depEdges; }
  int  getNodeId(TaskNode* n) const {
    auto it = _nodes.find(n);
    return (it == _nodes.end()) ? -1 : it->second;
  }

private:
  int getMaxNodeIdx();

public:
  TaskGraph(){}
  ~TaskGraph(){}
};

//////////////////
/// Some tool functions
//////////////////


} // namespace ADORA
} // namespace mlir
#endif