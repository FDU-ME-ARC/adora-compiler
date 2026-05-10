//===----------------------------------------------------------------------===//
// For Task Graph
//===----------------------------------------------------------------------===//
#include "ADORA/Dialect/ADORA/Transforms/TaskGraph/TaskGraph.h"
#include "ADORA/Dialect/ADORA/Transforms/DependencyAnalysis.h"

namespace mlir {
namespace ADORA {


/// @brief 
/// @param src 
/// @param dst 
/// @return 
static depType checkDefaultDependency(TaskNode* src, TaskNode* dst){
  if(isa<KernelNode>(src)){
    KernelNode* newsrc = dyn_cast<KernelNode>(src);
    if(isa<BlockStoreNode>(dst)){
      BlockStoreNode* newdst = dyn_cast<BlockStoreNode>(dst);
      if(newsrc->getKernelOp().hasKernelName()
          && newdst->getDataBlockStoreOp().hasKernelName()
          && newsrc->getKernelOp().getKernelName() == newdst->getDataBlockStoreOp().getKernelName()){
            return depType::Default;
      }
    }
  }
  else if(isa<BlockLoadNode>(src)){
    BlockLoadNode* newsrc = dyn_cast<BlockLoadNode>(src);
    if(isa<KernelNode>(dst)){
      KernelNode* newdst = dyn_cast<KernelNode>(dst);
      if(newsrc->getDataBlockLoadOp().hasKernelName()
          && newdst->getKernelOp().hasKernelName()
          && newsrc->getDataBlockLoadOp().getKernelName() == newdst->getKernelOp().getKernelName()){
            return depType::Default;
      }
    }
    else if(isa<BlockStoreNode>(dst)){
      BlockStoreNode* newdst = dyn_cast<BlockStoreNode>(dst);
      if(newsrc->getDataBlockLoadOp().hasKernelName()
          && newdst->getDataBlockStoreOp().hasKernelName()
          && newsrc->getDataBlockLoadOp().getKernelName() == newdst->getDataBlockStoreOp().getKernelName()
          && newsrc->getDataBlockLoadOp().getId() == newdst->getDataBlockStoreOp().getId()){
            return depType::SourceToStore;
      }
    }
  }
  
  else if(isa<LocalAllocNode>(src)){
    LocalAllocNode* newsrc = dyn_cast<LocalAllocNode>(src);
    if(isa<BlockStoreNode>(dst)){
      BlockStoreNode* newdst = dyn_cast<BlockStoreNode>(dst);
      if(newsrc->getLocalMemAllocOp().hasKernelName()
          && newdst->getDataBlockStoreOp().hasKernelName()
          && newsrc->getLocalMemAllocOp().getKernelName() == newdst->getDataBlockStoreOp().getKernelName()
          && newsrc->getLocalMemAllocOp().getId() == newdst->getDataBlockStoreOp().getId()){
            return depType::SourceToStore;
      }
    }
    if(isa<KernelNode>(dst)){
      KernelNode* newdst = dyn_cast<KernelNode>(dst);
      if(newsrc->getLocalMemAllocOp().hasKernelName()
          && newdst->getKernelOp().hasKernelName()
          && newsrc->getLocalMemAllocOp().getKernelName() == newdst->getKernelOp().getKernelName()){
            return depType::Default;
      }
    }
  }

  return depType::Undefine;
}








////////////////////////////////////////////
/// TaskGraph class
////////////////////////////////////////////
/// @brief add a node to graph, but do not analyze the dependency between this node and other nodes
/// @param node 
void TaskGraph::JustAddNode(TaskNode* node){
  int newIndex = getMaxNodeIdx() + 1;
  _nodes[node] = newIndex;
}

/// @brief add a node to graph and 
///        analyze the default dependency between this node and other nodes
///        Default dependendy is load-kernel-store dependency.  
/// @param node 
void TaskGraph::AddNodeAndAnalyzeDefaultDependency(TaskNode* newnode){
  JustAddNode(newnode);

  for (auto& pair : _nodes) {
    int index = pair.second;
    TaskNode* node = pair.first;

    if(checkDefaultDependency(/*src*/node, /*dst*/newnode) == depType::Default){
      addConnectionBetweenTwoNode(node, newnode, /*dep=*/depType::Default);
    }
    else if(checkDefaultDependency(/*src*/newnode, /*dst*/node) == depType::Default){
      addConnectionBetweenTwoNode(newnode, node, /*dep=*/depType::Default);
    }
    else if(checkDefaultDependency(/*src*/node, /*dst*/newnode) == depType::SourceToStore){
      addConnectionBetweenTwoNode(node, newnode, /*dep=*/depType::SourceToStore);
    }
  }
}

std::vector<TaskNode *> TaskGraph::getAllNodes(){
  std::vector<TaskNode *> result;
  for (const auto& pair : _nodes) {
    result.push_back(pair.first);
  }
  return result;
}

/// @brief Retrieves the TaskNode associated with the specified Operation.
/// @param op A pointer to the mlir::Operation to find the associated TaskNode for.
/// @return A pointer to the matching TaskNode, or nullptr if no match is found.
TaskNode* TaskGraph::getNode(mlir::Operation* op) {
    // Iterate through each TaskNode stored in _nodes
    // op->dump();
    // std::cout << op << "\n";
    for (const auto& pair : _nodes) {
        TaskNode* node = pair.first; // Get the TaskNode
        mlir::Operation* nodeOp = node->getOperation(); // Get the corresponding operation

        // nodeOp->dump();
        // std::cout << nodeOp << "\n";

        // Check if the current node's operation matches the given operation
        if (nodeOp == op) {
          return node; // Found a matching node, return it
        }
    }
    
    // Return nullptr if no matching node is found
    return nullptr;
}








int TaskGraph::getMaxNodeIdx(){
  int maxIndex = std::numeric_limits<int>::min();
  for (const auto& pair : _nodes) {
    if (pair.second > maxIndex) {
      maxIndex = pair.second;
    }
  }
  
  return (maxIndex == std::numeric_limits<int>::min()) ? -1 : maxIndex;
}

} // namespace ADORA
} // namespace mlir