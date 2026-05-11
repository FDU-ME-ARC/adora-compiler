//===----------------------------------------------------------------------===//
// For Task Graph
//===----------------------------------------------------------------------===//
#include "ADORA/Dialect/ADORA/Transforms/TaskGraph/TaskNode.h"

namespace mlir {
namespace ADORA {



/////////////////////////////
// class TaskNode
/////////////////////////////

void TaskNode::addInNode(TaskNode* node){
  if(_innodes.count(node) == 0){
    /// is not an Input
    _innodes[node] = depType::Undefine;
  }
}

static depType judegeDep(depType dep1, depType dep2){
  switch (dep1)
  {
  case depType::Undefine :
    return dep2;
    break;

  case depType::Default :
    switch (dep2)
    {
      case depType::Undefine :
      return dep1;
      break;

      case depType::Default :
      return dep1;
      break;

      case depType::Depend :
      return depType::Depend;
      break;

      case depType::SourceToStore :
      return depType::SourceToStore;
      break;
    }
    break;

  case depType::SourceToStore : 
    return depType::SourceToStore;
    break;

  case depType::Depend :
    return depType::Depend;
    break;

  default:
    return depType::Undefine;
  }
}

/// @brief Adds a TaskNode as an incoming dependency node.
/// Note that this function does not modify the outgoing nodes of @node.
///
/// @param node A pointer to the TaskNode to be added as an incoming dependency.
/// @param dep The type of dependency to associate with the incoming node.
void TaskNode::addInNode(TaskNode* node, depType dep){
  if(_innodes.count(node) > 0){
    /// is not an Input
    _innodes[node] = judegeDep(dep, _innodes[node]);
  }
  else {
    _innodes[node] = dep;
  }
}

/// @brief Deletes a TaskNode from the list of incoming dependency nodes.
/// It does not affect the outgoing nodes of @node. If @node is not 
/// @param node A pointer to the TaskNode to be removed from the incoming nodes.
void TaskNode::delInNode(TaskNode* node){
  if(_innodes.count(node) > 0){
    _innodes.erase(node);
  }
}

depType TaskNode::getInNodeDep(TaskNode* node){
  if(_innodes.count(node) > 0)
    return _innodes[node];
  else 
    return depType::NoDep;
}

std::vector<TaskNode *> TaskNode::getInNodes(){
  std::vector<TaskNode *> keys;
  
  for (const auto& pair : _innodes) {
    keys.push_back(pair.first);
  }

  return keys;
}

void TaskNode::addOutNode(TaskNode* node){
  _outnodes.push_back(node);
}

void TaskNode::delOutNode(TaskNode* node){
  auto it = std::find(_outnodes.begin(), _outnodes.end(), node);
  if (it != _outnodes.end()) {
    _outnodes.erase(it); 
  }
}

std::vector<TaskNode *> TaskNode::getOutNodes(){
  return _outnodes;
}

//////////////////
/// KernelNode
//////////////////
bool KernelNode::classof(const TaskNode* node){
  if(isa<ADORA::KernelOp>(node->getOperation())){
    return true;
  }
  else{
    return false;
  }
}

void KernelNode::addInNode(TaskNode* node){
  if(_innodes.count(node) == 0){
    /// is not an Input
    _innodes[node] = depType::Undefine;
  }
  if(isa<LocalAllocNode>(node)){
    dyn_cast<LocalAllocNode>(node)->getLocalMemAllocOp()
        .addAnotherKernelName(this->getKernelOp().getKernelName());
  }
}

void KernelNode::addInNode(TaskNode* node, depType dep){
  if(_innodes.count(node) > 0){
    /// is not an Input
    _innodes[node] = judegeDep(dep, _innodes[node]);
  }
  else {
    _innodes[node] = dep;
  }
  if(isa<LocalAllocNode>(node)){
    dyn_cast<LocalAllocNode>(node)->getLocalMemAllocOp()
        .addAnotherKernelName(this->getKernelOp().getKernelName());
  }
}

//////////////////
/// BlockLoadNode
//////////////////
bool BlockLoadNode::classof(const TaskNode* node){
  if(isa<ADORA::DataBlockLoadOp>(node->getOperation())){
    return true;
  }
  else{
    return false;
  }
}

std::vector<KernelNode*> BlockLoadNode::getKernelNodes(){
  std::vector<KernelNode*> kernels;
  for(auto outnode : getOutNodes()){
    if(isa<KernelNode>(outnode)){
      kernels.push_back(dyn_cast<KernelNode>(outnode));
    }
  }
  
  return kernels;
}

//////////////////
/// LocalAllocNode
//////////////////
bool LocalAllocNode::classof(const TaskNode* node){
  if(isa<ADORA::LocalMemAllocOp>(node->getOperation())){
    return true;
  }
  else{
    return false;
  }
}

std::vector<KernelNode*> LocalAllocNode::getKernelNodes(){
  std::vector<KernelNode*> kernels;
  for(auto outnode : getOutNodes()){
    if(isa<KernelNode>(outnode)){
      kernels.push_back(dyn_cast<KernelNode>(outnode));
    }
  }
  
  return kernels;
}

//////////////////
/// BlockStoreNode
//////////////////
bool BlockStoreNode::classof(const TaskNode* node){
  if(isa<ADORA::DataBlockStoreOp>(node->getOperation())){
    return true;
  }
  else{
    return false;
  }
}

KernelNode* BlockStoreNode::getKernelNode(){
  std::vector<KernelNode*> kernels;
  for(auto innode : getInNodes()){
    if(isa<KernelNode>(innode)){
      kernels.push_back(dyn_cast<KernelNode>(innode));
    }
  }
  if(kernels.size() == 0){
    return nullptr;
  }
  else{
    assert(kernels.size() == 1);
    return kernels[0];
  }
}

} // namespace ADORA
} // namespace mlir