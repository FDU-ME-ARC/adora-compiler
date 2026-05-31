//===----------------------------------------------------------------------===//
// For Task Graph
//===----------------------------------------------------------------------===//
#ifndef ADORA_TASK_Node_H
#define ADORA_TASK_Node_H
#include "ADORA/Dialect/ADORA/IR/ADORA.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#include <unordered_map>
#include <vector>

namespace mlir {
namespace ADORA {

class TaskNode;
class KernelNode;
class BlockLoadNode;
class BlockStoreNode;


enum depType{ 
  NoDep,    // parallelizable
  Default,  // default dependency: load->config->execution->store
  SourceToStore, // a special default dependency: blockload->corresponding store or alloc->store 
  Depend,    //
  Undefine 
};

/////////////////////////
/// base node class
/////////////////////////
class TaskNode{
protected:
  mlir::Operation* _operation;

  // std::vector<TaskNode *> _innodes;
  std::unordered_map<TaskNode *, depType> _innodes;
  std::vector<TaskNode *> _outnodes;  
  
public:
  void setOperation(mlir::Operation* _){ _operation = _;};
  mlir::Operation* getOperation(){ return _operation;};
  mlir::Operation* getOperation() const { return _operation;};
  mlir::Operation* Operation(){ return _operation;};

  void addInNode(TaskNode* node);
  void addInNode(TaskNode* node, depType dep);
  void delInNode(TaskNode* node);
  depType getInNodeDep(TaskNode* node);
  std::vector<TaskNode *> getInNodes();

  void addOutNode(TaskNode* node);
  void delOutNode(TaskNode* node);
  std::vector<TaskNode *> getOutNodes();

  // virtual void dump();

public:
  TaskNode(mlir::Operation* operation){setOperation(operation);}
  TaskNode(){}
  ~TaskNode(){}
};

//////////////////
/// Some tool functions
//////////////////
/// @brief 
/// @param n1 
/// @param n2 
/// @param dep 
template<typename T1, typename T2>
void addConnectionBetweenTwoNode(T1* n1, T2* n2, depType dep = depType::Undefine){
  n1->addOutNode(n2);
  n2->addInNode(n1, dep);
}


/// @brief 
/// @param n1 
/// @param n2 
template<typename T1, typename T2>
void delConnectionBetweenTwoNode(T1* n1, T2* n2){
  n1->delOutNode(n2);
  n2->delInNode(n1);
}

//////////////////
/// derived class for KernelOp
//////////////////
class KernelNode : public TaskNode
{
private:
  ADORA::KernelOp _kernelop;

public:
  void setKernelOp(ADORA::KernelOp& _) {
    _kernelop = _; 
    setOperation(_.getOperation());
  };
  ADORA::KernelOp getKernelOp(){ return _kernelop;}

  void addInNode(TaskNode* node);
  void addInNode(TaskNode* node, depType dep);

  static bool classof(const TaskNode * node);

public:
  KernelNode(ADORA::KernelOp& kernelop){setKernelOp(kernelop);}
  ~KernelNode(){}
};

//////////////////
/// derived class for DataBlockLoadOp
//////////////////
class BlockLoadNode : public TaskNode
{
private:
  ADORA::DataBlockLoadOp _blockloadop;

public:
  void setBlockLoadOp(ADORA::DataBlockLoadOp& _) {
    _blockloadop = _; 
    setOperation(_.getOperation());
  };
  ADORA::DataBlockLoadOp getDataBlockLoadOp(){ return _blockloadop;}

  std::vector<KernelNode*> getKernelNodes(); 

  static bool classof(const TaskNode * node);

public:
  BlockLoadNode(ADORA::DataBlockLoadOp& _){setBlockLoadOp(_);}
  ~BlockLoadNode(){}
};

//////////////////
/// derived class for LocalMemAllocOp
//////////////////
class LocalAllocNode : public TaskNode
{
private:
  ADORA::LocalMemAllocOp _localmemallocop;

public:
  void setLocalMemAllocOp(ADORA::LocalMemAllocOp& _) {
    _localmemallocop = _; 
    setOperation(_.getOperation());
  };
  ADORA::LocalMemAllocOp getLocalMemAllocOp(){ return _localmemallocop;}

  std::vector<KernelNode*> getKernelNodes(); 

  static bool classof(const TaskNode * node);

public:
  LocalAllocNode(ADORA::LocalMemAllocOp& _){setLocalMemAllocOp(_);}
  ~LocalAllocNode(){}
};

//////////////////
/// derived class for DataBlockStoreOp
//////////////////
class BlockStoreNode : public TaskNode
{
private:
  ADORA::DataBlockStoreOp _blockstoreop;

public:
  void setBlockStoreOp(ADORA::DataBlockStoreOp& _) {
    _blockstoreop = _; 
    setOperation(_.getOperation());
  };
  ADORA::DataBlockStoreOp getDataBlockStoreOp(){ return _blockstoreop;}

  /// Returns the single KernelNode predecessor, or nullptr if none.
  /// Prefer getKernelNodes() when multiple predecessors are possible.
  KernelNode* getKernelNode(); 

  /// Returns all KernelNode predecessors (may be >1 in multi-tile benchmarks).
  std::vector<KernelNode*> getKernelNodes();

  static bool classof(const TaskNode * node);

public:
  BlockStoreNode(ADORA::DataBlockStoreOp& _){setBlockStoreOp(_);}
  ~BlockStoreNode(){}
};

} // namespace ADORA
} // namespace mlir
#endif