#ifndef ADORA_EMITPYTEST_H
#define ADORA_EMITPYTEST_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Analysis/CallGraph.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/raw_ostream.h"

#include "ADORA/Dialect/ADORA/IR/ADORA.h"

#include "mapper/mapper.h"
#include "mapper/mapper_sa.h"
#include "mapper/io_scheduler.h"
#include "./Emit.h"

using namespace mlir;
using namespace mlir::ADORA;

namespace mlir {
namespace ADORA {
} /// ADORA
} /// mlir


// class CGRVOpEmitter;

class PytestEmitter : public BaseEmitter{
public:
  using BaseEmitter::BaseEmitter;
  /////////////////////////////////
  // The following functions must be implemented in your derived class.
  // These functions are defined as pure virtual functions in the base class
  // to ensure each subclass provides implementations according to its specific needs.
  /////////////////////////////////
  void emitFunctionHead(func::FuncOp &funcop, llvm::raw_ostream &os) override;
  void emitBlock(mlir::Block &block, llvm::raw_ostream &os) override;
  std::string GenerateCGRAConfig(ADORA::KernelOp& kernel, MapperSA* mapper) override;
  std::string GenerateCGRAConfig(ADORA::KernelOp& kernel, Configuration configuration, ADG* adg) override;
  void GenerateCGRACFGAndEXE(ADORA::KernelOp& kernel, MapperSA* mapper) override;
  void GenerateCGRACFGAndEXE(ADORA::KernelOp& kernel, Configuration configuration, ADG* adg) override;
  bool emitCGRACallFunction(llvm::raw_ostream &os) override;
  /////////////////////////////////
  bool emitPytest(llvm::raw_ostream &os);

  llvm::SmallDenseMap<mlir::Value, Op_Name_C> getValueNameList(){return _value_name_list;}
  void appendValueNameList(mlir::Value v, Op_Name_C info){_value_name_list[v] = info;}
  std::string lookupName(mlir::Value v){ 
    if(_value_name_list.count(v))
      return _value_name_list[v].name();
    else
      return "";
  }

  std::string lookupVarConfigName(const std::string);

  // PytestEmitter(mlir::ModuleOp m): BaseEmitter(m){}
  ~PytestEmitter(){};

  friend class PyOpEmitter;

  llvm::SmallDenseMap<ADORA::KernelOp, std::pair<std::string, int>> KnToCfgArrayInfo;   // map : kernelop -> (cfg array's name, cfgnum)
  llvm::SmallDenseMap<ADORA::KernelOp, std::string> KnToCfgData; // map : kernelop -> total cfg array declaration
  llvm::SmallDenseMap<ADORA::KernelOp, std::pair<std::string, std::string> > KnToPingpongCfgData; // map : kernelop -> ping cfg array and pong cfg array declaration
  llvm::SmallDenseMap<ADORA::KernelOp, std::string> KnToCfgExe;

  ////////////////
  /// Gemm
  ////////////////
  void emitGemmBlock(mlir::Block &block, llvm::raw_ostream &os);

  // Stream IDs computed from SSA token edges at emit time (replaces the
  // assign-streams IR attribute approach).  Populated by computeStreamIds()
  // at the start of emitBlock.  Public so PyOpEmitter (in EmitPytest.cpp)
  // can populate and read it.
  llvm::DenseMap<mlir::Operation *, int> _streamId;
  static constexpr int kMaxStreams = 4;

private:
  llvm::SmallDenseMap<mlir::Value, Op_Name_C> _value_name_list;

  /// reset Indent size for python emit
  // unsigned _currentIndent = 0;
  void addIndent() override{_currentIndent += 4;}
  void reduceIndent() override {_currentIndent = _currentIndent >= 4 ? _currentIndent - 4 : 0;}
};

namespace mlir {
namespace ADORA {




} // namespace ADORA
} // namespace mlir

#endif // CGRV_EMITPYTEST_H
