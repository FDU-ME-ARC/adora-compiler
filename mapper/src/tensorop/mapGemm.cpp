//===------------------ mapGemm.cpp - ADORATensor Lower process ----------------------===//
/// builtin dialect
#include "mlir/Dialect/Affine/IR/AffineOps.h"

/// ADORA dialect
#include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "ADORA/Dialect/ADORA/Utility/Utility.h"
#include "ADORA/Dialect/ADORA/Transforms/SimplifyLoadStore.h"
#include "ADORA/Dialect/ADORATensor/IR/ADORATensor.h"
#include "ADORA/Dialect/ADORATensor/Interface/SystolicImplInterface.h"

#include "tensorop/TensorOp.h"
#include "ADORA/Dialect/ADORATensor/Lowering/TensorOps/LowerGemm.h"

#include "ir/adg_ir.h"
#include "ir/dfg_ir.h"
#include "mapper/mapper_sa.h"

using namespace ::mlir::ADORA::ADORATensor;
using namespace ::mlir::affine;

namespace mlir{
namespace ADORA{

template <typename opT>
void simplifyAffineMapAndOperand(opT op){
  AffineMap origin_map = op.getAffineMap(); 
  SmallVector<Value> oprands = op.getMapOperands();
  origin_map.dump();
  op.dump();
  // origin_oprands.dump();
  // simplifyMapWithOperands(origin_map, origin_oprands);
  canonicalizeMapAndOperands(&origin_map, &oprands);

  origin_map.dump();
  op.setAffineMap(origin_map);
  op.setMapOperands(oprands);
  op.dump();
  // origin_oprands.dump();  
}

void tryToMoveOutBlockAccessOp(affine::AffineForOp forop){
  ////////// Only consider datablockload right now
  bool NoChange = false;
  while(!NoChange){ // Keep walking the func until no change occurs in this func
    NoChange = true;
    auto toHoists = GetAllHoistOp<DataBlockLoadOp>(forop);
    for(DataBlockLoadOp load : toHoists){
      load.dump();
      Operation* ParentOp = load.getOperation()->getParentOp();
      if(!isa<affine::AffineForOp>(ParentOp))
        continue;
      /// write here in the morning
      simplifyAffineMapAndOperand(load);

      load.getOperation()->moveBefore(ParentOp);
      ParentOp->getBlock()->dump();
      NoChange = false;
      break;
    }
    // forop.walk([&](ADORA::DataBlockLoadOp load) {
    //   if(load.hasAttr("ADORAGemm")){
    //     affine::AffineForOp parentFor = load.getParent
    //   }
    // });
   
  }
}

void TensorDataflowGen::MapNestedForOrKernel(
  ADORA_TENSOR_MAPPER* mapper, mlir::Operation* forOrKernel, std::string& OpNameFile_str){
  ADORA::KernelOp kernel;
  if(isa<ADORA::KernelOp>(forOrKernel)){
    kernel = dyn_cast<ADORA::KernelOp>(forOrKernel);
  }
  else if(isa<affine::AffineForOp>(forOrKernel)){
    kernel = findTheOnlyKernelInNestedLoop(dyn_cast<affine::AffineForOp>(forOrKernel));
  }
  kernel.getOperation()->setAttr("Pingpong", mlir::UnitAttr::get(forOrKernel->getContext()));

  /// Generating DFG
  std::string kernelName = kernel.getKernelName();

  if(kernelName.empty()){
    kernelName = "GEMM_"+ tensorOpCnt;
  }
  
  LLVMCDFG *CDFG = new LLVMCDFG(kernelName, OpNameFile_str);
  generateCDFGfromKernel(CDFG, kernel, /*verbose=*/_verbose);

  /// DFG Mapping to CGRA architecture
  DFGIR* dfg_ir = new DFGIR(CDFG);
  // DFGIR_Vec.push_back(dfg_ir);
  DFG* dfg = dfg_ir->getDFG();
  
  mapper->setDFG(dfg);

    // // some io nodes must be placed at some place
    // if(emit_type == "pytest"){
    //   PyEmitter.preestablishPlacementConstraints(kernel, mapper);
    // }
    // else{ /// default to be C
    //   CEmitter.preestablishPlacementConstraints(kernel, mapper);
    // }

  std::filesystem::create_directory(kernelName + "_map_result");
  CDFG->CDFGtoDOT(kernelName + "_map_result/before_map_" + CDFG->name_str() + "_CDFG.dot");
  bool succeed = mapper->execute(/*dumpCallFunc=*/false, /*dumpMappedViz*/true, /*resultDir=*/kernelName + "_map_result");
    // std::filesystem::create_directory("map_result");
    // CDFG->CDFGtoDOT("map_result/before_map_" + CDFG->name_str() + "_CDFG.dot");
    // bool succeed = mapper->execute(/*dumpCallFunc=*/false, /*dumpMappedViz*/true, /*resultDir=*/"map_result");
  if(succeed){
      // Mapping is successful, get all blockload and blockstore op and corresponding spad memory addresses.
      // if(_emit_type == "pytest"){
        pyEmitter->setMapResult(kernel, mapper);
        pyEmitter->DataBlockOperationsToSPADInfo(kernel, mapper);
        pyEmitter->setTileEnsForKernel(kernel);
        pyEmitter->GenerateCGRAConfig(kernel, mapper);

      // }
      // else{ /// default to be C
      //   cEmitter->setMapResult(kernel, mapper);
      //   cEmitter->DataBlockOperationsToSPADInfo(kernel, mapper);
      //   cEmitter->GenerateCGRAConfig(kernel, mapper);
      // }
    }
    // kernel_cnt++;
  forOrKernel->dump();

  // for(auto elem : pyEmitter->getLoadToSPMInfosMap()){
  //   ADORA::DataBlockLoadOp load = elem.first;
  //   load.dump();
  // }
  // for(auto elem : pyEmitter->getLocalAllocToSPMMap()){
  //   ADORA::LocalMemAllocOp alloc = elem.first;
  //   alloc.dump();
  // }
}

bool TensorDataflowGen::visitOp(ADORATensor::GemmOp op){
  SystolicImplInterface SystolicPara(op);
  ArrayRef<int64_t> tilesize = SystolicPara.getTileSize();
  StringRef strategy = SystolicPara.getStationaryKind();
  AffineForOp newfor;
  if(strategy == getDataflowStrategyStrRef(DataflowStrategy::WeightStationary)){
    newfor = TiledWeightStationaryGemm(opbuilder, op, tilesize); 
  }
  else if(strategy == getDataflowStrategyStrRef(DataflowStrategy::InputStationary)){
    newfor = TiledInputStationaryGemm(opbuilder, op, tilesize); 
  }
  else if(strategy == getDataflowStrategyStrRef(DataflowStrategy::OutputStationary)){
    newfor = TiledOutputStationaryGemm(opbuilder, op, tilesize); 
  }
  
  simplifyLoopLevelsInRegion(newfor.getRegion(), /*donttouchkernel=*/true);
  if(_verbose) newfor.dump();
  tryToMoveOutBlockAccessOp(newfor);
  if(_verbose) newfor.dump();
  SimplifyBlockAccessOp(newfor.getRegion());
  if(_verbose) newfor.dump();

  ADORA_TENSOR_MAPPER* mapper = new ADORA_TENSOR_MAPPER(_adg, _timeout_ms, _max_iters, _objOpt);
  mappers.push_back(mapper);
      
  // mlir::Operation* loweredIR = op->getNextNode();
  MapNestedForOrKernel(mapper, newfor, _OpNameFile_str);

  op.erase();

  return true;
}

}
}