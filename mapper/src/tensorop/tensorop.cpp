//===----------------------------------------------------------------------===//
//
// Copyright 2023-2024 The ADORA Authors.
//
//===----------------------------------------------------------------------===//

#include "tensorop/TensorOp.h"

namespace mlir{
namespace ADORA{


void MapAdoraTensorOp(MLIRContext* context, mlir::ModuleOp moduleop, 
                    std::vector<ADORA_TENSOR_MAPPER*> mappers,

                    CGRACallEmitter* CEmitter, 
                    PytestEmitter* PyEmitter,
                    VitisSDKEmitter* SdkEmitter,
                    
                    ADG* adg, std::string& OpNameFile_str,
                    int timeout_ms, int max_iters, bool objOpt,
                    bool verbose){
  TensorDataflowGen engine(context);
  engine.setEmitter(CEmitter);
  engine.setEmitter(PyEmitter);
  engine.setMappingArgs(adg, OpNameFile_str, timeout_ms, max_iters, objOpt);
  engine.setVerbose(verbose);
  if(verbose) moduleop.dump();
  moduleop.walk([&](mlir::Operation* op) {
    if(engine.dispatchVisitor(op)){
      if(verbose) {moduleop.dump();}
    }
    
 
    // mapper->setDFG(dfg);

    // // some io nodes must be placed at some place
    // if(emit_type == "pytest"){
    //   PyEmitter.preestablishPlacementConstraints(kernel, mapper);
    // }
    // else{ /// default to be C
    //   CEmitter.preestablishPlacementConstraints(kernel, mapper);
    // }


    // std::filesystem::create_directory(kernelName + "_map_result");
    // CDFG->CDFGtoDOT(kernelName + "_map_result/before_map_" + CDFG->name_str() + "_CDFG.dot");
    // bool succeed = mapper->execute(/*dumpCallFunc=*/false, /*dumpMappedViz*/true, /*resultDir=*/kernelName + "_map_result");
    // // std::filesystem::create_directory("map_result");
    // // CDFG->CDFGtoDOT("map_result/before_map_" + CDFG->name_str() + "_CDFG.dot");
    // // bool succeed = mapper->execute(/*dumpCallFunc=*/false, /*dumpMappedViz*/true, /*resultDir=*/"map_result");
    // if(succeed){
    //   // Mapping is successful, get all blockload and blockstore op and corresponding spad memory addresses.
    //   if(emit_type == "pytest"){
    //     PyEmitter.setMapResult(kernel, mapper);
    //     PyEmitter.DataBlockOperationsToSPADInfo(kernel, mapper);
    //     PyEmitter.GenerateCGRAConfig(kernel, mapper);
    //   }
    //   else{ /// default to be C
    //     CEmitter.setMapResult(kernel, mapper);
    //     CEmitter.DataBlockOperationsToSPADInfo(kernel, mapper);
    //     CEmitter.GenerateCGRAConfig(kernel, mapper);
    //   }
    // }
    // kernel_cnt++;
  });

}

}
}