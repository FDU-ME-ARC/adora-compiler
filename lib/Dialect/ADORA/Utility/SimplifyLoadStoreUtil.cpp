//===- SimplifyLoadStoreUtil.cpp - Utilities of SimplifyLoadStore Pass -----------===//
#include "mlir/Dialect/Affine/IR/AffineOps.h"
// #include "mlir/Dialect/Affine/Analysis/Utils.h"
// #include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
// #include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "mlir/Support/LLVM.h"
// #include "mlir/Support/FileUtilities.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/RegionUtils.h"
// #include "mlir/Transforms/RegionUtils.h"
// #include "mlir/Transforms/DialectConversion.h"

// #include <iostream>
// #include <fstream>
// #include <filesystem>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Debug.h"

#include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "ADORA/Dialect/ADORA/Transforms/Passes.h"
#include "ADORA/Dialect/ADORA/Transforms/DSE.h"
#include "ADORA/Dialect/ADORA/Transforms/SimplifyLoadStore.h"

using namespace llvm; // for llvm.errs()
using namespace mlir;
using namespace mlir::affine;
using namespace mlir::ADORA;

#define DEBUG_TYPE "adora-simplify-loadstore"
//===----------------------------------------------------------------------===//
// SimplifyLoadStoreInLoopNest
//===----------------------------------------------------------------------===//
/** 
 * In this pass, we simplify the load and store operations with two methods:
 * First, hoist loop store operations to outer level, from innermost to outermost level.
 * Second, remove redundant store-load.
*/


namespace
{

} // namespace


bool TwoLoadsAccessSameMemAddr(AffineLoadOp load0, AffineLoadOp load1) {
  return TwoAccessSameMemAddr(load0, load1);
}
bool StoreLoadAccessSameMemAddr(AffineStoreOp store0, AffineLoadOp load1) {
  return TwoAccessSameMemAddr(store0, load1);
}
bool LoadStoreAccessSameMemAddr(AffineLoadOp load0, AffineStoreOp store1) {
  return TwoAccessSameMemAddr(load0, store1);
}

static bool ConsecutiveLoadsAccessSameMemAddr(AffineLoadOp load0, AffineLoadOp load1){
  return ConsecutiveAccessSameMemAddr<AffineLoadOp, AffineLoadOp>(load0, load1);
}
static bool ConsecutiveStoreLoadAccessSameMemAddr(AffineStoreOp store0, AffineLoadOp load1){
  return ConsecutiveAccessSameMemAddr<AffineStoreOp, AffineLoadOp>(store0, load1);
}
static bool ConsecutiveStoresAccessSameMemAddr(AffineStoreOp store0, AffineStoreOp store1){
  return ConsecutiveAccessSameMemAddr<AffineStoreOp, AffineStoreOp>(store0, store1);
}

/// Remove Redundant Loads
template <typename RegionOpT>
static bool RemoveRedundantLoads(RegionOpT RegionOp){
  LLVM_DEBUG(llvm::errs() << "Before RemoveRedundantLoads:\n");
  LLVM_DEBUG(RegionOp.dump());

  //// remove Consecutive Loads which access the same mem Addr
  bool NoChange = false;
  while(!NoChange){ // Keep walking the func until no change occurs in this func
    NoChange = true;

    for(AffineLoadOp load : RegionOp.template getOps<AffineLoadOp>()){
      for(AffineLoadOp otherload : RegionOp.template getOps<AffineLoadOp>()){
        if(load == otherload){ continue; }
        else if(ConsecutiveLoadsAccessSameMemAddr(load, otherload)){
          otherload.getOperation()->replaceAllUsesWith(load);
          otherload->erase();
          NoChange = false;
          break;
        }
      }

      if(!NoChange){ break; }
    }
  }

  LLVM_DEBUG(llvm::errs() << "After RemoveRedundantLoads:\n");
  LLVM_DEBUG(RegionOp.dump());

  //// remove Consecutive store-load which access the same mem Addr
  NoChange = false;
  while(!NoChange){ // Keep walking the func until no change occurs in this func
    NoChange = true;

    for(AffineStoreOp store : RegionOp.template getOps<AffineStoreOp>()){
      for(AffineLoadOp load : RegionOp.template getOps<AffineLoadOp>()){
        if(store == load){ continue; }
        else if(ConsecutiveStoreLoadAccessSameMemAddr(store, load)){
          // Replace the load result with the stored SSA value.
          load.getResult().replaceAllUsesWith(store.getValue());
          load->erase();
          NoChange = false;
          break;
        }
      }
      
      if(!NoChange){ break; }
    }
  }


  LLVM_DEBUG(llvm::errs() << "After RemoveRedundantstore-load:\n");
  LLVM_DEBUG(RegionOp.dump());

  return true;
}


/// Remove Redundant Loads
template <typename RegionOpT>
static bool RemoveRedundantStores(RegionOpT RegionOp){
  LLVM_DEBUG(llvm::errs() << "Before RemoveRedundantStores:\n");
  LLVM_DEBUG(RegionOp.dump());

  //// remove Consecutive Loads which access the same mem Addr
  bool NoChange = false;
  while(!NoChange){ // Keep walking the func until no change occurs in this func
    NoChange = true;

    for(AffineStoreOp store : RegionOp.template getOps<AffineStoreOp>()){
      for(AffineStoreOp otherstore : RegionOp.template getOps<AffineStoreOp>()){
        if(store == otherstore){ continue; }
        else if(ConsecutiveStoresAccessSameMemAddr(store, otherstore)){
          // otherstore.getOperation()->replaceAllUsesWith(store);
          store->erase();
          NoChange = false;
          break;
        }
      }

      if(!NoChange){ break; }
    }
  }

  LLVM_DEBUG(llvm::errs() << "After RemoveRedundantStores:\n");
  LLVM_DEBUG(RegionOp.dump());

  return true;
}

void mlir::ADORA::SimplifyLoadStoreOpsInRegion(mlir::Region& region){
  SmallVector<ADORA::KernelOp> kernels;
  region.walk([&](ADORA::KernelOp kernel){
    kernels.push_back(kernel);
  });

  for(ADORA::KernelOp kernel: kernels){
    SmallVector<ADORA::ForNode> ForNodes = createAffineForTreeInsideKernel(kernel);
    for(auto n : ForNodes){
      n.dumpForOp();
      mlir::affine::AffineForOp forop = n.getForOp();
      RemoveRedundantLoads(forop);
      HoistLoadStoreOpsInOp(forop);
    }
  }

  for(ADORA::KernelOp kernel: kernels){
    SmallVector<ADORA::ForNode> ForNodes = createAffineForTreeInsideKernel(kernel);
    for(auto n : ForNodes){
      // n.dumpForOp();
      mlir::affine::AffineForOp forop = n.getForOp();
      RemoveRedundantStores(forop);
    }
  }


  return ;
}