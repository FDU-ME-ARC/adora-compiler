//===- TileAssignment.h - per-kernel tile assignment for task overlap -----===//
//
// Decides, for each ADORA.kernel in a func, WHICH tile(s) it should run on, and
// writes the decision as an "adora.tile_set" DenseI64ArrayAttr on the KernelOp.
// Independent kernels can be placed on different tiles to overlap (run in
// parallel).  The LLM ranker (reused via Analysis/LLMRankerClient.h) makes the
// choice; the compiler provides per-kernel resource counts + a tile lower bound
// and validates / falls back on any illegal response.
//
// This is a set of helper functions, NOT a separate pass — it is invoked from
// LLMPipelineSchedulePass::runOnOperation() before the dep_type logic.
//
//===----------------------------------------------------------------------===//
#ifndef ADORA_DIALECT_ADORA_TRANSFORMS_TASKPIPELINE_TILEASSIGNMENT_H_
#define ADORA_DIALECT_ADORA_TRANSFORMS_TASKPIPELINE_TILEASSIGNMENT_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace ADORA {

/// Walk all KernelOps in `func`, decide each kernel's tile_set, and write it as
/// the "adora.tile_set" attribute (DenseI64ArrayAttr).
///
/// @param func        the function to process
/// @param rankerCmd   ranker subprocess argv (empty => dry-run, no LLM call)
/// @param timeoutMs   LLM call timeout
/// @param numTiles    total tiles on the CGRA (= adg->tileNum())
/// @param pePerTile   GPEs per tile (= adg->numGpeNodes() / adg->tileNum())
/// @param dryRun      if true, skip LLM and assign [0..minTiles-1] per kernel
/// @param logPath     optional NDJSON decision log path ("" = no log)
void assignTiles(mlir::func::FuncOp func, llvm::StringRef rankerCmd,
                 int timeoutMs, int numTiles, int pePerTile,
                 bool dryRun, llvm::StringRef logPath);

} // namespace ADORA
} // namespace mlir

#endif // ADORA_DIALECT_ADORA_TRANSFORMS_TASKPIPELINE_TILEASSIGNMENT_H_
