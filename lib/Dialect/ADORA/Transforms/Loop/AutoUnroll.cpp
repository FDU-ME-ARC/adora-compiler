//===- AutoUnroll.cpp - Code to perform automated check between loop unroll, or unroll and jam ---------===//

#include "mlir/Dialect/Affine/Passes.h"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/FileSystem.h"
#include <optional>
#include <filesystem>
#include <fstream>

#include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "ADORA/Dialect/ADORA/Utility/Utility.h"
#include "ADORA/Dialect/ADORA/Transforms/DSE.h"
#include "ADORA/Dialect/ADORA/Transforms/Passes.h"
#include "ADORA/Misc/DFG.h"
#include "../PassDetail.h"

#define DEBUG_TYPE "adora-auto-unroll"

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::ADORA;

namespace {
  /// Automated unroll pass. Decide on whether to use unroll or unroll and jam
  /// with dependency analysis
  struct ADORAAutoUnroll
      : public ADORAAutoUnrollBase<ADORAAutoUnroll> {
    unsigned NumGPE = 0;
    unsigned NumIOB = 0; 
    explicit ADORAAutoUnroll() { }
    SmallVector<SmallVector<unsigned>> ConstructUnrollSpaceFromStrategy(SmallVector<ADORA::ForNode> ForNodes);
    LogicalResult chooseAndApplyUnrollStrategyWithDeps(ADORA::KernelOp kernel, mlir::ModuleOp& m);
    bool KernelIsInPerfectNestedLoop(ADORA::KernelOp kernel);
    void runOnOperation() override;
  };
} // namespace


std::unique_ptr<OperationPass<mlir::ModuleOp>>
    mlir::ADORA::createADORAAutoUnrollPass() {
  return std::make_unique<ADORAAutoUnroll>();
}

//////////
// Return true if operand "range" contains "value".
/////////
bool OperandRangeContainsValue(::mlir::Operation::operand_range range,  mlir::Value value){
  for(auto v : range){
    if(value == v)
      return true;
  }
  return false;
}

bool ADORAAutoUnroll::KernelIsInPerfectNestedLoop(ADORA::KernelOp kernel){
  mlir::Operation* parent = kernel.getOperation()->getParentOp();
  if(!isa_and_present<AffineForOp>(parent))
    return false;
  
  AffineForOp parentfor = dyn_cast<AffineForOp>(parent);

  // We already know that the block can't be empty.
  auto perfectNested = [](Block *block) {
    int nonKernelOpCount = 0;
    auto nextOpIt = block->begin(); 
    while(nextOpIt != block->end()){
      if(!( isa<ADORA::KernelOp>(nextOpIt) 
          ||isa<ADORA::DataBlockLoadOp>(nextOpIt) 
          ||isa<ADORA::DataBlockStoreOp>(nextOpIt)
          ||isa<ADORA::LocalMemAllocOp>(nextOpIt)) )
      {
        nonKernelOpCount++;
      }

      nextOpIt = std::next(nextOpIt);
    }

    return nonKernelOpCount == 1;
  };

  // parentForOp's body should be just this kernel and the terminator.
  if (!perfectNested(parentfor.getBody()))
    return false;

  return true;
}

/// @brief A design point is a number sequence:
///        tilefactor(loop0),unrollfactor(loop0),tilefactor(loop1),unrollfactor(loop1).......
/// for example:
///       (1,1),(1,1),(1,3) represents a 3-level nested loop and the innermost loop is unrolled by 3
/// @param ForNodes All for loops
/// @return A SmallVector containing all design point
SmallVector<SmallVector<unsigned>> ADORAAutoUnroll::ConstructUnrollSpaceFromStrategy(SmallVector<ADORA::ForNode> ForNodes){
  using namespace llvm; // for llvm.errs()
  SmallVector<SmallVector<unsigned>> AllDesignSpace;

  // unsigned point_num = 1;
  for (int i = 0; i < ForNodes.size(); i++) {
    ADORA::ForNode n = ForNodes[i];
    assert(ForNodes[i].IsOutOfKernel() || ForNodes[i].getUnrollStrategy() != UnrollStrategy::Undecided);

    if(ForNodes[i].getUnrollStrategy() == UnrollStrategy::Unroll){
      /// Find all unroll factor
      FindUnrollingFactors(ForNodes[i]); 

      UnrollDesignPoint point;

      /// For loops outer than this level, all factor will be 1.
      for(int j = 0; j < i; j++){
        point.push_back(ForNodes[j].getMaxUnrollFactor());
      }
      point.push_back(0); // point[i]
      /// For loops inner than this level, all factor will be tripcounts(fully unroll).
      for(int j = i + 1; j < ForNodes.size(); j++){
        point.push_back(1);
      }

      /// For this level loop,store all feasible factor.
      for(auto UF: ForNodes[i].UnrollFactors){
        point[i] = UF;
        if(findElement(AllDesignSpace, point) == -1){
          AllDesignSpace.push_back(point);
        }
      }
    }
    else if(ForNodes[i].getUnrollStrategy() == UnrollStrategy::Unroll_and_Jam){
      if(i == ForNodes.size() - 1) /// there is no loop level out of kernel
        break;
        
        
      assert(i < ForNodes.size() - 1); 
      ADORA::ForNode parentForNode = ForNodes[ForNodes.size() - 1];
      if(parentForNode.IsOutOfKernel()){
        /// Kernel is in one affine for op
        SmallVector<unsigned> parentForUnrollFactors = FindUnrollingFactors(parentForNode);
        
        UnrollDesignPoint point;
        /// For loops inner than this level, all factor will be tripcounts(fully unroll).
        for(int j = 0; j < i; j++){
          point.push_back(ForNodes[j].getMaxUnrollFactor());
        }

        /// For loops equal to or outer than this level, all factor will be 1.
        for(int j = i; j < ForNodes.size() - 1; j++){
          point.push_back(1);
        }

        /// For parent level loop,store all feasible factor.
        point.push_back(0);
        for(auto UF: parentForUnrollFactors){
          point[ForNodes.size() - 1] = UF;
          if(findElement(AllDesignSpace, point) == -1){
            AllDesignSpace.push_back(point);
          }
        }
      }
      // ForNodes[i].UnrollFactors.push_back(1);
      /// if one level can't be unrolled, then the outer level can't be unrolled either.
      break;
    }
    else if(ForNodes[i].getUnrollStrategy() == UnrollStrategy::CannotUnroll){
      ForNodes[i].UnrollFactors.push_back(1);
      /// If this level can't be unrolled, then out level can't be unrolled either.
      break;
    }
  }

  // /*** Print Design Space ***/
  llvm::errs() <<"//-----------  Design Space  ----------//\n";
  llvm::errs() <<"Loop-nested structure:\n";
  for (ADORA::ForNode Node : ForNodes) {
    if (!Node.HasParentFor())/// outermost loop
      Node.dumpTree();
  }

  llvm::errs() <<"Design Space:\n";
  for(UnrollDesignPoint point : AllDesignSpace){
    for (int i = 0; i < ForNodes.size(); i++) {
      llvm::errs() << point[i] << " ";
    }
    llvm::errs() << "\n";
  }

  llvm::errs() <<"//-------------------------------------//\n";
  return AllDesignSpace;
}

//////////
// A wrapper for KernelUnroll
/////////
LogicalResult ADORAAutoUnroll::
chooseAndApplyUnrollStrategyWithDeps(ADORA::KernelOp kernel, mlir::ModuleOp& m){
  /// make a new dir
  SmallVector<std::string> DesignSpaceFiles;
  std::filesystem::path currentPath = std::filesystem::current_path();
  std::string folderName = "DesignSpace";
  std::filesystem::path DesignSpacefolderPath = currentPath / folderName;
  std::filesystem::create_directory(DesignSpacefolderPath);
  int max_ALU = 0, max_LSU = 0;
  std::string final_FilePath="__";

  // Collect all legal candidates for LLM ranking.
  struct UnrollCandidate {
    std::string filePath;
    std::string fileName;
    int num_alu;
    int num_lsu;
  };
  SmallVector<UnrollCandidate> legalCandidates;
  int defaultIdx = -1; // index of greedy-best in legalCandidates

  ///////////////
  /// For store op, check whether exists port conflict 
  ///////////////
  SmallVector<ADORA::ForNode> ForNodes = createAffineForTreeAroundKernel(kernel);
  // ForNodes[0].dumpTree();

  //// Decide the unroll strategy of each level(Unroll or Unroll-and-Jam)
  // SmallVector<UnrollStrategy> urStrategies;
  for(ADORA::ForNode& node: ForNodes){
    if(node.IsOutOfKernel())  /// Out of kernel
      continue;
    UnrollStrategy strategy;
    AffineForOp forop = node.getForOp();

    uint64_t StripCount = getConstantTripCount(forop).value_or(0);
    /// StripCount = 0, means the loop is not constant strip count, can't be unrolled
    if(StripCount == 0){
      strategy = UnrollStrategy::CannotUnroll;
    } 
    /// StripCount = 1, skip
    else if(StripCount == 1){
      strategy = UnrollStrategy::Unroll;
    }
    else{
      /// choose Unroll Strategy
      strategy = UnrollStrategy::Unroll;
      mlir::Value IterValue = forop.getSingleInductionVar().value();
      forop.walk([&](AffineStoreOp storeop){
        if(OperandRangeContainsValue(storeop.getIndices(), IterValue)){
          //// port conflict exists
          strategy = UnrollStrategy::Unroll_and_Jam;
          WalkResult::interrupt();
        }

        /// TODO: when to return CannotUnroll
      });
    }
    /// If kernel is not in a perfectly nested loop, cannot unroll.
    if(strategy == UnrollStrategy::Unroll_and_Jam && !KernelIsInPerfectNestedLoop(kernel)){
      strategy = UnrollStrategy::CannotUnroll;
    }
    node.setUnrollStrategy(strategy);
    // urStrategies.push_back(strategy);
    // urStrategies.push_back(UnrollStrategy::CannotUnroll);
  }

  ///// push parent for op 
  // affine::AffineForOp Parentfor = dyn_cast_or_null<affine::AffineForOp> 
  //                                 (kernel.getOperation()->getParentOp()); 
  // SmallVector<unsigned> parentForUnrollFactors;
  // if(Parentfor != nullptr){
  //   ADORA::ForNode parentForNode(Parentfor, -1/*Level: out of kernel*/);
  //   parentForUnrollFactors = FindUnrollingFactors(parentForNode);
  // }

  SmallVector<SmallVector<unsigned>> UnrollDesignSpace;
  UnrollDesignSpace = ConstructUnrollSpaceFromStrategy(ForNodes);
  //// Traverse the design space to unroll step by step from innermost to outermost.
  //// Unroll and jam if loops inside kernel cann't be unrolled 
  for(UnrollDesignPoint point : UnrollDesignSpace){
    ModuleOp topmodule = cast<ModuleOp>(m.getOperation()->clone());
    ADORA::KernelOp kernelur = getKernelFromCopiedModule(topmodule, kernel);
  
    SmallVector<ADORA::ForNode> ForNodesUR = createAffineForTreeAroundKernel(kernelur);
    std::string fileName;
    if(kernelur.hasKernelName())
      fileName = kernelur.getKernelName();
    else
      fileName = "Unroll";
    
    /// Unroll every level from innermost to outermost
    for(unsigned node_cnt = 0; node_cnt < ForNodesUR.size(); node_cnt++){
      /// update filename 
      fileName += "_" + std::to_string(point[node_cnt]);
      // llvm::errs() <<"filename:" << fileName << "\n";
      /// Skip the tiling factor
      
      /// Apply unrolling by this factor
      unsigned unrollfactor = point[node_cnt];
      ADORA::ForNode NodeToUnroll = ForNodesUR[node_cnt];
      NodeToUnroll.dumpForOp();

      /// if this level could be unrolled
      if(unrollfactor > 1 && !NodeToUnroll.IsOutOfKernel()){
        if(failed(ADORA::loopUnrollByFactor_opt(NodeToUnroll.getForOp(), 
              unrollfactor, /*annotateFn=nullptr,*/ /*cleanUpUnroll=*/false))){
            llvm::errs() << "Unroll failed!!!\n";
            signalPassFailure();
            return LogicalResult::failure();
        }
      }
      /// if this level is out of kernel and could be unroll-jam
      else if(unrollfactor > 1 && NodeToUnroll.IsOutOfKernel()){
        if(failed(ADORA::loopUnrollAndJamByFactor(NodeToUnroll.getForOp(), unrollfactor))){
            llvm::errs() << "Unroll failed!!!\n";
            signalPassFailure();
            return LogicalResult::failure();
        }
      }
      // llvm::errs() << "[debug] new for:\n";
      // NodeToUnroll.dumpForOp();
    }

    /// create a new mlir file
    std::string filePath = DesignSpacefolderPath.string() + "/" + fileName + ".mlir";
    // llvm::errs() << "filePath: " << filePath << "\n";
    std::error_code ec;
    llvm::raw_fd_ostream outputFile(filePath, ec, sys::fs::FA_Write);
    if (ec) {
      llvm::errs() << "Error opening file: " << ec.message() << filePath << "\n";
      signalPassFailure();
      return LogicalResult::failure();
    }
    DesignSpaceFiles.push_back(fileName);
  
    // llvm::errs() << "[Info] design point " << fileName << ":\n";
    topmodule.dump();
  
    simplifyConstantAffineApplyOpsInRegion(kernelur.getBody());
    simplifyLoadAndStoreOpsInRegion(kernelur.getBody());
    
    topmodule.print(outputFile);
    topmodule.dump();      
  
    /// Generating DFG
    std::string GeneralOpNameFile_str;
    if (GeneralOpNameFile == nullptr) {
      std::cerr << "Environment variable \"GeneralOpNameFile\" is not set." << std::endl;
      GeneralOpNameFile_str = "lib/DFG/Documents/GeneralOpName.txt";
      std::cerr << "Using fallback: \"" << GeneralOpNameFile_str << "\"" << std::endl;
    }
    else
      GeneralOpNameFile_str = GeneralOpNameFile;
    LLVMCDFG *CDFG = new LLVMCDFG(fileName, GeneralOpNameFile_str);
    generateCDFGfromKernel(CDFG, kernelur, /*verbose=*/false);
    CDFG->CDFGtoDOT(DesignSpacefolderPath.string() + "/" + CDFG->name_str()+"_CDFG_unroll.dot");
  
    ADORA::DFGInfo dfginfo = GetDFGinfo(CDFG);       
    if(dfginfo.Num_ALU <= NumGPE && dfginfo.Num_LSU <= NumIOB) 
    {
      // Track greedy-best (max LSU, then max ALU) as the default fallback.
      if(dfginfo.Num_LSU > max_LSU || 
        (dfginfo.Num_ALU > max_ALU && dfginfo.Num_LSU == max_LSU))
      {
        final_FilePath = filePath;
        max_LSU = dfginfo.Num_LSU;
        max_ALU = dfginfo.Num_ALU;
        defaultIdx = (int)legalCandidates.size(); // will be the next push index
      }
      legalCandidates.push_back({filePath, fileName,
                                  (int)dfginfo.Num_ALU, (int)dfginfo.Num_LSU});
    }
    else{
      /// If the resource occupied by current dfg oversizes adg,
      /// we don't have to keep unrolling to get larger dfg.
      break;
    }

    // topmodule.erase();
  }/// End of traversing on every design point

  // --- LLM unroll selection ---
  // Write candidates.json so an external ranker can pick one.
  // Read selected.json if present; fall back to greedy best otherwise.
  if (!legalCandidates.empty()) {
    std::string candidatesPath = DesignSpacefolderPath.string() + "/candidates.json";
    std::string selectedPath   = DesignSpacefolderPath.string() + "/selected.json";

    // Write candidates.json (minimal hand-rolled JSON, no external deps).
    {
      std::ofstream out(candidatesPath);
      out << "{\n  \"candidates\": [\n";
      for (size_t i = 0; i < legalCandidates.size(); ++i) {
        const auto &c = legalCandidates[i];
        out << "    {\"index\": " << i
            << ", \"filePath\": \"" << c.filePath << "\""
            << ", \"fileName\": \"" << c.fileName << "\""
            << ", \"num_alu\": " << c.num_alu
            << ", \"num_lsu\": " << c.num_lsu << "}";
        if (i + 1 < legalCandidates.size()) out << ",";
        out << "\n";
      }
      out << "  ],\n  \"default_index\": " << defaultIdx << "\n}\n";
    }

    // Try to read selected.json written by external ranker.
    std::ifstream sel(selectedPath);
    if (sel.is_open()) {
      std::string line, content;
      while (std::getline(sel, line)) content += line;
      // Parse "selected_index": N — simple scan, no JSON lib needed.
      auto pos = content.find("\"selected_index\"");
      if (pos != std::string::npos) {
        pos = content.find(':', pos);
        if (pos != std::string::npos) {
          int idx = std::stoi(content.substr(pos + 1));
          if (idx >= 0 && idx < (int)legalCandidates.size()) {
            final_FilePath = legalCandidates[idx].filePath;
            llvm::errs() << "[AutoUnroll] LLM selected index " << idx
                         << " (" << legalCandidates[idx].fileName << ")\n";
          } else {
            llvm::errs() << "[AutoUnroll] LLM index " << idx
                         << " out of range, using greedy default\n";
          }
        }
      }
    }
  }
  // --- end LLM unroll selection ---

  /// try to unroll and jam parent for of kernel
  // kernel.walk([&](AffineStoreOp storeop){
  //   // kernels.push_back(kernel);
  //   MemRefAccess storeAccess(storeop);
  //   unsigned depth = getNestingDepth(kernel.getTopAffineFopInKernel());
  
  if(final_FilePath != "__"){
    //// Apply the optimal unroll result
    std::string errorMessage;
    auto file = openInputFile(final_FilePath, &errorMessage);
    if (!file) {
      llvm::errs() << errorMessage << "\n";
      assert(0);
    }
    
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());
    mlir::OwningOpRef<mlir::ModuleOp> final_m = parseSourceFile<ModuleOp>(sourceMgr, m.getContext()); 
    if (!final_m) {
      llvm::errs() << "Failed to parse selected design-point module: "
                   << final_FilePath << "\n";
      signalPassFailure();
      return LogicalResult::failure();
    }
    mlir::ModuleOp moduleop = final_m.get();

    // Replace the function that owns the current kernel by symbol name.
    // Do not replace the first function blindly: module order may be
    // @main_graph then @kernel_func, and replacing by index can miss the kernel.
    func::FuncOp oldfunc = kernel->getParentOfType<func::FuncOp>();
    if (!oldfunc) {
      llvm::errs() << "Cannot locate parent function for kernel.\n";
      signalPassFailure();
      return LogicalResult::failure();
    }
    func::FuncOp newfunc = moduleop.lookupSymbol<func::FuncOp>(oldfunc.getSymName());
    if (!newfunc) {
      llvm::errs() << "Cannot locate function '" << oldfunc.getSymName()
                   << "' in selected design-point module.\n";
      signalPassFailure();
      return LogicalResult::failure();
    }
    newfunc.getOperation()->moveBefore(oldfunc);
    oldfunc.getOperation()->erase();
  }
 

  return LogicalResult::success();
}

void ADORAAutoUnroll::runOnOperation() {
  ////////////
  /// get hardware info
  ///////////
  if(CGRAadg == "notdefined" || CGRAadg == ""){
    LLVM_DEBUG(llvm::errs() << "CGRAadg not defined.\n");
    NumGPE = 32;
    NumIOB = 16;
  }
  else{
    NumGPE = getInstanceNumFromADG(CGRAadg, "GPE");
    NumIOB = getInstanceNumFromADG(CGRAadg, "IOB");
  }
  // if (unrollJamFactor)
  //   this->unrollJamFactor = *unrollJamFactor;

  // if (getOperation().isExternal())
  //   return;
  auto m = getOperation();
  // func.dump();
  SmallVector<ADORA::KernelOp> kernels;
  m.walk([&](ADORA::KernelOp kernel){
    kernels.push_back(kernel);
  });

  // Iterate by index: after each chooseAndApplyUnrollStrategyWithDeps we replace
  // the function, so kernel pointers in `kernels` become invalid. Re-fetch the
  // i-th kernel from the current module each time.
  for (size_t i = 0; i < kernels.size(); ++i) {
    ADORA::KernelOp currentKernel;
    size_t count = 0;
    m.walk([&](ADORA::KernelOp k) {
      if (count == i) {
        currentKernel = k;
        return WalkResult::interrupt();
      }
      count++;
      return WalkResult::advance();
    });
    if (!currentKernel)
      continue;
    LogicalResult r = chooseAndApplyUnrollStrategyWithDeps(currentKernel, m);
    if (failed(r))
      return;
    m.dump();
  }

  for(auto func : m.getOps<func::FuncOp>()){
    ResetIndexOfBlockAccessOpInFunc(func);
  }
  m.dump();
}
