#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Debug/CLOptionsSetup.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/FileUtilities.h" 
#include "llvm/Support/MemoryBuffer.h"

#include "../../lib/DFG/inc/mlir_cdfg.h"
#include "ADORA/Dialect/ADORA/IR/ADORA.h"
// #include "ADORA/Dialect/ADORA/Transforms/Passes.h"
#include "ADORA/Dialect/ADORA/Transforms/Passes.h"
// #include "ADORA/Dialect/ADORA/Lowering/LowerPasses.h"
#include "ADORA/Dialect/ADORATensor/IR/ADORATensor.h"
#include "ADORA/Misc/Passes.h"
#include "ADORA/Misc/DFG.h"

#include <iostream>
#include <set>
#include <cstdlib>
#include <ctime>
#include <regex>
#include <sstream>
#include <thread>
#include <mutex>
#include <getopt.h>
#include <atomic>
#include <algorithm>
#include <filesystem>

#include "op/operations.h"
#include "ir/adg_ir.h"
#include "ir/dfg_ir.h"
#include "mapper/mapper_sa.h"
#include "mapper/agent_trace.h"
#include "mapper/online_ranker.h"
#include "spdlog/spdlog.h"
#include "spdlog/cfg/argv.h"
#include "emit/EmitCGRACall.h"
#include "emit/EmitPytest.h"
#include "emit/EmitVitisSDK.h"
#include "tensorop/TensorOp.h"

// #include "mlir/Dialect/Arith/Transforms/Passes.h"
// #include "mlir/Dialect/Func/Transforms/Passes.h"

// Defined in the test directory, no public header.
namespace mlir {
} // namespace mlir

using namespace llvm;
using namespace mlir;

// static int kernel_cnt = 0;

int main(int argc, char **argv) {
  // mlir::registerAllDialects();
  // mlir::registerAllPasses();

  mlir::DialectRegistry registry;

  //===--------------------------------------------------------------------===//
  // Register mlir dialects and passes
  //===--------------------------------------------------------------------===//
  // Add the following to selectively include the necessary dialects. You only
  // need to register dialects that will be *parsed* by the tool, not the one
  // generated
  // clang-format off
  registry.insert<mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect,
                  mlir::LLVM::LLVMDialect,
                  mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect,
                  mlir::scf::SCFDialect,
                  mlir::cf::ControlFlowDialect,
                  mlir::vector::VectorDialect,
                  mlir::arith::ArithDialect,
                  mlir::affine::AffineDialect,
                  mlir::DLTIDialect,
                  mlir::ml_program::MLProgramDialect,
                  mlir::tensor::TensorDialect,
                  mlir::bufferization::BufferizationDialect>();

  // Dialects
  registry.insert<mlir::ADORA::ADORADialect,
                  mlir::ADORA::ADORATensor::ADORATensorDialect>();
  // return failed(
  //     mlir::MlirOptMain(argc, argv, "Fail\n", registry)
  // );

  //===--------------------------------------------------------------------===//
  // Similar to MlirOptMian() in MlirOptMain.cpp
  //===--------------------------------------------------------------------===//

  /// User args
  /// A good example to use cgra-mapper is:
  ///  
  static cl::opt<std::string> inputFilename(
    cl::Positional, 
    cl::desc("<input file>"), 
    cl::init("-"));

  // static cl::opt<bool> dumpCallFunc(
  //   "dump-call-func",
  //   cl::Optional, 
  //   cl::desc("dump call function of CGRA (default)"), 
  //   cl::init(false));

  static cl::opt<bool> dumpMappedViz(
    "dump-mapped-viz",
    cl::Optional, 
    cl::desc("dump-mapped-viz"), 
    cl::init(true));
  
  static cl::opt<bool> objOpt(
    "obj-opt",
    cl::Optional, 
    cl::desc("obj-opt"), 
    cl::init(true));

  static cl::opt<int> timeout_ms(
    "timeout",
    cl::Optional, 
    cl::desc("timeout(ms)"), 
    cl::value_desc("int"), 
    cl::init(360000));

  static cl::opt<int> max_iters(
    "max-iters",
    cl::Optional, 
    cl::desc("max-iters"), 
    cl::value_desc("int"), 
    cl::init(2000));
    
  static cl::opt<std::string> adg_fn(
    "adg",
    cl::Required, 
    cl::desc("adg file"), 
    cl::value_desc("adg filename"), 
    cl::init("-"));

  static cl::opt<std::string> op_fn(
    "op-file",
    cl::Required, 
    cl::desc("op file"), 
    cl::value_desc("op filename"), 
    cl::init("-"));

  static cl::opt<std::string> emit_type(
    "output-type",
    cl::Required, 
    cl::desc("emit the execution file type: c(defualt), pytest, sdk(vitis sdk)"), 
    cl::value_desc("c/pytest/sdk"), 
    cl::init("pytest"));

  static cl::opt<std::string> outputFilename(
    "output", 
    cl::Optional, 
    cl::desc("Output filename"),
    cl::value_desc("filename"),
    cl::init("-"));

  static cl::opt<bool> verbose(
    "verbose", 
    cl::Optional, 
    cl::desc("Detail information"),
    cl::value_desc("bool"),
    cl::init(false));

  // PR6.4 — opt-in switch to run schedule-tasks + assign-streams +
  // lower-async-tokens on the module right before per-kernel mapping &
  // emit. Default off keeps cgra-mapper byte-identical to pre-PR6.
  static cl::opt<bool> enableAsync(
    "enable-async",
    cl::Optional,
    cl::desc("Run --adora-schedule-tasks + --adora-assign-streams + "
             "--adora-lower-async-tokens before emit so PR6.4 dep_summary "
             "path drives BlockStore await-gather. Default false."),
    cl::value_desc("bool"),
    cl::init(false));

  static cl::opt<bool> enableLLMSchedule(
    "enable-llm-schedule",
    cl::Optional,
    cl::desc("After schedule-tasks, run --llm-pipeline-schedule so an LLM ranker "
             "picks per-task hw_dep_type (drives BlockStore await-gather in the "
             "Python emit). Requires --enable-async. Default false. The ranker "
             "command / dry-run / timeout are controlled by the global "
             "--llm-pipeline-schedule-* flags."),
    cl::value_desc("bool"),
    cl::init(false));

  static cl::opt<int> specifictilenum(
    "tile",
    cl::Optional, 
    cl::desc("tile num to map"), 
    cl::value_desc("int"), 
    cl::init(9999999));
  // static cl::opt<int> nthreads(
  //   "j", 
  //   cl::Optional, 
  //   cl::desc("Allow N mapping jobs at once(default to be 1)"),
  //   cl::value_desc("[N]"),
  //   cl::init(1));
  static cl::opt<int> parallel_cores(
    "parallel-cores",
    cl::Optional, 
    cl::desc("Allow N mapping jobs at once within each func.func (default to be 1)"),
    cl::value_desc("[N]"),
    cl::init(1));

  static cl::opt<bool> emitAgentTrace(
    "emit-agent-trace",
    cl::Optional,
    cl::desc("emit mapper-agent trace JSONL sidecar artifacts"),
    cl::init(false));

  static cl::opt<std::string> agentTraceRoot(
    "agent-trace-root",
    cl::Optional,
    cl::desc("root directory for mapper-agent trace artifacts"),
    cl::value_desc("path"),
    cl::init("mapper_agent_trace"));

  static cl::opt<std::string> agentTracePolicyId(
    "agent-trace-policy-id",
    cl::Optional,
    cl::desc("policy id recorded in mapper-agent trace"),
    cl::value_desc("string"),
    cl::init("compiler_logged"));

  static cl::opt<bool> agentOnlinePlacement(
    "agent-online-placement",
    cl::Optional,
    cl::desc("call an online ranker before each mapper_place_node decision"),
    cl::init(false));

  static cl::opt<std::string> agentRankerCmd(
    "agent-ranker-cmd",
    cl::Optional,
    cl::desc("shell-quoted command that runs the online ranker (stdin/stdout JSON)"),
    cl::value_desc("cmd"),
    cl::init(""));

  static cl::opt<int> agentRankerTimeoutMs(
    "agent-ranker-timeout-ms",
    cl::Optional,
    cl::desc("hard timeout for one ranker call"),
    cl::value_desc("ms"),
    cl::init(15000));

  static cl::opt<int> agentOnlineBudget(
    "agent-online-budget",
    cl::Optional,
    cl::desc("max number of placement decisions sent to the ranker (-1 = unlimited)"),
    cl::value_desc("int"),
    cl::init(-1));

  static cl::opt<std::string> agentOnlineLog(
    "agent-online-log",
    cl::Optional,
    cl::desc("JSONL log of every ranker request/response (optional)"),
    cl::value_desc("path"),
    cl::init(""));

  static cl::opt<std::string> agentRankerMode(
    "agent-ranker-mode",
    cl::Optional,
    cl::desc("ranker process mode: 'once' (subprocess per call) or 'daemon' (one long-lived child)"),
    cl::value_desc("once|daemon"),
    cl::init("once"));

  static cl::opt<std::string> agentAdgSummary(
    "agent-adg-summary",
    cl::Optional,
    cl::desc("if set, dump a compact ADG description JSON to this path and reference it from every online request (mapper-adg-v0)"),
    cl::value_desc("path"),
    cl::init(""));

  static cl::opt<std::string> opNameFile(
    "op-name-file",
    cl::Optional,
    cl::desc("MLIR-op to DFG-type name mapping file (overrides GENERAL_OP_NAME_ENV)"),
    cl::value_desc("filename"),
    cl::init("lib/DFG/Documents/GeneralOpName.txt"));
  // spdlog::cfg::helpers::load_levels("true");

  InitLLVM y(argc, argv);

  MlirOptMainConfig::registerCLOptions(registry);
  // registerAsmPrinterCLOptions();
  // registerMLIRContextCLOptions();
  // registerPassManagerCLOptions();
  // tracing::DebugCounter::registerCLOptions();

  // Build the list of dialects as a header for the --help message.
  std::string helpHeader = "\nAvailable Dialects: ";
  {
    llvm::raw_string_ostream os(helpHeader);
    interleaveComma(registry.getDialectNames(), os,
                    [&](auto name) { os << name; });
  }
  // Parse pass names in main to ensure static initialization completed.
  cl::ParseCommandLineOptions(argc, argv, helpHeader);
  MlirOptMainConfig config = MlirOptMainConfig::createFromCLOptions();

  if(emitAgentTrace){
    std::string stem = std::filesystem::path(std::string(inputFilename)).stem().string();
    if(stem.empty()) stem = "stdin";
    std::string runId = stem + "_" + std::to_string(std::time(nullptr));
    AgentTrace::configure(agentTraceRoot, runId);
    AgentTrace::emit(
      "cgra_mapper",
      "start",
      "{\"input\":\"" + agentTraceJsonEscape(std::string(inputFilename)) + "\",\"adg\":\"" + agentTraceJsonEscape(std::string(adg_fn)) + "\",\"op_file\":\"" + agentTraceJsonEscape(std::string(op_fn)) + "\",\"output_type\":\"" + agentTraceJsonEscape(std::string(emit_type)) + "\",\"policy_id\":\"" + agentTraceJsonEscape(std::string(agentTracePolicyId)) + "\",\"max_iters\":" + std::to_string((int)max_iters) + ",\"timeout_ms\":" + std::to_string((int)timeout_ms) + "}"
    );
  }

  if(agentOnlinePlacement){
    if(std::string(agentRankerCmd).empty()){
      llvm::errs() << "[agent-online-placement] --agent-ranker-cmd is required when online placement is enabled\n";
      return 1;
    }
    OnlineRanker::Mode mode = OnlineRanker::Mode::Once;
    std::string modeStr = std::string(agentRankerMode);
    if(modeStr == "daemon"){
      mode = OnlineRanker::Mode::Daemon;
    } else if(modeStr != "once"){
      llvm::errs() << "[agent-online-placement] --agent-ranker-mode must be 'once' or 'daemon'\n";
      return 1;
    }
    OnlineRanker::configure(std::string(agentRankerCmd),
                            (int)agentRankerTimeoutMs,
                            (int)agentOnlineBudget,
                            std::string(agentOnlineLog),
                            mode);
    if(AgentTrace::enabled()){
      std::ostringstream cfg;
      cfg << "{\"ranker_cmd\":\"" << agentTraceJsonEscape(std::string(agentRankerCmd))
          << "\",\"timeout_ms\":" << (int)agentRankerTimeoutMs
          << ",\"budget\":" << (int)agentOnlineBudget
          << ",\"mode\":\"" << agentTraceJsonEscape(modeStr)
          << "\",\"log\":\"" << agentTraceJsonEscape(std::string(agentOnlineLog)) << "\"}";
      AgentTrace::emit("cgra_mapper", "online_ranker_configured", cfg.str());
    }
  }



  // When reading from stdin and the input is a tty, it is often a user mistake
  // and the process "appears to be stuck". Print a message to let the user know
  // about it!
  MLIRContext context(registry, MLIRContext::Threading::DISABLED);
  context.getOrLoadDialect(mlir::ADORA::ADORADialect::getDialectNamespace());
  context.getOrLoadDialect(mlir::ADORA::ADORATensor::ADORATensorDialect::getDialectNamespace());

  if (inputFilename == "-" &&
      sys::Process::FileDescriptorIsDisplayed(fileno(stdin)))
    llvm::errs() << "(processing input from stdin now, hit ctrl-c/ctrl-d to "
                    "interrupt)\n";

  std::string errorMessage;
  
  Twine t = (StringRef)inputFilename;
  // openInputFileImpl(t, errorMessage,
  //                          /*alignment=*/std::nullopt);
  // openInputFile((StringRef)inputFilename, &errorMessage);

  llvm::MemoryBuffer::getFileOrSTDIN(t);
  llvm::MemoryBuffer::getFileOrSTDIN(
      t, /*IsText=*/false, /*RequiresNullTerminator=*/true,
       /*alignment=*/std::nullopt);

  if(verbose){
    spdlog::set_level(spdlog::level::debug);
  }
  else{
    spdlog::set_level(spdlog::level::off);
  }

  if(verbose) {t.dump();}


  /////////////////////////
  /// Parse input file
  /////////////////////////
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    assert(0);
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> m = parseSourceFile<ModuleOp>(sourceMgr, &context); 
  if(!m){
    assert(0 && "Error when parsing mlir file.");
  }
  mlir::ModuleOp moduleop = m.get();
  // SymbolTable symbolTable(moduleop.getOperation());
  
  //////////////////////////////////////////
  /// Parse Operation file and ADG file
  //////////////////////////////////////////
  unsigned seed = time(0); // random seed using current time
  srand(seed);  // set random generator seed 
  std::cout << "Parse Operations: " << op_fn << std::endl;
  Operations::Instance(op_fn);
  // Operations::print();

  std::cout << "Parse ADG: " << adg_fn << std::endl;
  ADGIR adg_ir(adg_fn);
  ADG* adg = adg_ir.getADG();
  int numGpeNodes = adg->numGpeNodes();
  int numFuNodes = numGpeNodes + adg->numIobNodes();
  int numTiles = adg->tileNum();
  std::cout << "numGpeNodes: " << numGpeNodes 
            << ", numFuNodes(GPE+IOB): "  << numFuNodes 
            << ", numTiles: "  << numTiles << std::endl;
  std::vector<float>storePEusage;
  std::vector<float>storeFUusage;
  std::vector<int>bestLatency;

  ADG* subadg = adg->inducedSubgraphByFirstNTiles(specifictilenum);
  subadg->print();

  // Dump a compact ADG description for the online ranker (see online_schema.py
  // mapper-adg-v0). The mapper sends `adg_ref = {adg_hash, adg_summary_ref}`
  // on every placement request so a long-lived ranker can cache by hash.
  if(agentOnlinePlacement && !std::string(agentAdgSummary).empty()){
    OnlineRanker::AdgContext ctx;
    ctx.adg_summary_path = std::string(agentAdgSummary);
    ctx.adg_hash = dumpAdgSummary(subadg, ctx.adg_summary_path);
    if(ctx.adg_hash.empty()){
      llvm::errs() << "[agent-adg-summary] failed to write " << ctx.adg_summary_path << "\n";
      return 1;
    }
    OnlineRanker::setAdgContext(ctx);
    if(AgentTrace::enabled()){
      std::ostringstream cfg;
      cfg << "{\"adg_summary_ref\":\"" << agentTraceJsonEscape(ctx.adg_summary_path)
          << "\",\"adg_hash\":\"" << agentTraceJsonEscape(ctx.adg_hash)
          << "\",\"node_count\":" << subadg->nodes().size()
          << ",\"num_gpe_nodes\":" << subadg->numGpeNodes()
          << ",\"num_iob_nodes\":" << subadg->numIobNodes()
          << ",\"tile_num\":" << subadg->tileNum() << "}";
      AgentTrace::emit("cgra_mapper", "adg_summary_emitted", cfg.str());
    }
  }

  //////////////////////////////////////////
  /// Pre-set mapping
  //////////////////////////////////////////
  CGRACallEmitter CEmitter(moduleop);
  PytestEmitter PyEmitter(moduleop);
  VitisSDKEmitter SDKEmitter(moduleop);

  CEmitter.setTotalTileNum(numTiles);
  PyEmitter.setTotalTileNum(numTiles);
  SDKEmitter.setTotalTileNum(numTiles);
  
  std::vector<MapperSA*>mapper_Vec;
  std::vector<DFGIR*>DFGIR_Vec;

  std::vector<ADORA_TENSOR_MAPPER*>tensor_mapper_Vec;

  // Priority: GENERAL_OP_NAME_ENV > --op-name-file > default (lib/DFG/Documents/GeneralOpName.txt)
  std::string GeneralOpNameFile_str =
      (GeneralOpNameFile != nullptr) ? GeneralOpNameFile : opNameFile.getValue();

  /////////////////////////
  /// Map ADORA Tensor
  /////////////////////////
  MapAdoraTensorOp(&context, moduleop, tensor_mapper_Vec, &CEmitter, &PyEmitter, &SDKEmitter,
    subadg, GeneralOpNameFile_str, timeout_ms, max_iters, objOpt, verbose);
  // if(emit_type == "pytest"){
  //   MapAdoraTensorOp(tensor_mapper_Vec)
  // }
  // else{ /// default to be C
  //   CEmitter.preestablishPlacementConstraints(kernel, mapper);
  // }
  
  /////////////////////////
  /// Optimize module to make it suitable for emitting
  /////////////////////////
  /// Before emit C, simplify blockload and blockstore op and affineapply
  SimplifyBlockAccessOp(moduleop);
  ADORA::simplifyConstantAffineApplyOpsInRegion(moduleop.getBodyRegion());
  ADORA::simplifyAddAffineApplyOpsInRegionButOutOfKernel(moduleop.getBodyRegion());

  /// Optionally run schedule-tasks to insert !ADORA.token chain.
  /// Stream coloring is computed inside EmitPytest from SSA token edges
  /// (no assign-streams pass needed).  lower-async-tokens is only for
  /// the LLVM firmware path and must NOT run before Python emit.
  if (enableAsync.getValue()) {
    mlir::PassManager pm(&context);
    auto &fpm = pm.nest<mlir::func::FuncOp>();
    fpm.addPass(mlir::ADORA::createScheduleADORATasksPass());
    if (enableLLMSchedule.getValue()) {
      // TileAssignment (inside llm-pipeline-schedule) needs the CGRA geometry,
      // which lives in the ADG (this module) and is not visible to the dialect
      // pass.  Inject it via the pass's cl::opts before adding the pass.
      extern llvm::cl::opt<int> clNumTiles;
      extern llvm::cl::opt<int> clPePerTile;
      clNumTiles  = numTiles;
      clPePerTile = (numTiles > 0) ? (numGpeNodes / numTiles) : numGpeNodes;
      fpm.addPass(mlir::ADORA::createLLMPipelineSchedulePass());
    }
    if (mlir::failed(pm.run(moduleop))) {
      llvm::errs() << "cgra-mapper: --enable-async pipeline failed.\n";
      return 1;
    }
    if (verbose.getValue())
      llvm::errs() << "cgra-mapper: async pipeline (schedule-tasks"
                   << (enableLLMSchedule.getValue() ? " + llm-pipeline-schedule" : "")
                   << ") applied.\n";
  }

  moduleop.dump();

  //////////////////////////////////////////
  /// Start mapping
  //////////////////////////////////////////
  
  /// Traverse through whole module to get a mapping result
  //// TODO: multithread mapping
  // SmallVector<ADORA::KernelOp> kernels;
  // moduleop.walk([&](ADORA::KernelOp kernel) {
  //   kernels.push_back(kernel);
  // });

  std::atomic<int> kernel_cnt{0};
  std::mutex mlir_mutex;
  std::mutex emitter_mutex;
  std::mutex vector_mutex;
  int max_threads = std::max(1, parallel_cores.getValue());

  auto map_kernel = [&](ADORA::KernelOp kernel) {
    MapperSA* mapper = new MapperSA(subadg, timeout_ms, max_iters, objOpt);
    {
      std::lock_guard<std::mutex> lock(vector_mutex);
      mapper_Vec.push_back(mapper);
    }
    /// Generating DFG
    std::string kernelName;
    {
      std::lock_guard<std::mutex> lock(mlir_mutex);
      kernelName = kernel.getKernelName();
    }
    if(kernelName.empty()){
      kernelName = "kernel_" + std::to_string(kernel_cnt.fetch_add(1));
    }
    mapper->setAgentTraceContext(kernelName);
    if(AgentTrace::enabled()){
      AgentTrace::emit(
        "cgra_mapper",
        "kernel_start",
        "{\"kernel\":\"" + agentTraceJsonEscape(kernelName) + "\"}"
      );
    }
    LLVMCDFG *CDFG = new LLVMCDFG(kernelName, GeneralOpNameFile_str);
    {
      std::lock_guard<std::mutex> lock(mlir_mutex);
      generateCDFGfromKernel(CDFG, kernel, /*verbose=*/verbose);
    }
    // CDFG->CDFGtoDOT(CDFG->name_str()+"_CDFG.dot");

    /// DFG Mapping to CGRA architecture
    DFGIR* dfg_ir = new DFGIR(CDFG);
    {
      std::lock_guard<std::mutex> lock(vector_mutex);
      DFGIR_Vec.push_back(dfg_ir);
    }

    DFG* dfg = dfg_ir->getDFG();
    int numNodes = dfg->nodes().size();
    int numOpNodes = numNodes - dfg->ioNodes().size();
    std::cout << "numOpNodes: " << numOpNodes << ", numDfgNodes(Op+IO): "  << numNodes << std::endl;
    std::cout << "//============== Print DFG =================//" << std::endl;
    dfg->print();
    std::cout << "//============== End Print DFG =================//" << std::endl;
    // dfg->print();
    // map DFG to ADG
    mapper->setDFG(dfg);

    // some io nodes must be placed at some place
    {
      std::scoped_lock lock(mlir_mutex, emitter_mutex);
      if(emit_type == "pytest"){
        PyEmitter.preestablishPlacementConstraints(kernel, mapper);
      }
      else if(emit_type == "sdk"){
        SDKEmitter.preestablishPlacementConstraints(kernel, mapper);
      }
      else{ /// default to be C
        CEmitter.preestablishPlacementConstraints(kernel, mapper);
      }
    }

    // ---- Stage 2: tile-based placement constraints from adora.tile_set ----
    // The dialect pass `llm-pipeline-schedule` (TileAssignment) writes an
    // `adora.tile_set` DenseI64ArrayAttr on each KernelOp deciding WHICH tile(s)
    // the kernel should occupy.  Independent kernels can be steered onto
    // different tiles so they overlap.  Here we translate that decision into
    // per-DFG-node placement constraints: every compute (non-IO) DFG node that
    // is not already constrained (IO nodes already carry SPAD-bank constraints
    // from the emitter above) is restricted to the FU nodes whose ADG `tile()`
    // is in the allowed set.  The SA mapper's findCandidates() honours these
    // constraints, while routing (GIB selection) stays free inside the tiles.
    if (subadg->isMultipleTile()) {
      std::lock_guard<std::mutex> lock(mlir_mutex);
      if (auto tileSetAttr =
              kernel->getAttrOfType<mlir::DenseI64ArrayAttr>("adora.tile_set")) {
        std::set<int> allowedTiles;
        for (int64_t t : tileSetAttr.asArrayRef())
          allowedTiles.insert(static_cast<int>(t));

        if (!allowedTiles.empty()) {
          int constrained = 0;
          for (auto &elem : dfg->nodes()) {
            int nid = elem.first;
            DFGNode *dfgNode = elem.second;
            // IO nodes already constrained to accessible IOBs by the emitter.
            if (dfg->isIONode(nid)) continue;
            // Don't override any pre-existing (emitter) constraint.
            if (!mapper->getPlacementConstraints(dfgNode).empty()) continue;

            std::vector<ADGNode*> tileCandidates;
            for (auto &adgElem : subadg->nodes()) {
              ADGNode *adgNode = adgElem.second;
              if (adgNode->type() == "GIB") continue;  // routing, not a FU
              if (allowedTiles.count(adgNode->tile()) == 0) continue;
              FUNode *fuNode = dynamic_cast<FUNode*>(adgNode);
              if (fuNode && fuNode->opCapable(dfgNode->operation()))
                tileCandidates.push_back(adgNode);
            }
            // Only apply if we found at least one legal FU on the allowed
            // tiles; otherwise leave unconstrained (mapper searches all tiles)
            // so a bad/aggressive tile decision never makes mapping infeasible.
            if (!tileCandidates.empty()) {
              mapper->preestablishPlacementConstraints(dfgNode, tileCandidates);
              ++constrained;
            }
          }
          if (AgentTrace::enabled()) {
            std::string tileList = "[";
            bool first = true;
            for (int t : allowedTiles) {
              if (!first) tileList += ",";
              tileList += std::to_string(t);
              first = false;
            }
            tileList += "]";
            AgentTrace::emit(
              "cgra_mapper",
              "tile_constraints_applied",
              "{\"kernel\":\"" + agentTraceJsonEscape(kernelName) +
              "\",\"tile_set\":" + tileList +
              ",\"constrained_nodes\":" + std::to_string(constrained) + "}"
            );
          }
        }
      }
    }

    std::filesystem::create_directory(kernelName + "_map_result");
    CDFG->CDFGtoDOT(kernelName + "_map_result/before_map_" + CDFG->name_str() + "_CDFG.dot");
    bool succeed = mapper->execute(/*dumpCallFunc=*/false, /*dumpMappedViz*/true, /*resultDir=*/kernelName + "_map_result");
    if(AgentTrace::enabled()){
      AgentTrace::emit(
        "cgra_mapper",
        "kernel_end",
        "{\"kernel\":\"" + agentTraceJsonEscape(kernelName) + "\",\"succeed\":" + std::string(succeed ? "true" : "false") + "}"
      );
    }
    // std::filesystem::create_directory("map_result");
    // CDFG->CDFGtoDOT("map_result/before_map_" + CDFG->name_str() + "_CDFG.dot");
    // bool succeed = mapper->execute(/*dumpCallFunc=*/false, /*dumpMappedViz*/true, /*resultDir=*/"map_result");
    if(succeed){
      // Mapping is successful, get all blockload and blockstore op and corresponding spad memory addresses.
      std::scoped_lock lock(mlir_mutex, emitter_mutex);
      if(emit_type == "pytest"){
        PyEmitter.setMapResult(kernel, mapper);
        PyEmitter.DataBlockOperationsToSPADInfo(kernel, mapper);
        PyEmitter.setTileEnsForKernel(kernel);
        PyEmitter.GenerateCGRAConfig(kernel, mapper);
      }
      else if(emit_type == "sdk"){
        SDKEmitter.setMapResult(kernel, mapper);
        SDKEmitter.DataBlockOperationsToSPADInfo(kernel, mapper);
        SDKEmitter.setTileEnsForKernel(kernel);
        SDKEmitter.GenerateCGRAConfig(kernel, mapper);
      }
      else{ /// default to be C
        CEmitter.setMapResult(kernel, mapper);
        CEmitter.DataBlockOperationsToSPADInfo(kernel, mapper);
        CEmitter.setTileEnsForKernel(kernel);
        CEmitter.GenerateCGRAConfig(kernel, mapper);
      }
    }
  };

  // Phase 0 T1: consume adora.dep_summary / adora.lc_dep_summary produced by
  // ScheduleADORATasksPass on each FuncOp. For now this is observational only
  // — we emit a single AgentTrace event per function that reports the number
  // of regular / loop-carried edges seen. Future work (T3/T4) will feed these
  // into the OnlineRanker/PipelineScheduler decision context instead of the
  // hand-rolled dep graph inside mapping.cpp.
  if(AgentTrace::enabled()){
    moduleop.walk([&](func::FuncOp func) {
      auto depAttr   = func->getAttrOfType<mlir::ArrayAttr>("adora.dep_summary");
      auto lcDepAttr = func->getAttrOfType<mlir::ArrayAttr>("adora.lc_dep_summary");
      if(!depAttr && !lcDepAttr) return WalkResult::advance();
      std::ostringstream payload;
      payload << "{\"func\":\""
              << agentTraceJsonEscape(func.getSymName().str())
              << "\",\"edges\":"
              << (depAttr   ? (int)depAttr.size()   : 0)
              << ",\"lc_edges\":"
              << (lcDepAttr ? (int)lcDepAttr.size() : 0)
              << ",\"source\":\"FuncOp.adora.dep_summary\"}";
      AgentTrace::emit("cgra_mapper", "dep_summary_consumed", payload.str());
      return WalkResult::advance();
    });
  }

  moduleop.walk([&](func::FuncOp func) {
    SmallVector<ADORA::KernelOp> kernels;
    func.walk([&](ADORA::KernelOp kernel) {
      if(kernel->hasAttr("ADORAGemm") ||kernel->hasAttr("ADORAConv"))
        return WalkResult::advance();
      kernels.push_back(kernel);
      return WalkResult::advance();
    });

    if(kernels.empty()){
      return WalkResult::advance();
    }

    size_t num_workers = std::min<size_t>(max_threads, kernels.size());
    std::atomic<size_t> next_index{0};
    std::vector<std::thread> workers;
    workers.reserve(num_workers);
    for(size_t i = 0; i < num_workers; ++i){
      workers.emplace_back([&]() {
        while(true){
          size_t idx = next_index.fetch_add(1);
          if(idx >= kernels.size()){
            break;
          }
          map_kernel(kernels[idx]);
        }
      });
    }
    for(auto& worker : workers){
      worker.join();
    }
    return WalkResult::advance();
  });

  /// Emit module to a C source file
  if(verbose) {moduleop.dump();}

  if(emit_type == "pytest"){
    if(outputFilename == "-")
      PyEmitter.emitPytest(llvm::errs());
    else{
      std::error_code ec;
      llvm::raw_fd_ostream outputFile(outputFilename, ec, sys::fs::FA_Write);
      PyEmitter.emitPytest(outputFile);
    }
  }
  else if(emit_type == "sdk"){
    if(outputFilename == "-")
      SDKEmitter.emitCGRACallFunction(llvm::errs());
    else{
      std::error_code ec;
      llvm::raw_fd_ostream outputFile(outputFilename, ec, sys::fs::FA_Write);
      SDKEmitter.emitCGRACallFunction(outputFile);
    }
  }
  else{ /// default to be C
    if(outputFilename == "-")
      CEmitter.emitCGRACallFunction(llvm::errs());
    else{
      std::error_code ec;
      llvm::raw_fd_ostream outputFile(outputFilename, ec, sys::fs::FA_Write);
      CEmitter.emitCGRACallFunction(outputFile);
    }
  }

  moduleop.dump();

  /// free
  for(auto mapper: mapper_Vec)
    delete mapper;
  for(auto ir: DFGIR_Vec)
    delete ir;
  for(auto mapper: tensor_mapper_Vec)
    delete mapper;

  // delete adg;

  if(AgentTrace::enabled()){
    AgentTrace::emit("cgra_mapper", "end", "{}");
  }

  return 0; 
}
