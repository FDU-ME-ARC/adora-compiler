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
#include <cstdint>
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

#include "op/operations.h"
#include "ir/adg_ir.h"
#include "ir/dfg_ir.h"
#include "mapper/mapper_sa.h"
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

static unsigned deriveKernelSeed(unsigned globalSeed, uint64_t ordinal,
                                 llvm::StringRef kernelName) {
  uint32_t hash = 2166136261u;
  auto mixByte = [&](uint8_t value) {
    hash ^= value;
    hash *= 16777619u;
  };
  for (unsigned shift = 0; shift < 32; shift += 8)
    mixByte(static_cast<uint8_t>(globalSeed >> shift));
  for (unsigned shift = 0; shift < 64; shift += 8)
    mixByte(static_cast<uint8_t>(ordinal >> shift));
  for (char value : kernelName)
    mixByte(static_cast<uint8_t>(value));
  return hash;
}

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

  static cl::opt<unsigned> randomSeed(
    "seed",
    cl::Optional,
    cl::desc("random seed"),
    cl::value_desc("uint"));
    
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

  // Opt-in switch to run adora-schedule-tasks on the module right before
  // per-kernel mapping & emit, so the schedule-derived SSA !ADORA.token
  // dependencies surface in the generated code. Tokens are consumed directly
  // by the emit layer (stream colouring / await-gather in pytest, LD_DEP/EX_DEP
  // flags in C); they are NOT lowered here. Default off keeps cgra-mapper
  // byte-identical to the non-async baseline.
  static cl::opt<bool> enableAsync(
    "enable-async",
    cl::Optional,
    cl::desc("Run --adora-schedule-tasks before emit so SSA !ADORA.token "
             "dependencies drive BlockStore await-gather / dep flags. "
             "Tokens are consumed directly by emit (not lowered). "
             "Default false."),
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

  unsigned seed = randomSeed.getNumOccurrences() > 0
                      ? randomSeed.getValue()
                      : static_cast<unsigned>(time(0));
  std::cout << "Random seed: " << seed << std::endl;



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
  if (failed(MapAdoraTensorOp(&context, moduleop, tensor_mapper_Vec, &CEmitter,
                              &PyEmitter, &SDKEmitter, subadg,
                              GeneralOpNameFile_str, timeout_ms, max_iters,
                              objOpt, verbose))) {
    llvm::errs() << "cgra-mapper: tensor CDFG generation failed.\n";
    return 1;
  }
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
  ///
  /// Skip if the module was already scheduled (carries adora.scheduled):
  /// schedule-tasks is not a no-op on already-async IR, and the existing
  /// SSA tokens already drive emit, so re-running would be redundant.
  if (enableAsync.getValue()) {
    if (moduleop->hasAttr("adora.scheduled")) {
      if (verbose.getValue())
        llvm::errs() << "cgra-mapper: --enable-async: module already scheduled "
                        "(adora.scheduled); skipping schedule-tasks.\n";
    } else {
      mlir::PassManager pm(&context);
      auto &fpm = pm.nest<mlir::func::FuncOp>();
      fpm.addPass(mlir::ADORA::createScheduleADORATasksPass());
      if (mlir::failed(pm.run(moduleop))) {
        llvm::errs() << "cgra-mapper: --enable-async pipeline failed.\n";
        return 1;
      }
      if (verbose.getValue())
        llvm::errs() << "cgra-mapper: async pipeline (schedule-tasks) applied.\n";
    }
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

  std::atomic<bool> generation_failed{false};
  std::atomic<bool> mapping_failed{false};
  std::mutex mlir_mutex;
  std::mutex emitter_mutex;
  std::mutex vector_mutex;
  int max_threads = std::max(1, parallel_cores.getValue());

  auto map_kernel = [&](ADORA::KernelOp kernel, size_t kernelOrdinal) {
    if (generation_failed.load())
      return;
    /// Generating DFG
    std::string kernelName;
    {
      std::lock_guard<std::mutex> lock(mlir_mutex);
      kernelName = kernel.getKernelName();
    }
    if(kernelName.empty()){
      kernelName = "kernel_" + std::to_string(kernelOrdinal);
    }
    MapperSA* mapper = new MapperSA(
        subadg, timeout_ms, max_iters, objOpt,
        deriveKernelSeed(seed, kernelOrdinal, kernelName));
    {
      std::lock_guard<std::mutex> lock(vector_mutex);
      mapper_Vec.push_back(mapper);
    }
    LLVMCDFG *CDFG = new LLVMCDFG(kernelName, GeneralOpNameFile_str);
    LogicalResult generationResult = failure();
    {
      std::lock_guard<std::mutex> lock(mlir_mutex);
      generationResult = generateCDFGfromKernel(CDFG, kernel,
                                                /*verbose=*/verbose);
    }
    if (failed(generationResult)) {
      delete CDFG;
      generation_failed.store(true);
      return;
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

    std::filesystem::create_directory(kernelName + "_map_result");
    CDFG->CDFGtoDOT(kernelName + "_map_result/before_map_" + CDFG->name_str() + "_CDFG.dot");
    bool succeed = mapper->execute(/*dumpCallFunc=*/false, /*dumpMappedViz*/true, /*resultDir=*/kernelName + "_map_result");
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
    else {
      mapping_failed.store(true);
    }
  };

  size_t nextKernelOrdinal = 0;
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
    const size_t ordinalBase = nextKernelOrdinal;
    nextKernelOrdinal += kernels.size();

    size_t num_workers = std::min<size_t>(max_threads, kernels.size());
    std::atomic<size_t> next_index{0};
    std::vector<std::thread> workers;
    workers.reserve(num_workers);
    for(size_t i = 0; i < num_workers; ++i){
      workers.emplace_back([&]() {
        while(true){
          size_t idx = next_index.fetch_add(1);
          if(idx >= kernels.size() || generation_failed.load()){
            break;
          }
          map_kernel(kernels[idx], ordinalBase + idx);
        }
      });
    }
    for(auto& worker : workers){
      worker.join();
    }
    return WalkResult::advance();
  });

  if (generation_failed.load() || mapping_failed.load()) {
    for (auto mapper : mapper_Vec)
      delete mapper;
    for (auto ir : DFGIR_Vec)
      delete ir;
    for (auto mapper : tensor_mapper_Vec)
      delete mapper;
    return 1;
  }

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

  return 0; 
}
