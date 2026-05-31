//===- LLMPipelineSchedule.cpp - LLM-guided task pipeline scheduling ------===//
//
// Reads the adora.dep_summary written by ScheduleAdoraTasks and lets an
// external LLM ranker (task_schedule_ranker.py) select a dep_type assignment
// for each task pair.  The selected dep_types are written as "hw_dep_type"
// StringAttr on the relevant DataBlockLoadOp / DataBlockStoreOp / KernelOp
// so that downstream emitters (EmitCGRACall, EmitPytest) pick them up.
//
// Pass options (all optional – defaults give safe no-op behaviour):
//   llm-ranker-cmd     path to task_schedule_ranker.py  (default: disabled)
//   llm-ranker-timeout timeout in ms for one LLM call   (default: 10000)
//   llm-ranker-log     NDJSON log path                  (default: "")
//   adg-path           ADG JSON for hw_model            (default: "")
//   dry-run            use default plan (idx 0) always  (default: false)
//
// When llm-ranker-cmd is empty (or dry-run is true) the pass runs in
// dry-run mode: it still walks the IR and computes the legal candidate set
// but always selects plan index 0 (conservative serial schedule).  This lets
// you test the full pipeline end-to-end without an LLM.
//
//===----------------------------------------------------------------------===//

// Dialect headers MUST precede PassDetail.h (which includes Passes.h.inc).
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "ADORA/Dialect/ADORA/Utility/Utility.h"
#include "ADORA/Dialect/ADORA/Transforms/Passes.h"
#include "PassDetail.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace mlir;

// ---------------------------------------------------------------------------
// Global pass options (command-line flags)
// ---------------------------------------------------------------------------

static llvm::cl::opt<std::string> clRankerCmd(
    "llm-pipeline-schedule-ranker-cmd",
    llvm::cl::desc("Path to task_schedule_ranker.py (empty = dry-run)"),
    llvm::cl::init(""));

static llvm::cl::opt<int> clRankerTimeoutMs(
    "llm-pipeline-schedule-ranker-timeout",
    llvm::cl::desc("LLM ranker call timeout in ms"),
    llvm::cl::init(10000));

static llvm::cl::opt<std::string> clRankerLog(
    "llm-pipeline-schedule-ranker-log",
    llvm::cl::desc("NDJSON log path for ranker decisions"),
    llvm::cl::init(""));

static llvm::cl::opt<bool> clDryRun(
    "llm-pipeline-schedule-dry-run",
    llvm::cl::desc("Always use default plan (idx 0), skip LLM call"),
    llvm::cl::init(false));

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Dep-type string constants that mirror the hardware ISA.
static constexpr llvm::StringLiteral kDepNone      = "LD_DEP_NONE";
static constexpr llvm::StringLiteral kDepExLast    = "LD_DEP_EX_LAST_TASK";
static constexpr llvm::StringLiteral kDepStLast    = "LD_DEP_ST_LAST_TASK";

/// Attribute name written onto load/store/kernel ops.
static constexpr llvm::StringLiteral kHwDepTypeAttr = "hw_dep_type";

// ---------------------------------------------------------------------------
// Task descriptor collected during IR walk
// ---------------------------------------------------------------------------

struct TaskDesc {
  int taskIdx = -1;
  ADORA::DataBlockLoadOp  loadOp;
  ADORA::KernelOp         kernelOp;
  ADORA::DataBlockStoreOp storeOp;

  // Data filled during analysis
  bool rawDepOnPrev  = false;   // true if loadOp reads storeOp of prev task
  bool bankConflict  = false;   // true if active banks overlap (from ADG)

  // Legal dep_type options for the LOAD of this task (vs previous task).
  // [0] is always the conservative default (ST_LAST_TASK).
  std::vector<std::string> legalOptions;
};

// ---------------------------------------------------------------------------
// Simple JSON helpers (no external deps)
// ---------------------------------------------------------------------------

static std::string jsonStr(llvm::StringRef s) {
  return "\"" + s.str() + "\"";
}

static std::string buildRequestJson(
    llvm::StringRef kernelName,
    const std::vector<TaskDesc> &tasks)
{
  std::ostringstream os;
  os << "{\n";
  os << "  \"schema_version\": \"schedule-online-v0\",\n";
  os << "  \"phase\": \"schedule_select\",\n";
  os << "  \"kernel\": " << jsonStr(kernelName) << ",\n";

  // task_graph
  os << "  \"task_graph\": [\n";
  for (size_t i = 0; i < tasks.size(); ++i) {
    const auto &t = tasks[i];
    os << "    {\"task_id\": " << t.taskIdx << "}";
    if (i + 1 < tasks.size()) os << ",";
    os << "\n";
  }
  os << "  ],\n";

  // execution_plan_candidates (Gamma)
  // For now we enumerate two representative plans:
  //   idx 0: all ST_LAST_TASK (conservative default = a_0)
  //   idx 1: all NONE for pairs where legal (aggressive)
  os << "  \"execution_plan_candidates\": [\n";
  os << "    {\"idx\": 0, \"label\": \"serial_default\"},\n";
  os << "    {\"idx\": 1, \"label\": \"max_overlap\"}\n";
  os << "  ],\n";
  os << "  \"default_idx\": 0\n";
  os << "}\n";
  return os.str();
}

// ---------------------------------------------------------------------------
// External ranker call (fork+exec single-shot, like OnlineRanker::rankOnce)
// ---------------------------------------------------------------------------

struct RankerResponse {
  bool   ok           = false;
  int    selectedIdx  = 0;
  std::string scratchpad;
  std::string error;
};

static RankerResponse callRanker(
    llvm::StringRef cmd,
    const std::string &requestJson,
    int timeoutMs)
{
  RankerResponse resp;
  // Write request to a temp file and read response from ranker stdout.
  // Simple implementation: pipe via popen (blocking, single-shot).
  (void)timeoutMs;  // TODO: add alarm-based timeout

  std::string fullCmd = cmd.str() + " 2>/dev/null";
  FILE *fp = popen(fullCmd.c_str(), "r+");  // may not be available on all platforms
  if (!fp) {
    resp.error = "popen failed for: " + cmd.str();
    return resp;
  }
  // Write request to stdin of subprocess
  fputs(requestJson.c_str(), fp);
  fflush(fp);

  // Read response
  std::string response;
  char buf[4096];
  while (fgets(buf, sizeof(buf), fp))
    response += buf;
  pclose(fp);

  // Parse {"selected_idx": N, "scratchpad": "..."}
  // Minimal parse: find "selected_idx": <N>
  auto pos = response.find("\"selected_idx\"");
  if (pos == std::string::npos) {
    resp.error = "no selected_idx in response: " + response;
    return resp;
  }
  auto colon = response.find(':', pos);
  if (colon == std::string::npos) { resp.error = "parse error"; return resp; }
  int idx = std::stoi(response.substr(colon + 1));
  resp.ok = true;
  resp.selectedIdx = idx;

  auto sp = response.find("\"scratchpad\"");
  if (sp != std::string::npos) {
    auto q1 = response.find('"', response.find(':', sp) + 1);
    auto q2 = response.find('"', q1 + 1);
    if (q1 != std::string::npos && q2 != std::string::npos)
      resp.scratchpad = response.substr(q1 + 1, q2 - q1 - 1);
  }
  return resp;
}

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------

struct LLMPipelineSchedulePass
    : public LLMPipelineScheduleBase<LLMPipelineSchedulePass> {

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    mlir::Builder builder(func.getContext());

    // ---- Step 1: collect tasks in program order ----
    std::vector<TaskDesc> tasks;
    {
      int idx = 0;
      // Walk block-by-block; each block may contain multiple task triples.
      func.walk([&](Block *block) {
        // Collect ops in order within this block.
        ADORA::DataBlockLoadOp  pendingLoad;
        ADORA::KernelOp         pendingKernel;

        for (auto &op : *block) {
          if (auto load = dyn_cast<ADORA::DataBlockLoadOp>(&op)) {
            pendingLoad = load;
          } else if (auto kern = dyn_cast<ADORA::KernelOp>(&op)) {
            pendingKernel = kern;
          } else if (auto store = dyn_cast<ADORA::DataBlockStoreOp>(&op)) {
            if (pendingLoad || pendingKernel) {
              TaskDesc td;
              td.taskIdx  = idx++;
              td.loadOp   = pendingLoad;
              td.kernelOp = pendingKernel;
              td.storeOp  = store;
              tasks.push_back(td);
              pendingLoad   = {};
              pendingKernel = {};
            }
          }
        }
      });
    }

    if (tasks.size() < 2) return;  // nothing to pipeline

    // ---- Step 2: compute RAW deps between consecutive pairs ----
    //  (simplified: if T_{n+1} has any async token dep on T_n's storeOp)
    for (size_t i = 1; i < tasks.size(); ++i) {
      if (!tasks[i].loadOp || !tasks[i-1].storeOp) continue;
      for (auto dep : ADORA::getAsyncDeps(tasks[i].loadOp)) {
        if (dep.getDefiningOp() == tasks[i-1].storeOp.getOperation()) {
          tasks[i].rawDepOnPrev = true;
          break;
        }
      }
    }

    // ---- Step 3: build legal options per task (Gamma generation) ----
    for (size_t i = 1; i < tasks.size(); ++i) {
      if (tasks[i].rawDepOnPrev) {
        tasks[i].legalOptions = {kDepStLast.str()};
      } else if (tasks[i].bankConflict) {
        tasks[i].legalOptions = {kDepStLast.str(), kDepExLast.str()};
      } else {
        tasks[i].legalOptions = {kDepStLast.str(), kDepExLast.str(),
                                  kDepNone.str()};
      }
    }

    // ---- Step 4: query LLM (or dry-run) ----
    // Build assignment: task index → chosen dep_type string.
    std::vector<std::string> chosen(tasks.size(), kDepStLast.str());

    bool useLLM = !clDryRun && !clRankerCmd.getValue().empty();
    if (useLLM) {
      std::string kernelName = func.getName().str();
      std::string req = buildRequestJson(kernelName, tasks);
      auto resp = callRanker(clRankerCmd.getValue(), req,
                             clRankerTimeoutMs.getValue());
      if (!resp.ok) {
        func.emitWarning("LLMPipelineSchedulePass: ranker failed ("
                         + resp.error + "), using default plan");
      } else {
        int sel = resp.selectedIdx;
        // Plan idx 0 = all ST_LAST (already default).
        // Plan idx 1 = all NONE where legal.
        if (sel == 1) {
          for (size_t i = 1; i < tasks.size(); ++i) {
            const auto &opts = tasks[i].legalOptions;
            if (std::find(opts.begin(), opts.end(), kDepNone.str())
                != opts.end())
              chosen[i] = kDepNone.str();
            else
              chosen[i] = opts[0];  // fallback to first legal option
          }
        }
        // Log scratchpad
        if (!clRankerLog.getValue().empty() && !resp.scratchpad.empty()) {
          std::ofstream log(clRankerLog.getValue(), std::ios::app);
          log << "{\"kernel\":\"" << func.getName().str()
              << "\",\"selected_idx\":" << sel
              << ",\"scratchpad\":\"" << resp.scratchpad << "\"}\n";
        }
      }
    }

    // ---- Step 5: write hw_dep_type attrs ----
    for (size_t i = 1; i < tasks.size(); ++i) {
      llvm::StringRef depType = chosen[i];
      auto attr = builder.getStringAttr(depType);

      // Set on LoadOp (consumed by EmitCGRACall::computeDepFlag)
      if (tasks[i].loadOp)
        tasks[i].loadOp->setAttr(kHwDepTypeAttr, attr);

      // Set on StoreOp as well (consumed by EmitPytest::getDepsTaskNames)
      if (tasks[i].storeOp)
        tasks[i].storeOp->setAttr(kHwDepTypeAttr, attr);
    }
  }
};

}  // namespace

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::ADORA::createLLMPipelineSchedulePass() {
  return std::make_unique<LLMPipelineSchedulePass>();
}
