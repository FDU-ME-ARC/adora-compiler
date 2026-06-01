//===- LLMPipelineSchedule.cpp - LLM-guided task pipeline scheduling ------===//
//
// Reads the adora.dep_summary written by ScheduleAdoraTasks and lets an
// external LLM ranker (task_schedule_ranker.py) select a dep_type assignment
// for each task pair.  The selected dep_types are written as "hw_dep_type"
// StringAttr on the relevant DataBlockLoadOp / DataBlockStoreOp / KernelOp
// so that downstream emitters (EmitCGRACall, EmitPytest) pick them up.
//
// Pass options (all optional - defaults give safe no-op behaviour):
//   llm-pipeline-schedule-ranker-cmd      argv for task_schedule_ranker.py
//   llm-pipeline-schedule-ranker-timeout  per-call timeout in ms (default 10000)
//   llm-pipeline-schedule-ranker-log      NDJSON decision log path
//   llm-pipeline-schedule-dry-run         always use plan idx 0, skip LLM call
//
// When ranker-cmd is empty (or dry-run is true) the pass runs in dry-run mode:
// it still walks the IR and computes the legal candidate set but always selects
// plan index 0 (conservative serial schedule).  The ranker subprocess is
// invoked via fork+exec with a bidirectional pipe and a poll-based timeout
// (mirrors mapper/src/mapper/online_ranker.cpp::rankOnce) -- the request JSON
// is written to the child's stdin and the one-line JSON response is read back
// from its stdout.  API key / endpoint / model are passed to the child via the
// inherited environment (OPENAI_API_KEY, PTL_BASE_URL, PTL_MODEL), never baked
// into the binary or the IR.
//
//===----------------------------------------------------------------------===//

// Dialect headers MUST precede PassDetail.h (which includes Passes.h.inc).
#include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "ADORA/Dialect/ADORA/Transforms/Passes.h"
#include "ADORA/Dialect/ADORA/Utility/Utility.h"
#include "../PassDetail.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <cerrno>
#include <csignal>
#include <cstring>
#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>

#define DEBUG_TYPE "llm-pipeline-schedule"

using namespace mlir;
using namespace mlir::ADORA;

// ---------------------------------------------------------------------------
// Global pass options (command-line flags)
// ---------------------------------------------------------------------------

static llvm::cl::opt<std::string> clRankerCmd(
    "llm-pipeline-schedule-ranker-cmd",
    llvm::cl::desc("argv for task_schedule_ranker.py (empty = dry-run)"),
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
static constexpr llvm::StringLiteral kDepNone   = "LD_DEP_NONE";
static constexpr llvm::StringLiteral kDepExLast = "LD_DEP_EX_LAST_TASK";
static constexpr llvm::StringLiteral kDepStLast = "LD_DEP_ST_LAST_TASK";

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
  bool rawDepOnPrev = false;  // true if loadOp reads storeOp of prev task
  bool bankConflict = false;  // true if active banks overlap (from ADG)

  // Legal dep_type options for the LOAD of this task (vs previous task).
  // [0] is always the conservative default (ST_LAST_TASK).
  std::vector<std::string> legalOptions;
};

// ---------------------------------------------------------------------------
// Simple JSON helpers (no external deps)
// ---------------------------------------------------------------------------

static std::string jsonStr(llvm::StringRef s) { return "\"" + s.str() + "\""; }

static std::string buildRequestJson(llvm::StringRef kernelName,
                                    const std::vector<TaskDesc> &tasks) {
  std::ostringstream os;
  os << "{\n";
  os << "  \"schema_version\": \"schedule-online-v0\",\n";
  os << "  \"phase\": \"schedule_select\",\n";
  os << "  \"kernel\": " << jsonStr(kernelName) << ",\n";

  os << "  \"task_graph\": [\n";
  for (size_t i = 0; i < tasks.size(); ++i) {
    const auto &t = tasks[i];
    os << "    {\"task_id\": " << t.taskIdx
       << ", \"raw_dep_on_prev\": " << (t.rawDepOnPrev ? "true" : "false")
       << ", \"legal_options\": [";
    for (size_t j = 0; j < t.legalOptions.size(); ++j) {
      os << jsonStr(t.legalOptions[j]);
      if (j + 1 < t.legalOptions.size()) os << ", ";
    }
    os << "]}";
    if (i + 1 < tasks.size()) os << ",";
    os << "\n";
  }
  os << "  ],\n";

  // execution_plan_candidates (Gamma)
  //   idx 0: all ST_LAST_TASK (conservative default = a_0)
  //   idx 1: all NONE for pairs where legal (aggressive overlap)
  os << "  \"execution_plan_candidates\": [\n";
  os << "    {\"idx\": 0, \"label\": \"serial_default\"},\n";
  os << "    {\"idx\": 1, \"label\": \"max_overlap\"}\n";
  os << "  ],\n";
  os << "  \"default_idx\": 0\n";
  os << "}\n";
  return os.str();
}

// ---------------------------------------------------------------------------
// External ranker call: fork + exec + bidirectional pipe + poll timeout.
// Mirrors mapper/src/mapper/online_ranker.cpp::rankOnce.  We deliberately
// inline this (rather than link CGRAMapperLIB) to keep MLIRADORATaskPipeline
// free of any dependency on the mapper module.
// ---------------------------------------------------------------------------

struct RankerResponse {
  bool        ok          = false;
  int         selectedIdx = 0;
  std::string scratchpad;
  std::string error;
};

/// Tokenise an argv string on whitespace (no quoting support needed: the
/// ranker command is a fixed program path + flags).
static std::vector<std::string> splitArgs(llvm::StringRef cmd) {
  std::vector<std::string> args;
  std::istringstream is(cmd.str());
  std::string tok;
  while (is >> tok) args.push_back(tok);
  return args;
}

static RankerResponse callRanker(llvm::StringRef cmd,
                                 const std::string &requestJson,
                                 int timeoutMs) {
  RankerResponse resp;

  std::vector<std::string> args = splitArgs(cmd);
  if (args.empty()) {
    resp.error = "empty ranker command";
    return resp;
  }

  int inPipe[2];   // parent writes -> child stdin
  int outPipe[2];  // child stdout -> parent reads
  if (pipe(inPipe) != 0 || pipe(outPipe) != 0) {
    resp.error = std::string("pipe() failed: ") + std::strerror(errno);
    return resp;
  }

  pid_t pid = fork();
  if (pid < 0) {
    resp.error = std::string("fork() failed: ") + std::strerror(errno);
    ::close(inPipe[0]); ::close(inPipe[1]);
    ::close(outPipe[0]); ::close(outPipe[1]);
    return resp;
  }

  if (pid == 0) {
    // ---- child ----
    ::dup2(inPipe[0], STDIN_FILENO);
    ::dup2(outPipe[1], STDOUT_FILENO);
    ::close(inPipe[0]); ::close(inPipe[1]);
    ::close(outPipe[0]); ::close(outPipe[1]);

    std::vector<char *> argv;
    argv.reserve(args.size() + 1);
    for (auto &a : args) argv.push_back(const_cast<char *>(a.c_str()));
    argv.push_back(nullptr);

    execvp(argv[0], argv.data());
    // exec failed
    ::fprintf(stderr, "execvp(%s) failed: %s\n", argv[0], std::strerror(errno));
    _exit(127);
  }

  // ---- parent ----
  ::close(inPipe[0]);
  ::close(outPipe[1]);

  // Write the request, then close stdin so the child sees EOF.
  {
    const char *p = requestJson.data();
    size_t remaining = requestJson.size();
    while (remaining > 0) {
      ssize_t n = ::write(inPipe[1], p, remaining);
      if (n < 0) {
        if (errno == EINTR) continue;
        break;  // child may have died; let the read/wait path report it
      }
      p += n;
      remaining -= static_cast<size_t>(n);
    }
  }
  ::close(inPipe[1]);

  // Read stdout with a poll-based timeout.
  std::string response;
  {
    struct pollfd pfd;
    pfd.fd = outPipe[0];
    pfd.events = POLLIN;
    char buf[4096];
    bool timedOut = false;

    while (true) {
      int pr = ::poll(&pfd, 1, timeoutMs);
      if (pr < 0) {
        if (errno == EINTR) continue;
        resp.error = std::string("poll() failed: ") + std::strerror(errno);
        break;
      }
      if (pr == 0) { timedOut = true; break; }
      ssize_t n = ::read(outPipe[0], buf, sizeof(buf));
      if (n < 0) {
        if (errno == EINTR) continue;
        resp.error = std::string("read() failed: ") + std::strerror(errno);
        break;
      }
      if (n == 0) break;  // EOF: child closed stdout
      response.append(buf, static_cast<size_t>(n));
    }
    ::close(outPipe[0]);

    if (timedOut) {
      ::kill(pid, SIGKILL);
      resp.error = "ranker timed out after " + std::to_string(timeoutMs) + "ms";
    }
  }

  int status = 0;
  ::waitpid(pid, &status, 0);

  if (!resp.error.empty()) return resp;
  if (WIFEXITED(status) && WEXITSTATUS(status) == 127) {
    resp.error = "ranker exec failed (exit 127)";
    return resp;
  }

  // Parse {"selected_idx": N, "scratchpad": "..."}
  auto pos = response.find("\"selected_idx\"");
  if (pos == std::string::npos) {
    resp.error = "no selected_idx in response: " + response;
    return resp;
  }
  auto colon = response.find(':', pos);
  if (colon == std::string::npos) {
    resp.error = "parse error (no colon after selected_idx)";
    return resp;
  }
  resp.selectedIdx = std::stoi(response.substr(colon + 1));
  resp.ok = true;

  auto sp = response.find("\"scratchpad\"");
  if (sp != std::string::npos) {
    auto q1 = response.find('"', response.find(':', sp) + 1);
    auto q2 = (q1 == std::string::npos) ? std::string::npos
                                        : response.find('"', q1 + 1);
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
      func.walk([&](Block *block) {
        ADORA::DataBlockLoadOp pendingLoad;
        ADORA::KernelOp        pendingKernel;

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
    for (size_t i = 1; i < tasks.size(); ++i) {
      if (!tasks[i].loadOp || !tasks[i - 1].storeOp) continue;
      for (mlir::Value dep : ADORA::getAsyncDeps(tasks[i].loadOp.getOperation())) {
        if (dep.getDefiningOp() == tasks[i - 1].storeOp.getOperation()) {
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
    std::vector<std::string> chosen(tasks.size(), kDepStLast.str());

    bool useLLM = !clDryRun && !clRankerCmd.getValue().empty();
    if (useLLM) {
      std::string kernelName = func.getName().str();
      std::string req = buildRequestJson(kernelName, tasks);
      auto resp = callRanker(clRankerCmd.getValue(), req,
                             clRankerTimeoutMs.getValue());
      if (!resp.ok) {
        func.emitWarning("LLMPipelineSchedulePass: ranker failed (" +
                         resp.error + "), using default plan");
      } else {
        int sel = resp.selectedIdx;
        // Plan idx 0 = all ST_LAST (already the default).
        // Plan idx 1 = all NONE where legal (aggressive overlap).
        if (sel == 1) {
          for (size_t i = 1; i < tasks.size(); ++i) {
            const auto &opts = tasks[i].legalOptions;
            if (std::find(opts.begin(), opts.end(), kDepNone.str()) !=
                opts.end())
              chosen[i] = kDepNone.str();
            else
              chosen[i] = opts[0];  // fallback to first legal option
          }
        }
        if (!clRankerLog.getValue().empty() && !resp.scratchpad.empty()) {
          std::ofstream log(clRankerLog.getValue(), std::ios::app);
          log << "{\"kernel\":\"" << kernelName
              << "\",\"selected_idx\":" << sel
              << ",\"scratchpad\":\"" << resp.scratchpad << "\"}\n";
        }
      }
    }

    // ---- Step 5: write hw_dep_type attrs ----
    for (size_t i = 1; i < tasks.size(); ++i) {
      auto attr = builder.getStringAttr(chosen[i]);
      if (tasks[i].loadOp)
        tasks[i].loadOp->setAttr(kHwDepTypeAttr, attr);
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
