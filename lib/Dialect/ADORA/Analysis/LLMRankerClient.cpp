//===----------------------------------------------------------------------===//
// LLMRankerClient.cpp — 通用 LLM ranker 调用层（实现）
//
// 从 LLMPipelineSchedule.cpp 抽出的纯进程通信代码：splitArgs + fork + exec +
// 双向 pipe + poll 超时 + scratchpad 解析。不认识任何决策字段，调用方自行解析
// RankerRawResult.response。镜像 mapper/src/mapper/online_ranker.cpp::rankOnce。
//
// 故意内联而不链接 CGRAMapperLIB，保持 MLIRADORAAnalysis 不依赖 mapper 模块。
//===----------------------------------------------------------------------===//
#include "ADORA/Dialect/ADORA/Analysis/LLMRankerClient.h"

#include <sstream>
#include <string>
#include <vector>

#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>

namespace mlir {
namespace ADORA {

namespace {
/// Tokenise an argv string on whitespace (no quoting support needed: the
/// ranker command is a fixed program path + flags).
static std::vector<std::string> splitArgs(llvm::StringRef cmd) {
  std::vector<std::string> args;
  std::istringstream is(cmd.str());
  std::string tok;
  while (is >> tok) args.push_back(tok);
  return args;
}
} // namespace

RankerRawResult callRanker(llvm::StringRef cmd,
                           const std::string &requestJson,
                           int timeoutMs) {
  RankerRawResult resp;

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

  // 通信成功：返回原始响应，调用方自行解析决策字段。
  resp.response = response;
  resp.ok = true;

  // scratchpad 是 LLM 的推理日志（通用元信息），顺手解析出来给所有调用方用。
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

} // namespace ADORA
} // namespace mlir
