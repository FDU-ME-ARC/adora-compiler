#include "mapper/online_ranker.h"
#include "adg/adg.h"
#include "adg/adg_node.h"

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <map>
#include <poll.h>
#include <set>
#include <signal.h>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

std::mutex OnlineRanker::_mutex;
bool OnlineRanker::_enabled = false;
std::string OnlineRanker::_cmd;
int OnlineRanker::_timeoutMs = 15000;
int OnlineRanker::_budget = -1;  // -1 == unlimited
int OnlineRanker::_budgetUsed = 0;
std::string OnlineRanker::_logFile;
OnlineRanker::Mode OnlineRanker::_mode = OnlineRanker::Mode::Once;
OnlineRanker::AdgContext OnlineRanker::_adgContext;
pid_t OnlineRanker::_daemonPid = -1;
int OnlineRanker::_daemonStdin = -1;
int OnlineRanker::_daemonStdout = -1;

namespace {

std::vector<std::string> shellSplit(const std::string& cmd) {
    // Minimal POSIX-style whitespace splitter with double-quote support. Sufficient
    // for the typical ranker command: `python3 -m agent_runtime.online_ranker ...`.
    std::vector<std::string> tokens;
    std::string current;
    bool inQuotes = false;
    for(char c : cmd) {
        if(c == '"') {
            inQuotes = !inQuotes;
            continue;
        }
        if(!inQuotes && std::isspace(static_cast<unsigned char>(c))) {
            if(!current.empty()) {
                tokens.push_back(current);
                current.clear();
            }
            continue;
        }
        current.push_back(c);
    }
    if(!current.empty()) {
        tokens.push_back(current);
    }
    return tokens;
}

void appendLog(const std::string& path, const std::string& record) {
    if(path.empty()) return;
    std::ofstream ofs(path, std::ios::app);
    if(!ofs.good()) return;
    ofs << record << "\n";
}

bool parseSelectedIndex(const std::string& body, int& outIndex, std::string& outRationale, bool& outFallback, std::string& outError) {
    // Tiny purpose-built JSON peek. We expect a flat object containing an
    // integer `selected_index`, plus optional `rationale`, `used_fallback`,
    // `client_error` strings/booleans. Parsing nested objects is unnecessary.
    auto findKey = [&](const std::string& key) -> std::string::size_type {
        std::string needle = "\"" + key + "\"";
        return body.find(needle);
    };

    auto valueStart = [&](std::string::size_type keyPos) -> std::string::size_type {
        if(keyPos == std::string::npos) return std::string::npos;
        auto colon = body.find(':', keyPos);
        if(colon == std::string::npos) return std::string::npos;
        auto pos = colon + 1;
        while(pos < body.size() && std::isspace(static_cast<unsigned char>(body[pos]))) ++pos;
        return pos;
    };

    auto siPos = valueStart(findKey("selected_index"));
    if(siPos == std::string::npos) return false;
    char* end = nullptr;
    long parsed = std::strtol(body.c_str() + siPos, &end, 10);
    if(end == body.c_str() + siPos) return false;
    outIndex = static_cast<int>(parsed);

    auto ratPos = valueStart(findKey("rationale"));
    if(ratPos != std::string::npos && body[ratPos] == '"') {
        auto endQuote = body.find('"', ratPos + 1);
        while(endQuote != std::string::npos && body[endQuote - 1] == '\\') {
            endQuote = body.find('"', endQuote + 1);
        }
        if(endQuote != std::string::npos) {
            outRationale = body.substr(ratPos + 1, endQuote - ratPos - 1);
        }
    }

    auto fbPos = valueStart(findKey("used_fallback"));
    if(fbPos != std::string::npos) {
        outFallback = (body.compare(fbPos, 4, "true") == 0);
    }

    auto errPos = valueStart(findKey("client_error"));
    if(errPos != std::string::npos && body[errPos] == '"') {
        auto endQuote = body.find('"', errPos + 1);
        if(endQuote != std::string::npos) {
            outError = body.substr(errPos + 1, endQuote - errPos - 1);
        }
    }
    return true;
}

}  // namespace

void OnlineRanker::configure(const std::string& cmd, int timeoutMs, int budget, const std::string& logFile, Mode mode) {
    std::lock_guard<std::mutex> lock(_mutex);
    if(_daemonPid > 0){
        killDaemonLocked();
    }
    _cmd = cmd;
    _timeoutMs = timeoutMs > 0 ? timeoutMs : 15000;
    _budget = budget;
    _budgetUsed = 0;
    _logFile = logFile;
    _mode = mode;
    _enabled = !cmd.empty();
    static bool atexitInstalled = false;
    if(!atexitInstalled){
        std::atexit([](){ OnlineRanker::shutdown(); });
        atexitInstalled = true;
    }
}

void OnlineRanker::setAdgContext(const AdgContext& ctx) {
    std::lock_guard<std::mutex> lock(_mutex);
    _adgContext = ctx;
}

OnlineRanker::AdgContext OnlineRanker::adgContext() {
    std::lock_guard<std::mutex> lock(_mutex);
    return _adgContext;
}

void OnlineRanker::shutdown() {
    std::lock_guard<std::mutex> lock(_mutex);
    if(_daemonPid > 0){
        killDaemonLocked();
    }
}

void OnlineRanker::killDaemonLocked() {
    if(_daemonStdin >= 0){
        ::close(_daemonStdin);
        _daemonStdin = -1;
    }
    if(_daemonStdout >= 0){
        ::close(_daemonStdout);
        _daemonStdout = -1;
    }
    if(_daemonPid > 0){
        ::kill(_daemonPid, SIGTERM);
        int status = 0;
        for(int i = 0; i < 10; ++i){
            pid_t r = ::waitpid(_daemonPid, &status, WNOHANG);
            if(r == _daemonPid || r < 0) break;
            ::usleep(20000);
        }
        ::waitpid(_daemonPid, &status, WNOHANG);
        _daemonPid = -1;
    }
}

bool OnlineRanker::enabled() {
    std::lock_guard<std::mutex> lock(_mutex);
    if(!_enabled) return false;
    if(_budget < 0) return true;
    return _budgetUsed < _budget;
}

int OnlineRanker::budget() {
    std::lock_guard<std::mutex> lock(_mutex);
    return _budget;
}

int OnlineRanker::budgetUsed() {
    std::lock_guard<std::mutex> lock(_mutex);
    return _budgetUsed;
}

OnlineRanker::Decision OnlineRanker::rank(const std::string& requestJson, int candidate_count) {
    Decision decision;
    Mode mode;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if(!_enabled) {
            return decision;
        }
        if(_budget >= 0 && _budgetUsed >= _budget) {
            return decision;
        }
        _budgetUsed++;
        mode = _mode;
    }

    decision.used = true;
    if(candidate_count <= 0) {
        decision.error = "no_candidates";
        return decision;
    }

    if(mode == Mode::Daemon) {
        return rankDaemon(requestJson, candidate_count);
    }
    return rankOnce(requestJson, candidate_count);
}

OnlineRanker::Decision OnlineRanker::rankOnce(const std::string& requestJson, int candidate_count) {
    Decision decision;
    decision.used = true;

    std::string cmd;
    int timeoutMs;
    std::string logFile;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        cmd = _cmd;
        timeoutMs = _timeoutMs;
        logFile = _logFile;
    }

    auto argv_strings = shellSplit(cmd);
    if(argv_strings.empty()) {
        decision.error = "empty_ranker_cmd";
        return decision;
    }
    std::vector<char*> argv;
    argv.reserve(argv_strings.size() + 1);
    for(auto& s : argv_strings) {
        argv.push_back(const_cast<char*>(s.c_str()));
    }
    argv.push_back(nullptr);

    int inPipe[2];
    int outPipe[2];
    if(pipe(inPipe) == -1 || pipe(outPipe) == -1) {
        decision.error = std::string("pipe_failed: ") + std::strerror(errno);
        return decision;
    }

    pid_t pid = fork();
    if(pid < 0) {
        ::close(inPipe[0]); ::close(inPipe[1]);
        ::close(outPipe[0]); ::close(outPipe[1]);
        decision.error = std::string("fork_failed: ") + std::strerror(errno);
        return decision;
    }
    if(pid == 0) {
        ::dup2(inPipe[0], STDIN_FILENO);
        ::dup2(outPipe[1], STDOUT_FILENO);
        ::close(inPipe[0]); ::close(inPipe[1]);
        ::close(outPipe[0]); ::close(outPipe[1]);
        ::execvp(argv[0], argv.data());
        std::fprintf(stderr, "online_ranker: execvp failed: %s\n", std::strerror(errno));
        _exit(127);
    }

    ::close(inPipe[0]);
    ::close(outPipe[1]);

    const char* buf = requestJson.c_str();
    size_t remaining = requestJson.size();
    while(remaining > 0) {
        ssize_t n = ::write(inPipe[1], buf, remaining);
        if(n < 0) {
            if(errno == EINTR) continue;
            break;
        }
        buf += n; remaining -= static_cast<size_t>(n);
    }
    ::close(inPipe[1]);

    std::string body;
    auto start = std::chrono::steady_clock::now();
    bool timedOut = false;
    while(true) {
        struct pollfd pfd { outPipe[0], POLLIN, 0 };
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();
        int remaining_ms = timeoutMs - static_cast<int>(elapsed);
        if(remaining_ms <= 0) { timedOut = true; break; }
        int pr = ::poll(&pfd, 1, remaining_ms);
        if(pr == 0) { timedOut = true; break; }
        if(pr < 0) {
            if(errno == EINTR) continue;
            decision.error = std::string("poll_failed: ") + std::strerror(errno);
            break;
        }
        char chunk[4096];
        ssize_t n = ::read(outPipe[0], chunk, sizeof(chunk));
        if(n == 0) break;
        if(n < 0) {
            if(errno == EINTR) continue;
            decision.error = std::string("read_failed: ") + std::strerror(errno);
            break;
        }
        body.append(chunk, static_cast<size_t>(n));
        if(body.size() > 1 << 20) break;
    }
    ::close(outPipe[0]);

    if(timedOut) {
        ::kill(pid, SIGKILL);
        decision.error = "timeout";
    }
    int status = 0;
    ::waitpid(pid, &status, 0);

    if(!body.empty()) {
        int idx = 0;
        std::string rationale;
        std::string clientError;
        bool fallback = false;
        if(parseSelectedIndex(body, idx, rationale, fallback, clientError)) {
            if(idx < 0 || idx >= candidate_count) {
                decision.error = "selected_index_out_of_range";
            } else {
                decision.succeeded = true;
                decision.selected_index = idx;
                decision.used_fallback = fallback;
                decision.rationale = rationale;
                if(decision.error.empty() && !clientError.empty()) {
                    decision.error = clientError;
                }
            }
        } else if(decision.error.empty()) {
            decision.error = "parse_failed";
        }
    } else if(decision.error.empty()) {
        decision.error = "empty_response";
    }

    if(!logFile.empty()) {
        std::ostringstream rec;
        rec << "{"
            << "\"request\":" << requestJson << ","
            << "\"response_raw\":\"" << onlineRankerJsonEscape(body) << "\","
            << "\"succeeded\":" << (decision.succeeded ? "true" : "false") << ","
            << "\"selected_index\":" << decision.selected_index << ","
            << "\"used_fallback\":" << (decision.used_fallback ? "true" : "false") << ","
            << "\"error\":\"" << onlineRankerJsonEscape(decision.error) << "\""
            << "}";
        appendLog(logFile, rec.str());
    }

    return decision;
}

bool OnlineRanker::ensureDaemon() {
    // Caller must hold _mutex.
    if(_daemonPid > 0 && _daemonStdin >= 0 && _daemonStdout >= 0){
        return true;
    }
    if(_daemonPid > 0){
        killDaemonLocked();
    }
    auto argv_strings = shellSplit(_cmd);
    if(argv_strings.empty()){
        return false;
    }
    std::vector<char*> argv;
    argv.reserve(argv_strings.size() + 1);
    for(auto& s : argv_strings){
        argv.push_back(const_cast<char*>(s.c_str()));
    }
    argv.push_back(nullptr);

    int inPipe[2];
    int outPipe[2];
    if(pipe(inPipe) == -1 || pipe(outPipe) == -1){
        return false;
    }
    pid_t pid = fork();
    if(pid < 0){
        ::close(inPipe[0]); ::close(inPipe[1]);
        ::close(outPipe[0]); ::close(outPipe[1]);
        return false;
    }
    if(pid == 0){
        ::dup2(inPipe[0], STDIN_FILENO);
        ::dup2(outPipe[1], STDOUT_FILENO);
        ::close(inPipe[0]); ::close(inPipe[1]);
        ::close(outPipe[0]); ::close(outPipe[1]);
        ::execvp(argv[0], argv.data());
        std::fprintf(stderr, "online_ranker(daemon): execvp failed: %s\n", std::strerror(errno));
        _exit(127);
    }
    ::close(inPipe[0]);
    ::close(outPipe[1]);
    _daemonPid = pid;
    _daemonStdin = inPipe[1];
    _daemonStdout = outPipe[0];
    return true;
}

OnlineRanker::Decision OnlineRanker::rankDaemon(const std::string& requestJson, int candidate_count) {
    Decision decision;
    decision.used = true;

    int timeoutMs;
    std::string logFile;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        timeoutMs = _timeoutMs;
        logFile = _logFile;
        if(!ensureDaemon()){
            decision.error = "daemon_spawn_failed";
            return decision;
        }
    }

    // Send one NDJSON line.
    std::string line = requestJson;
    // Strip embedded newlines for safety; the schema is single-line JSON anyway.
    for(auto& c : line){ if(c == '\n' || c == '\r') c = ' '; }
    line.push_back('\n');

    {
        std::lock_guard<std::mutex> lock(_mutex);
        const char* buf = line.c_str();
        size_t remaining = line.size();
        while(remaining > 0){
            ssize_t n = ::write(_daemonStdin, buf, remaining);
            if(n < 0){
                if(errno == EINTR) continue;
                decision.error = std::string("daemon_write_failed: ") + std::strerror(errno);
                killDaemonLocked();
                return decision;
            }
            buf += n;
            remaining -= static_cast<size_t>(n);
        }
    }

    // Read one NDJSON line back with timeout.
    std::string body;
    auto start = std::chrono::steady_clock::now();
    bool timedOut = false;
    bool gotEol = false;
    while(!gotEol){
        struct pollfd pfd { _daemonStdout, POLLIN, 0 };
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();
        int remaining_ms = timeoutMs - static_cast<int>(elapsed);
        if(remaining_ms <= 0){ timedOut = true; break; }
        int pr = ::poll(&pfd, 1, remaining_ms);
        if(pr == 0){ timedOut = true; break; }
        if(pr < 0){
            if(errno == EINTR) continue;
            decision.error = std::string("daemon_poll_failed: ") + std::strerror(errno);
            break;
        }
        char chunk[4096];
        ssize_t n = ::read(_daemonStdout, chunk, sizeof(chunk));
        if(n == 0){
            decision.error = "daemon_eof";
            std::lock_guard<std::mutex> lock(_mutex);
            killDaemonLocked();
            break;
        }
        if(n < 0){
            if(errno == EINTR) continue;
            decision.error = std::string("daemon_read_failed: ") + std::strerror(errno);
            break;
        }
        body.append(chunk, static_cast<size_t>(n));
        if(body.find('\n') != std::string::npos){ gotEol = true; }
        if(body.size() > 1 << 20) break;
    }

    if(timedOut){
        decision.error = "timeout";
        std::lock_guard<std::mutex> lock(_mutex);
        killDaemonLocked();
    }

    if(!body.empty()){
        // Trim to first line.
        auto nl = body.find('\n');
        std::string firstLine = (nl != std::string::npos) ? body.substr(0, nl) : body;
        int idx = 0;
        std::string rationale;
        std::string clientError;
        bool fallback = false;
        if(parseSelectedIndex(firstLine, idx, rationale, fallback, clientError)){
            if(idx < 0 || idx >= candidate_count){
                decision.error = "selected_index_out_of_range";
            } else {
                decision.succeeded = true;
                decision.selected_index = idx;
                decision.used_fallback = fallback;
                decision.rationale = rationale;
                if(decision.error.empty() && !clientError.empty()){
                    decision.error = clientError;
                }
            }
        } else if(decision.error.empty()){
            decision.error = "parse_failed";
        }
    } else if(decision.error.empty()){
        decision.error = "empty_response";
    }

    if(!logFile.empty()){
        std::ostringstream rec;
        rec << "{"
            << "\"mode\":\"daemon\","
            << "\"request\":" << requestJson << ","
            << "\"response_raw\":\"" << onlineRankerJsonEscape(body) << "\","
            << "\"succeeded\":" << (decision.succeeded ? "true" : "false") << ","
            << "\"selected_index\":" << decision.selected_index << ","
            << "\"used_fallback\":" << (decision.used_fallback ? "true" : "false") << ","
            << "\"error\":\"" << onlineRankerJsonEscape(decision.error) << "\""
            << "}";
        appendLog(logFile, rec.str());
    }
    return decision;
}

std::string onlineRankerJsonEscape(const std::string& value) {
    std::ostringstream os;
    for(char c : value) {
        switch(c) {
            case '\\': os << "\\\\"; break;
            case '"': os << "\\\""; break;
            case '\n': os << "\\n"; break;
            case '\r': os << "\\r"; break;
            case '\t': os << "\\t"; break;
            default: os << c; break;
        }
    }
    return os.str();
}

namespace {

// Cheap content hash so the ranker can key its ADG cache. Not cryptographic;
// FNV-1a 64-bit is plenty for "did the architecture change?" detection.
std::string fnv1a64Hex(const std::string& bytes) {
    uint64_t hash = 1469598103934665603ULL;
    for(unsigned char c : bytes) {
        hash ^= static_cast<uint64_t>(c);
        hash *= 1099511628211ULL;
    }
    std::ostringstream os;
    os << "fnv1a64:" << std::hex << std::setw(16) << std::setfill('0') << hash;
    return os.str();
}

}  // namespace

// Compose a tiny **structural** description of the CGRA architecture, NOT a
// per-node dump. Goal: a few hundred bytes that let the LLM reason in terms of
// "I'm on a 4x4 mesh with PEs supporting these ops, IOBs along the left edge,
// each IOB has 4096-byte SPAD banks", without paying 10 KB+ per request to
// relist every PE.
//
// Schema: `mapper-arch-v0`
//
// {
//   "schema_version": "mapper-arch-v0",
//   "adg_hash":  "<fnv1a64 hex>",
//   "topology":  "mesh_xy",
//   "tile_num":  1,
//   "mesh":      { "rows": 4, "cols": 4 },
//   "pe_types":  {
//     "GPE": { "count": 32, "common_ops": ["FADD32",...] },
//     "IOB": { "count": 16, "common_ops": ["INPUT","LOAD","OUTPUT","STORE"] }
//   },
//   "iob_layout": "left_edge" | "perimeter" | "scattered",
//   "memory":    {
//     "iob_spad_bank_size": 4096,
//     "iob_ag_nest_levels": 2,
//     "cfg_spad_size": 1024,
//     "cfg_data_width": 32,
//     "cfg_addr_width": 16
//   },
//   "interconnect": {
//     "switchbox": "GIB",
//     "gib_count": 25,
//     "neighbors_per_pe_estimate": 4
//   }
// }
std::string dumpAdgSummary(ADG* adg, const std::string& path) {
    if(!adg) return "";

    // 1. Inspect FU nodes to compute mesh shape + common-op intersection.
    int gpeMinX = INT32_MAX, gpeMaxX = INT32_MIN;
    int gpeMinY = INT32_MAX, gpeMaxY = INT32_MIN;
    int iobMinX = INT32_MAX, iobMaxX = INT32_MIN;
    int iobMinY = INT32_MAX, iobMaxY = INT32_MIN;
    int gpeCount = 0, iobCount = 0, gibCount = 0;
    std::set<std::string> gpeCommonOps, iobCommonOps;
    bool gpeCommonInit = false, iobCommonInit = false;

    auto intersectInPlace = [](std::set<std::string>& acc, const std::set<std::string>& other){
        std::set<std::string> next;
        for(const auto& op : acc){
            if(other.count(op)) next.insert(op);
        }
        acc.swap(next);
    };

    for(auto& kv : adg->nodes()){
        ADGNode* node = kv.second;
        if(!node) continue;
        const std::string t = node->type();
        if(t == "GPE"){
            ++gpeCount;
            gpeMinX = std::min(gpeMinX, node->x()); gpeMaxX = std::max(gpeMaxX, node->x());
            gpeMinY = std::min(gpeMinY, node->y()); gpeMaxY = std::max(gpeMaxY, node->y());
            FUNode* fu = dynamic_cast<FUNode*>(node);
            if(fu){
                if(!gpeCommonInit){ gpeCommonOps = fu->operations(); gpeCommonInit = true; }
                else intersectInPlace(gpeCommonOps, fu->operations());
            }
        } else if(t == "IOB"){
            ++iobCount;
            iobMinX = std::min(iobMinX, node->x()); iobMaxX = std::max(iobMaxX, node->x());
            iobMinY = std::min(iobMinY, node->y()); iobMaxY = std::max(iobMaxY, node->y());
            FUNode* fu = dynamic_cast<FUNode*>(node);
            if(fu){
                if(!iobCommonInit){ iobCommonOps = fu->operations(); iobCommonInit = true; }
                else intersectInPlace(iobCommonOps, fu->operations());
            }
        } else if(t == "GIB"){
            ++gibCount;
        }
    }

    int rows = 0, cols = 0;
    if(gpeCount > 0){
        // GPE coordinates use a (col=x, row=y) convention in adora ADG.
        cols = std::max(0, gpeMaxX - gpeMinX) + 1;
        rows = std::max(0, gpeMaxY - gpeMinY) + 1;
    }

    // 2. Heuristic IOB layout classification.
    std::string iobLayout = "scattered";
    if(iobCount > 0){
        bool leftEdge   = (iobMinX == iobMaxX) && iobMinX <= gpeMinX;
        bool rightEdge  = (iobMinX == iobMaxX) && iobMinX > gpeMaxX;
        bool topEdge    = (iobMinY == iobMaxY) && iobMinY <= gpeMinY;
        bool bottomEdge = (iobMinY == iobMaxY) && iobMinY > gpeMaxY;
        if(leftEdge) iobLayout = "left_edge";
        else if(rightEdge) iobLayout = "right_edge";
        else if(topEdge) iobLayout = "top_edge";
        else if(bottomEdge) iobLayout = "bottom_edge";
        else if(iobMinX == iobMaxX || iobMinY == iobMaxY) iobLayout = "perimeter";
    }

    auto emitOpsArray = [&](std::ostringstream& os, const std::set<std::string>& ops){
        os << "[";
        bool firstOp = true;
        for(const auto& op : ops){
            if(!firstOp) os << ",";
            firstOp = false;
            os << "\"" << onlineRankerJsonEscape(op) << "\"";
        }
        os << "]";
    };

    std::ostringstream body;
    body << "{\n";
    body << "  \"schema_version\": \"mapper-arch-v0\",\n";
    body << "  \"topology\": \"mesh_xy\",\n";
    body << "  \"tile_num\": " << adg->tileNum() << ",\n";
    body << "  \"mesh\": {\"rows\": " << rows << ", \"cols\": " << cols << "},\n";
    body << "  \"pe_types\": {\n";
    body << "    \"GPE\": {\"count\": " << gpeCount << ", \"common_ops\": ";
    emitOpsArray(body, gpeCommonOps);
    body << "},\n";
    body << "    \"IOB\": {\"count\": " << iobCount << ", \"common_ops\": ";
    emitOpsArray(body, iobCommonOps);
    body << "}\n  },\n";
    body << "  \"iob_layout\": \"" << iobLayout << "\",\n";
    body << "  \"memory\": {"
         << "\"iob_spad_bank_size\": " << adg->iobSpadBankSize()
         << ", \"iob_ag_nest_levels\": " << adg->iobAgNestLevels()
         << ", \"cfg_spad_size\": " << adg->cfgSpadSize()
         << ", \"cfg_data_width\": " << adg->cfgDataWidth()
         << ", \"cfg_addr_width\": " << adg->cfgAddrWidth()
         << "},\n";
    body << "  \"interconnect\": {"
         << "\"switchbox\": \"GIB\""
         << ", \"gib_count\": " << gibCount
         << ", \"neighbors_per_pe_estimate\": 4"
         << "}\n";
    body << "}\n";

    std::string text = body.str();
    std::string hash = fnv1a64Hex(text);

    // Inject the hash into the JSON body before writing so the on-disk file is
    // self-describing. We do a tiny string replace on the schema_version line.
    std::string headed = text;
    const std::string marker = "\"schema_version\": \"mapper-arch-v0\",";
    auto pos = headed.find(marker);
    if(pos != std::string::npos){
        std::string injected = marker + std::string("\n  \"adg_hash\": \"") + hash + "\",";
        headed.replace(pos, marker.size(), injected);
    }

    std::ofstream ofs(path);
    if(!ofs.good()) return "";
    ofs << headed;
    ofs.close();
    return hash;
}
