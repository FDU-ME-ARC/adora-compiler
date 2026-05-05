#ifndef __ONLINE_RANKER_H__
#define __ONLINE_RANKER_H__

#include <mutex>
#include <string>
#include <sys/types.h>
#include <vector>

// Online placement ranker bridge.
//
// Speaks the JSON contract documented in
// `src/agent_runtime/online_schema.py` (schema_version "mapper-online-v0").
//
// Two execution modes:
//
//   - "once":   one ranker subprocess per consultation (fork+exec each call).
//               Simple and isolated, but pays process-startup latency every
//               time. The ranker is launched in --mode once.
//   - "daemon": one long-lived ranker subprocess for the whole mapper run.
//               Requests/responses are NDJSON over pipes. Lets the ranker keep
//               warm caches (e.g. parsed ADG summaries, OpenAI client state).
//               The ranker is launched in --mode daemon.
//
// Behavior is conservative: if disabled, out of budget, the spawned process
// fails, the response is malformed, or the timeout fires, the caller MUST
// fall back to the mapper's original candidate ordering. The class itself
// only guarantees that a returned `selected_index` is in
// `[0, candidate_count)`; mapping legality (route, ports, ...) is still
// validated downstream by `Mapping::mapDfgNode`.
class OnlineRanker {
public:
    struct Decision {
        bool used = false;          // ranker was actually consulted
        bool succeeded = false;     // a valid index was returned
        int selected_index = 0;     // valid only if succeeded
        bool used_fallback = false; // ranker reported fallback path
        std::string rationale;
        std::string error;
    };

    enum class Mode { Once, Daemon };

    // ADG context emitted once at mapper startup. Stored by hash so the
    // request payload can carry a stable {adg_hash, adg_summary_ref}
    // pair and a long-lived ranker can cache the architecture.
    struct AdgContext {
        std::string adg_hash;
        std::string adg_summary_path; // absolute or process-cwd-relative
    };

    static void configure(const std::string& cmd,
                          int timeoutMs,
                          int budget,
                          const std::string& logFile,
                          Mode mode = Mode::Once);

    static void setAdgContext(const AdgContext& ctx);
    static AdgContext adgContext();

    static bool enabled();
    static int budget();
    static int budgetUsed();

    // Build a JSON request for the current placement decision and ask the
    // ranker. `requestJson` is the full request body as a JSON string. The
    // caller must populate `candidate_count` so we can validate the response.
    static Decision rank(const std::string& requestJson, int candidate_count);

    // Tear down the daemon child (if any). Safe to call multiple times. Also
    // invoked implicitly at exit in case the mapper forgets.
    static void shutdown();

private:
    static Decision rankOnce(const std::string& requestJson, int candidate_count);
    static Decision rankDaemon(const std::string& requestJson, int candidate_count);
    static bool ensureDaemon();
    static void killDaemonLocked();

    static std::mutex _mutex;
    static bool _enabled;
    static std::string _cmd;
    static int _timeoutMs;
    static int _budget;
    static int _budgetUsed;
    static std::string _logFile;
    static Mode _mode;
    static AdgContext _adgContext;

    // Daemon-mode persistent state (guarded by _mutex):
    static pid_t _daemonPid;
    static int _daemonStdin;
    static int _daemonStdout;
};

std::string onlineRankerJsonEscape(const std::string& value);

// Dump a compact JSON description of the ADG to `path`. Returns the SHA-1-like
// content hash (a short hex string). The schema is `mapper-adg-v0`. Returns
// empty string on failure.
class ADG;
std::string dumpAdgSummary(ADG* adg, const std::string& path);

#endif
