#ifndef __AGENT_TRACE_H__
#define __AGENT_TRACE_H__

#include <mutex>
#include <string>

class AgentTrace {
public:
    static void configure(const std::string& root, const std::string& runId);
    static bool enabled();
    static void emit(const std::string& phase, const std::string& event, const std::string& dataJson = "{}");
    static const std::string& runId();

private:
    static std::mutex _mutex;
    static bool _enabled;
    static std::string _root;
    static std::string _runId;
    static std::string _tracePath;
    static long _seq;
};

std::string agentTraceJsonEscape(const std::string& value);

#endif
