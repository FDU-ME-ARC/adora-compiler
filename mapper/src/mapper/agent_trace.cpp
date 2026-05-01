#include "mapper/agent_trace.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <system_error>

std::mutex AgentTrace::_mutex;
bool AgentTrace::_enabled = false;
std::string AgentTrace::_root;
std::string AgentTrace::_runId;
std::string AgentTrace::_tracePath;
long AgentTrace::_seq = 0;

void AgentTrace::configure(const std::string& root, const std::string& runId) {
    std::lock_guard<std::mutex> lock(_mutex);
    _root = root;
    _runId = runId;
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(root) / "data" / "mapper_traces", ec);
    _tracePath = (std::filesystem::path(root) / "data" / "mapper_traces" / (runId + ".events.jsonl")).string();
    _enabled = !ec;
    _seq = 0;
}

bool AgentTrace::enabled() {
    return _enabled;
}

const std::string& AgentTrace::runId() {
    return _runId;
}

void AgentTrace::emit(const std::string& phase, const std::string& event, const std::string& dataJson) {
    if(!_enabled) return;
    auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
    std::lock_guard<std::mutex> lock(_mutex);
    std::ofstream ofs(_tracePath, std::ios::app);
    if(!ofs.good()) return;
    ofs << "{"
        << "\"schema_version\":\"mapper-trace-v0\","
        << "\"run_id\":\"" << agentTraceJsonEscape(_runId) << "\","
        << "\"seq\":" << _seq++ << ","
        << "\"timestamp_ms\":" << now << ","
        << "\"phase\":\"" << agentTraceJsonEscape(phase) << "\","
        << "\"event\":\"" << agentTraceJsonEscape(event) << "\","
        << "\"data\":" << (dataJson.empty() ? "{}" : dataJson)
        << "}\n";
}

std::string agentTraceJsonEscape(const std::string& value) {
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
