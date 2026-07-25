#include "rtlil/yosys_frontend.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>
#include "nlohmann/json.hpp"

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

std::string sanitizeName(std::string name){
    for(char& c : name){
        if(!std::isalnum(static_cast<unsigned char>(c)) && c != '_') c = '_';
    }
    if(name.empty() || std::isdigit(static_cast<unsigned char>(name.front())))
        name = "dfg_" + name;
    return name;
}

std::string rtlilWire(int nodeId, int port){
    return "$n" + std::to_string(nodeId) + "_" + std::to_string(port);
}

int edgeWidth(DFG* dfg, int edgeId){
    auto* edge = dfg->edge(edgeId);
    return edge ? edge->bitWidth() : dfg->CGWidth();
}

void emitInputConnections(std::ostream& os, DFG* dfg, DFGNode* node){
    for(auto& input : node->inputEdges()){
        DFGEdge* edge = dfg->edge(input.second);
        char port = static_cast<char>('A' + input.first);
        os << "    connect \\" << port << " "
           << rtlilWire(edge->srcId(), edge->srcPortIdx()) << "\n";
    }
    if(node->hasImm()){
        int width = node->bitWidth();
        std::string bits = std::bitset<64>(node->imm()).to_string();
        bits = bits.substr(64 - width);

        os << "    connect \\" << static_cast<char>('A' + node->immIdx())
            << " " << width << "'" << bits << "\n";
    }
    for(auto& immediate : node->fineImmediates()){
        os << "    connect \\" << static_cast<char>('A' + immediate.first)
           << " 1'" << immediate.second << "\n";
    }
}

void emitDfgRtlil(DFG* dfg, const fs::path& filename, const std::string& designName){
    std::ofstream os(filename);
    if(!os) throw std::runtime_error("cannot create RTLIL file: " + filename.string());
    os << "module \\" << sanitizeName(designName) << "\n\n";

    for(auto& nodeItem : dfg->nodes()){
        DFGNode* node = nodeItem.second;
        for(auto& output : node->outputEdges()){
            if(output.second.empty()) continue;
            int width = edgeWidth(dfg, *output.second.begin());
            if(width > 1) os << "  attribute \\keep \"coarse_wire\"\n";
            os << "  wire width " << width << " "
               << rtlilWire(node->id(), output.first) << "\n";
        }
    }

    for(auto& nodeItem : dfg->nodes()){
        DFGNode* node = nodeItem.second;
        std::string op = node->operation();
        if(op == "LUT"){
            std::string truthBits;
            for(char c : node->LUTconfig()){
                if(c == '0' || c == '1') truthBits.push_back(c);
            }
            if(truthBits.empty()) truthBits = "0";
            os << "  cell $lut $n" << node->id() << "\n";
            os << "    parameter \\WIDTH " << node->LUTsize() << "\n";
            os << "    parameter \\LUT " << (uint64_t(1) << node->LUTsize())
               << "'" << truthBits << "\n";
            os << "    connect \\A {";
            for(int operand = node->LUTsize() - 1; operand >= 0; --operand){
                int eid = node->inputEdge(1, operand);
                if(eid >= 0){
                    DFGEdge* edge = dfg->edge(eid);
                    os << " " << rtlilWire(edge->srcId(), edge->srcPortIdx());
                }else{
                    auto immediate = node->fineImmediates().find(operand);
                    os << " 1'" << (immediate == node->fineImmediates().end()
                        ? 0 : immediate->second);
                }
            }
            os << " }\n";
            if(!node->outputEdges(1).empty()){
                os << "    connect \\Y " << rtlilWire(node->id(), 0) << "\n";
            }
            os << "  end\n\n";
            continue;
        }
        bool fineLogic = node->bitWidths().size() == 1 &&
                         node->bitWidths().count(1) &&
                         (op == "AND" || op == "OR" || op == "XOR" ||
                          op == "EQ" || op == "NOT");
        std::string cellType = fineLogic ? "$" : "\\";
        std::string emittedOp = op;
        if(fineLogic){
            std::transform(emittedOp.begin(), emittedOp.end(), emittedOp.begin(),
                           [](unsigned char c){ return std::tolower(c); });
        }else{
            os << "  attribute \\keep \"aufora_node_" << node->id() << "\"\n";
        }
        os << "  cell " << cellType << emittedOp << " $n" << node->id() << "\n";

        std::set<int> parameterizedInputs;
        for(auto& input : node->inputEdges()){
            int width = edgeWidth(dfg, input.second);
            char port = static_cast<char>('A' + input.first);
            if(fineLogic) os << "    parameter \\" << port << "_SIGNED 0\n";
            os << "    parameter \\" << port << "_WIDTH " << width << "\n";
            parameterizedInputs.insert(input.first);
        }
        if(node->hasImm() && !parameterizedInputs.count(node->immIdx())){
            os << "    parameter \\" << static_cast<char>('A' + node->immIdx())
               << "_WIDTH " << node->bitWidth() << "\n";
        }
        for(auto& immediate : node->fineImmediates()){
            if(!parameterizedInputs.count(immediate.first)){
                os << "    parameter \\" << static_cast<char>('A' + immediate.first)
                   << "_WIDTH 1\n";
            }
        }
        for(auto& output : node->outputEdges()){
            if(output.second.empty()) continue;
            int width = edgeWidth(dfg, *output.second.begin());
            os << "    parameter \\" << static_cast<char>('Y' + output.first)
               << "_WIDTH " << width << "\n";
        }
        if(node->operation() == "CONST") os << "    parameter \\VALUE " << node->imm() << "\n";
        emitInputConnections(os, dfg, node);
        for(auto& output : node->outputEdges()){
            os << "    connect \\" << static_cast<char>('Y' + output.first)
               << " " << rtlilWire(node->id(), output.first) << "\n";
        }
        os << "  end\n\n";
    }
    os << "end\n";
}

void emitCellLibrary(DFG* dfg, const fs::path& filename){
    std::ofstream os(filename);
    if(!os) throw std::runtime_error("cannot create Yosys cell library");
    std::set<std::string> emitted;
    for(auto& nodeItem : dfg->nodes()){
        DFGNode* node = nodeItem.second;
        std::string op = node->operation();
        bool fineLogic = node->bitWidths().size() == 1 &&
                         node->bitWidths().count(1) &&
                         (op == "AND" || op == "OR" || op == "XOR" ||
                          op == "EQ" || op == "NOT");
        if(op == "LUT" || fineLogic || !emitted.insert(op).second) continue;

        std::set<int> inputs;
        std::set<int> outputs;
        for(auto& item : node->inputEdges()) inputs.insert(item.first);
        if(node->hasImm()) inputs.insert(node->immIdx());
        for(auto& item : node->fineImmediates()) inputs.insert(item.first);
        for(auto& item : node->outputEdges()) outputs.insert(item.first);

        os << "(* blackbox *) module \\" << op << " #(\n";
        bool firstParam = true;
        for(int port : inputs){
            if(!firstParam) os << ",\n";
            os << "  parameter " << static_cast<char>('A' + port) << "_WIDTH = 32";
            firstParam = false;
        }
        for(int port : outputs){
            if(!firstParam) os << ",\n";
            os << "  parameter " << static_cast<char>('Y' + port) << "_WIDTH = 32";
            firstParam = false;
        }
        if(op == "CONST"){
            if(!firstParam) os << ",\n";
            os << "  parameter VALUE = 0";
            firstParam = false;
        }
        os << "\n) (";
        bool firstPort = true;
        for(int port : inputs){
            if(!firstPort) os << ", ";
            os << static_cast<char>('A' + port);
            firstPort = false;
        }
        for(int port : outputs){
            if(!firstPort) os << ", ";
            os << static_cast<char>('Y' + port);
            firstPort = false;
        }
        os << ");\n";
        for(int port : inputs){
            char name = static_cast<char>('A' + port);
            os << "  input [" << name << "_WIDTH-1:0] " << name << ";\n";
        }
        for(int port : outputs){
            char name = static_cast<char>('Y' + port);
            os << "  output [" << name << "_WIDTH-1:0] " << name << ";\n";
        }
        os << "endmodule\n\n";
    }
}

long long binaryValue(const json& value, long long fallback = 0){
    if(value.is_number_integer()) return value.get<long long>();
    if(!value.is_string()) return fallback;
    std::string text = value.get<std::string>();
    if(text.empty()) return fallback;
    long long result = 0;
    for(char c : text){
        if(c != '0' && c != '1') continue;
        result = (result << 1) | (c - '0');
    }
    return result;
}

std::string normalizeCellType(std::string type){
    if(!type.empty() && (type.front() == '\\' || type.front() == '$')) type.erase(type.begin());
    std::transform(type.begin(), type.end(), type.begin(),
                   [](unsigned char c){ return std::toupper(c); });
    return type;
}

int portIndex(const std::string& name, int fallback){
    if(name.size() == 1 && name[0] >= 'A' && name[0] <= 'X') return name[0] - 'A';
    return fallback;
}

struct Producer {
    int node = -1;
    int port = 0;
    int bit = 0;
};

int originalNodeId(const json& cellJson){
    auto attributes = cellJson.find("attributes");
    if(attributes == cellJson.end()) return -1;
    auto keep = attributes->find("keep");
    if(keep == attributes->end() || !keep->is_string()) return -1;
    const std::string value = keep->get<std::string>();
    const std::string prefix = "aufora_node_";
    if(value.rfind(prefix, 0) != 0) return -1;
    try {
        return std::stoi(value.substr(prefix.size()));
    } catch(const std::exception&) {
        return -1;
    }
}

void inheritNodeMetadata(DFGNode* node, DFGNode* original){
    if(!node || !original) return;
    node->setOpLatency(original->opLatency());
    node->setCommutative(original->commutative());
    node->setAccumulative(original->accumulative());
    if(original->accumulative()){
        node->setIsAccFirst(original->isAccFirst());
        node->setInitVal(original->initVal());
        node->setCycles(original->cycles());
        node->setInterval(original->interval());
        node->setRepeats(original->repeats());
    }
    node->setAdditionalStartDelay(original->additionalStartDelay());

    auto* io = dynamic_cast<DFGIONode*>(node);
    auto* originalIo = dynamic_cast<DFGIONode*>(original);
    if(!io || !originalIo) return;
    io->setMemRefName(originalIo->memRefName());
    io->setMemOffset(originalIo->memOffset());
    io->setReducedMemOffset(originalIo->reducedMemOffset());
    io->setMemSize(originalIo->memSize());
    for(const auto& level : originalIo->pattern())
        io->addPatternLevel(level.first, level.second);
    io->setPingpong(originalIo->isPingpong());
}

DFG* parseYosysJson(
    const fs::path& filename, const std::string& designName, DFG* originalDfg){
    std::ifstream is(filename);
    if(!is) throw std::runtime_error("cannot open Yosys JSON: " + filename.string());
    json root;
    is >> root;
    auto& modules = root.at("modules");
    auto moduleIt = modules.find(sanitizeName(designName));
    if(moduleIt == modules.end()) moduleIt = modules.begin();
    if(moduleIt == modules.end()) throw std::runtime_error("Yosys JSON contains no module");
    auto& cells = moduleIt.value().at("cells");

    DFG* dfg = new DFG();
    dfg->setId(0);
    dfg->setFineGrained(false);
    std::map<std::string, int> cellIds;
    int nextNodeId = 1;
    for(auto& cellItem : cells.items()){
        auto& cellJson = cellItem.value();
        std::string type = cellJson.at("type").get<std::string>();
        std::string op = normalizeCellType(type);
        DFGNode* node;
        if(op == "INPUT" || op == "OUTPUT" || op == "CINPUT" || op == "COUTPUT" ||
           op == "LOAD" || op == "STORE" || op == "CLOAD" || op == "CSTORE"){
            node = new DFGIONode();
            dfg->addIONode(nextNodeId);
        }else{
            node = new DFGNode();
        }
        if(op == "LUT"){
            auto& params = cellJson.at("parameters");
            int lutSize = static_cast<int>(binaryValue(params.at("WIDTH")));
            node->setLUTsize(lutSize);
            node->setLUTconfig(params.at("LUT").get<std::string>());
            dfg->addLUTNode(nextNodeId);
            dfg->setFineGrained(true);
        }
        node->setId(nextNodeId);
        node->setName(cellItem.key());
        node->setOperation(op);
        int originalId = originalNodeId(cellJson);
        inheritNodeMetadata(
            node, originalId >= 0 ? originalDfg->node(originalId) : nullptr);
        dfg->addNode(node);
        cellIds[cellItem.key()] = nextNodeId++;
    }

    std::map<int, Producer> producers;
    for(auto& cellItem : cells.items()){
        int nodeId = cellIds.at(cellItem.key());
        auto& cellJson = cellItem.value();
        int outputPort = 0;
        for(auto& portItem : cellJson.at("connections").items()){
            std::string direction = cellJson.at("port_directions").at(portItem.key()).get<std::string>();
            if(direction != "output") continue;
            int bitIndex = 0;
            for(auto& bit : portItem.value()){
                if(bit.is_number_integer()){
                    producers[bit.get<int>()] = Producer{nodeId, outputPort, bitIndex};
                }
                ++bitIndex;
            }
            ++outputPort;
        }
    }

    int nextEdgeId = 0;
    for(auto& cellItem : cells.items()){
        int dstId = cellIds.at(cellItem.key());
        DFGNode* dstNode = dfg->node(dstId);
        auto& cellJson = cellItem.value();
        int inputOrdinal = 0;
        for(auto& portItem : cellJson.at("connections").items()){
            std::string direction = cellJson.at("port_directions").at(portItem.key()).get<std::string>();
            if(direction != "input") continue;
            int basePort = portIndex(portItem.key(), inputOrdinal);
            auto& bits = portItem.value();
            if(dstNode->operation() == "LUT"){
                int lutSize = static_cast<int>(bits.size());
                for(int i = 0; i < lutSize; ++i){
                    int operand = i;
                    auto& bit = bits[i];
                    if(bit.is_string()){
                        std::string value = bit.get<std::string>();
                        dstNode->setFineImmediate(operand, value == "1");
                        continue;
                    }
                    auto producer = producers.find(bit.get<int>());
                    if(producer == producers.end())
                        throw std::runtime_error("LUT input has no producer");
                    DFGEdge* edge = new DFGEdge(nextEdgeId);
                    edge->setEdge(1, producer->second.node, producer->second.port,
                                  dstId, operand);
                    dfg->addEdge(edge);
                    ++nextEdgeId;
                }
            }else{
                bool allConstant = true;
                uint64_t immediate = 0;
                for(int i = 0; i < static_cast<int>(bits.size()); ++i){
                    if(bits[i].is_number_integer()) allConstant = false;
                    else if(bits[i].get<std::string>() == "1") immediate |= uint64_t(1) << i;
                }
                if(allConstant){
                    if(bits.size() == 1) dstNode->setFineImmediate(basePort, immediate);
                    else {
                        dstNode->setImm(immediate);
                        dstNode->setImmIdx(basePort);
                    }
                    ++inputOrdinal;
                    continue;
                }
                Producer first;
                bool haveFirst = false;
                for(auto& bit : bits){
                    if(!bit.is_number_integer())
                        throw std::runtime_error("mixed constant/net coarse Yosys port is unsupported");
                    auto it = producers.find(bit.get<int>());
                    if(it == producers.end())
                        throw std::runtime_error("Yosys input has no producer");
                    if(!haveFirst){ first = it->second; haveFirst = true; }
                    else if(it->second.node != first.node || it->second.port != first.port)
                        throw std::runtime_error("coarse input bus has multiple producers");
                }
                int width = static_cast<int>(bits.size());
                DFGEdge* edge = new DFGEdge(nextEdgeId);
                edge->setEdge(width, first.node, first.port, dstId, basePort);
                dfg->addEdge(edge);
                if(width == 1) dfg->setFineGrained(true);
                else dfg->setCGWidth(width);
                ++nextEdgeId;
            }
            ++inputOrdinal;
        }
    }

    dfg->topoSortNodes();
    return dfg;
}

std::string shellQuote(const fs::path& path){
    std::string value = path.string();
    std::string quoted = "\"";
    for(char c : value){
        if(c == '"') quoted += "\\\"";
        else quoted += c;
    }
    return quoted + "\"";
}

} // namespace

DFG* YosysFrontend::synthesize(
    DFG* input,
    const std::string& workDir,
    const std::string& designName,
    int maxLutInputs,
    const std::string& yosysExecutable,
    const std::string& cellLibrary){
    if(maxLutInputs <= 0) throw std::runtime_error("ADG exposes no LUT inputs");
    fs::path dir = fs::absolute(workDir);
    fs::create_directories(dir);
    fs::path rtlilFile = dir / "aufora_yosys_input.il";
    fs::path jsonFile = dir / "aufora_yosys_mapped.json";
    fs::path scriptFile = dir / "aufora_yosys.ys";
    fs::path generatedLibrary = dir / "aufora_cells.v";
    emitDfgRtlil(input, rtlilFile, designName);
    emitCellLibrary(input, generatedLibrary);

    std::ofstream script(scriptFile);
    if(!script) throw std::runtime_error("cannot create Yosys script");
    fs::path library = cellLibrary.empty() ? generatedLibrary : fs::absolute(cellLibrary);
    script << "read_verilog -lib \"" << library.generic_string() << "\"\n"
           << "read_rtlil \"" << rtlilFile.generic_string() << "\"\n"
           << "hierarchy -top " << sanitizeName(designName) << "\n"
           << "opt\ncheck\ntechmap -map +/techmap.v\n"
           << "abc -lut " << maxLutInputs << "\n"
           << "opt\ncheck\nwrite_json \"" << jsonFile.generic_string() << "\"\n";
    script.close();

    std::string command = "\"" + yosysExecutable + "\" -q -s " + shellQuote(scriptFile);
    int status = std::system(command.c_str());
    if(status != 0)
        throw std::runtime_error("Yosys failed with status " + std::to_string(status));
    return parseYosysJson(jsonFile, designName, input);
}
