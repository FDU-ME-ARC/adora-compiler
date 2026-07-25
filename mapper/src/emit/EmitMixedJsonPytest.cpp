#include "emit/EmitMixedJsonPytest.h"

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "llvm/Support/Format.h"
#include "mapper/configuration.h"

namespace {

struct IoRecord {
    int nodeId;
    int iobIndex;
    int deviceAddress;
    int sizeBytes;
    bool isInput;
    std::string tag;
    std::string nodeName;
    std::string refName;
    std::string operation;
};

bool isInputOp(const std::string& op){
    return op == "INPUT" || op == "LOAD" ||
           op == "CINPUT" || op == "CLOAD";
}

bool isOutputOp(const std::string& op){
    return op == "OUTPUT" || op == "STORE" ||
           op == "COUTPUT" || op == "CSTORE";
}

std::string pythonName(std::string name){
    for(char& c : name){
        if(!std::isalnum(static_cast<unsigned char>(c)) && c != '_') c = '_';
    }
    if(name.empty() || std::isdigit(static_cast<unsigned char>(name.front())))
        name = "dfg_" + name;
    return name;
}

std::string pythonString(const std::string& value){
    std::string result = "'";
    for(char c : value){
        if(c == '\\' || c == '\'') result.push_back('\\');
        if(c == '\n') result += "\\n";
        else result.push_back(c);
    }
    return result + "'";
}

std::vector<std::string> enableBytes(int bitCount, const std::set<int>& enabled){
    std::vector<unsigned> values(std::max(1, (bitCount + 7) / 8), 0);
    for(int bit : enabled){
        if(bit >= 0 && bit < bitCount)
            values[bit / 8] |= 1u << (bit % 8);
    }
    std::vector<std::string> result;
    for(unsigned value : values){
        std::ostringstream os;
        os << "0x" << std::hex << std::setw(2) << std::setfill('0') << value;
        result.push_back(os.str());
    }
    return result;
}

void emitStringList(llvm::raw_ostream& os,
                    const std::vector<std::string>& values){
    os << "[";
    for(size_t i = 0; i < values.size(); ++i){
        if(i) os << ", ";
        os << values[i];
    }
    os << "]";
}

} // namespace

void MixedJsonPytestEmitter::emit(
    llvm::raw_ostream& os, Mapping* mapping, const std::string& designName){
    if(!mapping) throw std::runtime_error("cannot emit an unmapped DFG");
    ADG* adg = mapping->getADG();
    DFG* dfg = mapping->getDFG();
    if(!adg || !dfg) throw std::runtime_error("mapping has no ADG or DFG");

    const int dataBytes = std::max(1, adg->bitWidth() / 8);
    const int bankSize = adg->iobSpadBankSize();
    if(bankSize <= 0) throw std::runtime_error("ADG has no IOB SPAD bank size");

    Configuration configuration(mapping);
    std::set<int> usedBanks;
    std::set<int> enabledIobs;
    std::vector<IoRecord> ioRecords;
    int inputOrdinal = 0;
    int outputOrdinal = 0;

    for(int nodeId : dfg->ioNodes()){
        auto* dfgIo = dynamic_cast<DFGIONode*>(dfg->node(nodeId));
        ADGNode* mapped = dfgIo ? mapping->mappedNode(dfgIo) : nullptr;
        auto* iob = dynamic_cast<IOBNode*>(mapped);
        if(!dfgIo || !iob) continue;

        const std::string op = dfgIo->operation();
        const bool input = isInputOp(op);
        const bool output = isOutputOp(op);
        if(!input && !output) continue;

        const std::vector<int>& banks = adg->iobToSpadBanks(iob->index());
        if(banks.empty()){
            throw std::runtime_error("mapped IOB has no connected SPAD bank");
        }
        int selectedBank = -1;
        for(int bank : banks){
            if(!usedBanks.count(bank)){
                selectedBank = bank;
                break;
            }
        }
        if(selectedBank < 0) selectedBank = banks.front();
        usedBanks.insert(selectedBank);

        int minBank = *std::min_element(banks.begin(), banks.end());
        int iobBaseAddress = ((selectedBank - minBank) * bankSize) / dataBytes;
        configuration.setDfgIoSpadAddr(nodeId, iobBaseAddress);

        int sizeBytes = dfgIo->memSize();
        if(sizeBytes <= 0) sizeBytes = dataBytes;
        std::string tag = input
            ? "input_" + std::to_string(inputOrdinal++)
            : "output_" + std::to_string(outputOrdinal++);
        ioRecords.push_back({
            nodeId, iob->index(), selectedBank * bankSize, sizeBytes, input,
            tag, dfgIo->name(), dfgIo->memRefName(), op
        });
        enabledIobs.insert(iob->index());
    }

    std::vector<CfgDataPacket> cfgData;
    configuration.getCfgData(cfgData);
    const int cfgAddrWidth = adg->cfgAddrWidth();
    const int alignWidth = cfgAddrWidth > 16 ? 32 : 16;
    const int alignHex = alignWidth / 4;

    std::set<int> configuredTiles = configuration.getConfiguredTiles();
    std::vector<std::string> iobEnable =
        enableBytes(adg->numIobNodes(), enabledIobs);
    std::vector<std::string> tileEnable =
        enableBytes(adg->tileNum(), configuredTiles);
    std::string functionName = "run_" + pythonName(designName);
    std::string publicFunctionName = pythonName(designName);
    std::string configName = "cfgbit_" + pythonName(designName);
    std::map<std::string, std::string> refArguments;
    std::set<std::string> inputRefs;
    std::set<std::string> outputRefs;
    for(const IoRecord& io : ioRecords){
        if(io.refName.empty()) continue;
        refArguments.emplace(io.refName, pythonName(io.refName));
        (io.isInput ? inputRefs : outputRefs).insert(io.refName);
    }

    os << "\"\"\"\n"
          "Automatically generated pytest/cocotb helper for a mixed-grained "
          "DFG JSON mapping.\n"
          "\"\"\"\n"
          "from test_runif import DeviceConfig, DeviceData, DeviceRuntime\n"
          "import numpy as np\n\n";

    os << configName << " = [\n";
    for(const CfgDataPacket& packet : cfgData){
        os << "    ";
        for(uint32_t data : packet.data){
            if(alignWidth == 32){
                os << "0x" << llvm::format_hex_no_prefix(data, alignHex)
                   << ", ";
            }else{
                os << "0x" << llvm::format_hex_no_prefix(data & 0xffff, alignHex)
                   << ", "
                   << "0x" << llvm::format_hex_no_prefix(data >> 16, alignHex)
                   << ", ";
            }
        }
        os << "0x" << llvm::format_hex_no_prefix(packet.addr, alignHex)
           << ",\n";
    }
    os << "]\n\n";

    os << "IO_METADATA = [\n";
    for(const IoRecord& io : ioRecords){
        os << "    {"
           << "'tag': " << pythonString(io.tag)
           << ", 'node': " << pythonString(io.nodeName)
           << ", 'ref_name': " << pythonString(io.refName)
           << ", 'operation': " << pythonString(io.operation)
           << ", 'direction': " << pythonString(io.isInput ? "input" : "output")
           << ", 'iob_index': " << io.iobIndex
           << ", 'address': 0x" << llvm::format_hex_no_prefix(io.deviceAddress, 1)
           << ", 'size_bytes': " << io.sizeBytes
           << "},\n";
    }
    os << "]\n\n";

    os << "async def " << functionName
       << "(runtime: DeviceRuntime, inputs=None):\n"
          "    \"\"\"Configure and run the mapped DFG; return output arrays by tag.\"\"\"\n"
          "    inputs = {} if inputs is None else inputs\n"
          "    iptrs, idata = [], []\n"
          "    optrs, outputs = [], {}\n"
          "    all_ptrs = []\n"
          "    for meta in IO_METADATA:\n"
          "        ptr = DeviceData(meta['address'], meta['size_bytes'])\n"
          "        all_ptrs.append(ptr)\n"
          "        words = (meta['size_bytes'] + 3) // 4\n"
          "        if meta['direction'] == 'input':\n"
          "            value = inputs.get(meta['tag'], inputs.get(meta['ref_name']))\n"
          "            if value is None:\n"
          "                value = np.zeros(words, dtype=np.uint32)\n"
          "            value = np.ascontiguousarray(value, dtype=np.uint32)\n"
          "            if value.nbytes > meta['size_bytes']:\n"
          "                raise ValueError(f\"{meta['tag']} exceeds its mapped SPAD buffer\")\n"
          "            iptrs.append(ptr)\n"
          "            idata.append(value)\n"
          "        else:\n"
          "            value = np.zeros(words, dtype=np.uint32)\n"
          "            optrs.append(ptr)\n"
          "            outputs[meta['tag']] = value\n\n"
          "    config = DeviceConfig(\n"
          "        config_values=" << configName << ",\n"
          "        iob_en=";
    emitStringList(os, iobEnable);
    os << ",\n        tile_en=";
    emitStringList(os, tileEnable);
    os << ",\n"
          "        data_ptr=all_ptrs,\n"
          "    )\n"
          "    stream = runtime.create_stream()\n"
          "    await stream.apply([config])\n"
          "    await stream.config(config_id=0)\n"
          "    for ptr, value in zip(iptrs, idata):\n"
          "        await stream.memcpyHostToDevice(\n"
          "            d_data=ptr, h_data=value, size=value.size, dtype='i')\n"
          "    # Predicate-false COUTPUT locations are untouched by hardware.\n"
          "    # Clear each output bank so multiport results can be merged safely.\n"
          "    for ptr, value in zip(optrs, outputs.values()):\n"
          "        await stream.memcpyHostToDevice(\n"
          "            d_data=ptr, h_data=value, size=value.size, dtype='i')\n"
          "    await stream.execution_start()\n"
          "    for ptr, value in zip(optrs, outputs.values()):\n"
          "        await stream.memcpyDeviceToHost(\n"
          "            d_data=ptr, h_data=value, size=value.nbytes)\n"
          "    await stream.synchronize()\n"
          "    return outputs\n";

    if(!refArguments.empty()){
        os << "\n\nasync def " << publicFunctionName
           << "(runtime: DeviceRuntime";
        for(const auto& ref : refArguments)
            os << ", " << ref.second << ": np.ndarray";
        os << "):\n"
              "    \"\"\"GEMM-style entry point using arrays named by JSON ref_name.\"\"\"\n"
              "    input_values = {\n";
        for(const std::string& ref : inputRefs)
            os << "        " << pythonString(ref) << ": "
               << refArguments.at(ref) << ",\n";
        os << "    }\n"
              "    raw_outputs = await " << functionName
           << "(runtime, inputs=input_values)\n"
              "    output_targets = {\n";
        for(const std::string& ref : outputRefs)
            os << "        " << pythonString(ref) << ": "
               << refArguments.at(ref) << ",\n";
        os << "    }\n"
              "    for ref_name, target_value in output_targets.items():\n"
              "        target = np.asarray(target_value).reshape(-1)\n"
              "        target.fill(0)\n"
              "        for meta in IO_METADATA:\n"
              "            if (meta['direction'] != 'output' or\n"
              "                    meta['ref_name'] != ref_name):\n"
              "                continue\n"
              "            source = raw_outputs[meta['tag']].reshape(-1)\n"
              "            count = min(target.size, source.size)\n"
              "            active = source[:count] != 0\n"
              "            target[:count][active] = source[:count][active]\n";
    }
}
