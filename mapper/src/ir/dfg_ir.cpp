
#include "ir/dfg_ir.h"
#include "ir/cstore_contract.h"

namespace {

bool requiresRoutedConstant(DFGNode *destination, int logicalPort) {
    if(!destination)
        return false;
    const std::string &operation = destination->operation();
    // CSTORE consumes all three values on physical IOB lanes. Ordinary STORE
    // retains its static address in the IOB access pattern, but its data port
    // is likewise a physical lane and cannot consume a node-local immediate.
    return (operation == "CSTORE" && logicalPort >= 0 && logicalPort < 3) ||
           (operation == "STORE" && logicalPort == 0);
}

DFGNode *materializeRoutedConstant(DFG *dfg, int &nextNodeId,
                                   uint64_t value) {
    DFGNode *node = new DFGNode();
    node->setId(++nextNodeId);
    node->setName("PASS" + std::to_string(node->id()));
    node->setOperation("PASS");
    node->setImm(value);
    node->setImmIdx(0);
    dfg->addNode(node);
    return node;
}

} // namespace


DFGIR::DFGIR(std::string filename)
{
    _dfg = parseDFG(filename);
}

DFGIR::DFGIR(LLVMCDFG * CDFG)
{
    _dfg = parseDFGJFromMLIRCDFG(CDFG);
}

DFGIR::~DFGIR()
{
    if(_dfg){
        delete _dfg;
    }
}

// get node ID according to name
int DFGIR::nodeId(std::string name){
    if(_nodeName2id.count(name)){
        return _nodeName2id[name];
    } else{
        return -1;
    }
}


void DFGIR::setNodeId(std::string name, int id){
    _nodeName2id[name] = id;
}

// // get input index according to name
// int DFGIR::inputIdx(std::string name){
//     if(_inputName2idx.count(name)){
//         return _inputName2idx[name];
//     } else{
//         return -1;
//     }
// }


// void DFGIR::setInputIdx(std::string name, int idx){
//     _inputName2idx[name] = idx;
// }

// // get output index according to name
// int DFGIR::outputIdx(std::string name){
//     if(_outputName2idx.count(name)){
//         return _outputName2idx[name];
//     } else{
//         return -1;
//     }
// }

// void DFGIR::setOutputIdx(std::string name, int idx){
//     _outputName2idx[name] = idx;
// }

// get constant value according to name
uint64_t DFGIR::constValue(std::string name){
    if(_name2Consts.count(name)){
        return _name2Consts[name];
    } else{
        return 0;
    }
}

bool DFGIR::isConst(std::string name){
    return _name2Consts.count(name);
}


void DFGIR::setConst(std::string name, uint64_t value){
    _name2Consts[name] = value;
}


// // get input index according to id
// int DFGIR::inputIdx(int id){
//     if(_inputId2idx.count(id)){
//         return _inputId2idx[id];
//     } else{
//         return -1;
//     }
// }


// void DFGIR::setInputIdx(int id, int idx){
//     _inputId2idx[id] = idx;
// }

// // get output index according to id
// int DFGIR::outputIdx(int id){
//     if(_outputId2idx.count(id)){
//         return _outputId2idx[id];
//     } else{
//         return -1;
//     }
// }

// void DFGIR::setOutputIdx(int id, int idx){
//     _outputId2idx[id] = idx;
// }

// get constant value according to id
uint64_t DFGIR::constValue(int id){
    if(_id2Consts.count(id)){
        return _id2Consts[id];
    } else{
        return 0;
    }
}

bool DFGIR::isConst(int id){
    return _id2Consts.count(id);
}


void DFGIR::setConst(int id, uint64_t value){
    _id2Consts[id] = value;
}

// // deprecated!!!
// DFG* DFGIR::parseDFGDot(std::string filename){
//     std::ifstream ifs(filename);
//     if(!ifs){
//         std::cout << "Cannot open DFG file: " << filename << std::endl;
//         exit(1);
//     }
//     DFG* dfg = new DFG();
//     dfg->setId(0); // DFG id = 0, node id = 1,...,n
//     std::string line;
//     int edgeIdx = 0;
//     std::stringstream edge_stream;
//     while (getline(ifs, line))
//     {
//         remove(line.begin(), line.end(), ' ');
//         if(line.empty() || line.substr(0, 2) == "//"){
//             continue;
//         }
//         int idx0 = line.find("[");
//         int idx1 = line.find("=");
//         int idx2 = line.find("]");
//         int idx3 = line.find("->");
//         if (idx0 != std::string::npos && idx3 == std::string::npos) { //nodes
//             std::string nodeName = line.substr(0, idx0);
//             std::string opName = line.substr(idx1 + 1, idx2-idx1-1);
//             std::transform(opName.begin(), opName.end(), opName.begin(), toupper);
//             int id = _nodeName2id.size() + 1;
//             setNodeId(nodeName, id);
//             if(opName == "INPUT"){
//                 int idx = _inputName2idx.size();
//                 setInputIdx(nodeName, idx);
//                 dfg->setInputName(idx, nodeName);
//             } else if(opName == "OUTPUT"){
//                 int idx = _outputName2idx.size();
//                 setOutputIdx(nodeName, idx);
//                 dfg->setOutputName(idx, nodeName);
//             } else if(opName == "CONST"){
//                 setConst(nodeName, 0); // ???????????????? PARSE CONST VALUE
//             } else{
//                 DFGNode* dfg_node = new DFGNode();
//                 dfg_node->setId(id);
//                 dfg_node->setName(nodeName);
//                 dfg_node->setOperation(opName);
//                 dfg->addNode(dfg_node);
//             }
//         }
//     }
//     // rescan the file to parse the edges, making sure all the nodes are already parsed
//     ifs.clear();
//     ifs.seekg(0);
//     while (getline(ifs, line))
//     {
//         remove(line.begin(), line.end(), ' ');
//         if(line.empty() || line.substr(0, 2) == "//"){
//             continue;
//         }
//         int idx0 = line.find("[");
//         int idx1 = line.find("=");
//         int idx2 = line.find("]");
//         int idx3 = line.find("->");
//         if (idx0 != std::string::npos && idx3 != std::string::npos) { // edges            
//             std::string srcName = line.substr(0, idx3);
//             std::string dstName = line.substr(idx3+2, idx0-idx3-2);
//             int dstPort = std::stoi(line.substr(idx1+1, idx2-idx1-1));
//             int srcPort = 0; // default one output for each node
//             if(isConst(srcName)){ // merge const node into the node connected to it
//                 DFGNode* node = dfg->node(nodeId(dstName));
//                 node->setImm(constValue(srcName));
//                 node->setImmIdx(dstPort);
//             } else{
//                 DFGEdge* edge = new DFGEdge(edgeIdx++);
//                 if(inputIdx(srcName) >= 0){ // INPUT                        
//                     edge->setEdge(0, inputIdx(srcName), nodeId(dstName), dstPort);                    
//                 } else if(outputIdx(dstName) >= 0){ // output
//                     edge->setEdge(nodeId(srcName), srcPort, 0, outputIdx(dstName));
//                 } else{
//                     edge->setEdge(nodeId(srcName), srcPort, nodeId(dstName), dstPort);
//                 }
//                 dfg->addEdge(edge);
//             }         
//         }
//     }
//     return dfg;
// }

/// remove the blank at the beginning and the end of one string
static std::string trim(std::string s) 
{
	if (s.empty()) 
	{
		return s;
	}
	s.erase(0,s.find_first_not_of(" "));
	s.erase(s.find_last_not_of(" ") + 1);
	return s;
}

// split a string and convert to integer array
std::vector<int> strSplit2Int(const std::string &str, char split){
    std::vector<int> res;
    std::string new_str = str + split;
    size_t pos = new_str.find(split);
    while(pos != new_str.npos){        
        int value = std::stoi(new_str.substr(0, pos));
        res.push_back(value);
        new_str = new_str.substr(pos+1, new_str.size());
        pos = new_str.find(split);
    }
    return res;
}

// split a string and convert to uint64_t array
std::vector<uint64_t> strSplit2Uint64(const std::string &str, char split){
    std::vector<uint64_t> res;
    std::string new_str = str + split;
    size_t pos = new_str.find(split);
    while(pos != new_str.npos){        
        int value = std::stoull(new_str.substr(0, pos));
        res.push_back(value);
        new_str = new_str.substr(pos+1, new_str.size());
        pos = new_str.find(split);
    }
    return res;
}

// split a string and convert to uint64_t array
std::vector<std::string> strSplit2Str(const std::string &str, char split){
    std::vector<std::string> res;
    std::string new_str = str + split;
    size_t pos = new_str.find(split);
    while(pos != new_str.npos){        
        std::string value = new_str.substr(0, pos);
        res.push_back(trim(value));
        new_str = new_str.substr(pos+1, new_str.size());
        pos = new_str.find(split);
    }
    return res;
}

// Json file transformed from dot file using graphviz
DFG* DFGIR::parseDFGJson(std::string filename){
    std::ifstream ifs(filename);
    if(!ifs){
        std::cout << "Cannot open DFG file: " << filename << std::endl;
        exit(1);
    }
    json dfgJson;
    ifs >> dfgJson;
    DFG* dfg = new DFG();
    dfg->setId(0); // DFG id = 0, node id = 1,...,n
    int nextMaterializedNodeId = 0;
    // parse nodes
    for(auto& nodeJson : dfgJson["objects"]){
        std::string nodeName = nodeJson["name"].get<std::string>();
        std::string opName = nodeJson["opcode"].get<std::string>();
        std::transform(opName.begin(), opName.end(), opName.begin(), toupper);
        int id = nodeJson["_gvid"].get<int>() + 1; // start from 1
        nextMaterializedNodeId = std::max(nextMaterializedNodeId, id);
        // if(opName == "INPUT"){
        //     int idx = _inputId2idx.size();
        //     setInputIdx(id, idx);
        //     dfg->setInputName(idx, nodeName);
        // } else if(opName == "OUTPUT"){
        //     int idx = _outputId2idx.size();
        //     setOutputIdx(id, idx);
        //     dfg->setOutputName(idx, nodeName);
        // } else if(opName == "CONST"){
        if(opName == "CONST"){
            uint64_t value = 0;
            if(nodeJson.contains("value")){
                value = std::stoull(nodeJson["value"].get<std::string>());
            }
            setConst(id, value); 
        } else{
            DFGNode* dfg_node;            
            if(opName == "INPUT" || opName == "OUTPUT" || opName == "LOAD" || opName == "STORE" || opName == "CLOAD" || opName == "CSTORE"){
                dfg->addIONode(id);
                DFGIONode* dfg_io_node = new DFGIONode();
                if(nodeJson.contains("ref_name")){
                    dfg_io_node->setMemRefName(nodeJson["ref_name"].get<std::string>());
                }
                if(nodeJson.contains("offset")){
                    int offset = std::stoi(nodeJson["offset"].get<std::string>());
                    std::string patStr = nodeJson["offset"].get<std::string>();
                    std::vector<int> values = strSplit2Int(patStr, ',');
                    dfg_io_node->setMemOffset(values[0]);
                    if(values.size() > 1){
                        dfg_io_node->setReducedMemOffset(values[1]);
                    }else{
                        dfg_io_node->setReducedMemOffset(0);
                    }
                }else{
                    dfg_io_node->setMemOffset(0);
                    dfg_io_node->setReducedMemOffset(0);
                }
                if(nodeJson.contains("size")){
                    int size = std::stoi(nodeJson["size"].get<std::string>());
                    dfg_io_node->setMemSize(size);
                }
                if(nodeJson.contains("pattern")){
                    std::string patStr = nodeJson["pattern"].get<std::string>();
                    std::vector<int> values = strSplit2Int(patStr, ',');
                    for(int i = 0; i < values.size(); i += 2){
                        dfg_io_node->addPatternLevel(values[i], values[i+1]);
                    }
                }
                if(nodeJson.contains("cycles")){
                    int cycles = std::stoi(nodeJson["cycles"].get<std::string>());
                    dfg_io_node->addPatternLevel(0, cycles);
                }
                dfg_node = dfg_io_node;
            }else{
                dfg_node = new DFGNode();
                if(nodeJson.contains("acc_params")){
                    std::string patStr = nodeJson["acc_params"].get<std::string>();
                    std::vector<uint64_t> values = strSplit2Uint64(patStr, ',');
                    assert(values.size() >= 3);
                    dfg_node->setInitVal(values[0]);
                    dfg_node->setCycles((int)values[1]);
                    dfg_node->setInterval((int)values[2]);
                    dfg_node->setRepeats((int)values[3]);
                    int first = std::stoi(nodeJson["acc_first"].get<std::string>());
                    if(first){
                        dfg_node->setIsAccFirst(true);
                    }else{
                        dfg_node->setIsAccFirst(false);
                    }
                }
                if(nodeJson.contains("loop_index_acc") &&
                   ((nodeJson["loop_index_acc"].is_string() &&
                     nodeJson["loop_index_acc"].get<std::string>() == "1") ||
                    (nodeJson["loop_index_acc"].is_boolean() &&
                     nodeJson["loop_index_acc"].get<bool>()))) {
                    dfg_node->setLoopIndexAcc();
                }
            }
            dfg_node->setId(id);
            dfg_node->setName(nodeName);
            dfg_node->setOperation(opName);
            dfg->addNode(dfg_node);
            // if(std::find(this->addsub.begin(), this->addsub.end(), opName) != this->addsub.end()){
			//     this->optypecount.numaddsub +=1;
		    // }else if(std::find(this->logic.begin(), this->logic.end(), opName) != this->logic.end()){
			//     this->optypecount.numlogic +=1;
		    // }else if(std::find(this->multiplier.begin(), this->multiplier.end(), opName) != this->multiplier.end()){
			//     this->optypecount.nummul +=1;
		    // }else if(std::find(this->comp.begin(), this->comp.end(), opName) != this->comp.end()){
			//     this->optypecount.numcomp +=1;
		    // }else{
			//     std::cout << "Operand not find"<< opName << std::endl;
		    // }
        }
    }
    // parse edges
    std::map<int, int> immediateCount;
    for(auto& edgeJson : dfgJson["edges"]){
        int srcId = edgeJson["tail"].get<int>() + 1;
        int dstId = edgeJson["head"].get<int>() + 1;
        int dstPort;
        int srcPort; // default one output for each node
        if(edgeJson.contains("operand")){
            dstPort = std::stoi(edgeJson["operand"].get<std::string>());
        }else if(edgeJson.contains("headport")){
            auto str = edgeJson["headport"].get<std::string>(); // in0, in1...
            dstPort = std::stoi(str.substr(2, 1));
        // }else if(outputIdx(dstId) < 0){ // not annotate dst-port, not output port
        //     DFGNode* node = dfg->node(dstId);
        //     dstPort = node->numInputs();
        }
        if(edgeJson.contains("tailport")){
            auto str = edgeJson["tailport"].get<std::string>(); // out0, out1...
            srcPort = std::stoi(str.substr(3, 1));
        }else{ // default one output for each node
            srcPort = 0;
        }
        if(isConst(srcId)){ // merge const node into the node connected to it
            DFGNode* node = dfg->node(dstId);
            if(node && (node->operation() == "STORE" ||
                        node->operation() == "CSTORE"))
                ++immediateCount[dstId];
            if(requiresRoutedConstant(node, dstPort)){
                // VITRA IOBs do not have an operation-local immediate input.
                // Materialize the one accepted STORE/CSTORE constant in a GPE
                // PASS so it becomes a real routed operand at the exact logical
                // port instead of silently reading zero at the IOB boundary.
                DFGNode *constant = materializeRoutedConstant(
                    dfg, nextMaterializedNodeId, constValue(srcId));
                int edgeId = edgeJson["_gvid"].get<int>();
                DFGEdge* edge = new DFGEdge(edgeId);
                edge->setEdge(constant->id(), 0, dstId, dstPort);
                dfg->addEdge(edge);
            }else{
                node->setImm(constValue(srcId));
                node->setImmIdx(dstPort);
            }
        } else{
            int edgeId = edgeJson["_gvid"].get<int>();
            DFGEdge* edge = new DFGEdge(edgeId);
            // if(inputIdx(srcId) >= 0){ // INPUT                        
            //     edge->setEdge(0, inputIdx(srcId), dstId, dstPort);                    
            // } else if(outputIdx(dstId) >= 0){ // output
            //     edge->setEdge(srcId, srcPort, 0, outputIdx(dstId));
            // } else{
            edge->setEdge(srcId, srcPort, dstId, dstPort);
            // }
            dfg->addEdge(edge);
        }         
    }
    for(auto &elem : dfg->nodes()){
        DFGNode *node = elem.second;
        if(node->operation() != "STORE" && node->operation() != "CSTORE")
            continue;
        const std::string violation =
            CStoreContract::immediateViolation(immediateCount[node->id()]);
        if(!violation.empty()){
            std::cout << "Invalid " << node->operation() << " DFG node "
                      << node->id() << ": " << violation << std::endl;
            exit(1);
        }
    }
    // // add const operand for nodes with only one operand, should be avoid
    // for(auto &elem : dfg->nodes()){
    //     DFGNode* node = elem.second;
    //     if(!dfg->isIONode(elem.first) && node->numInputs() < 2){ // just for verification, not all cases
    //         node->setImm(0);
    //         node->setImmIdx(1);
    //     }
    // }
    return dfg;
}

static inline bool IsPatternConstant(const std::vector<std::pair<std::string, std::string>>& pattern){
    for(auto pair : pattern){
        if(pair.first != "__const__" && pair.first != "-" && pair.first.size() != 0
         ||pair.second != "__const__" && pair.second != "-" && pair.second.size() != 0){
            return false;
        }
    }
    return true;
}

static inline bool IsAccConstant(const std::vector<std::string>& acc){
    for(auto str : acc){
        if(str != "__const__" && str != "-" && str.size() != 0){
            return false;
        }
    }
    return true;
}


// Get a DFG from MLIRCDFG in mlir_cdfg.h
DFG* DFGIR::parseDFGJFromMLIRCDFG(LLVMCDFG * CDFG){
    DFG* dfg = new DFG();
    dfg->setId(0); // DFG id = 0, node id = 1,...,n
    // parse nodes
    const std::map<int, LLVMCDFGNode*> nodes = CDFG->nodes();
    int nextMaterializedNodeId = nodes.empty() ? 0 : nodes.rbegin()->first + 1;
    for(auto &elem : nodes){
        int id = elem.first + 1;
        LLVMCDFGNode* node = elem.second;
        std::string NodeName = node->getTypeName() + std::to_string(id);
        std::string opName = node->getTypeName();
        std::cout << NodeName << "\n";
        std::transform(opName.begin(), opName.end(), opName.begin(), toupper);
        // int id = nodeJson["_gvid"].get<int>() + 1; // start from 1
        // if(opName == "INPUT"){
        //     int idx = _inputId2idx.size();
        //     setInputIdx(id, idx);
        //     dfg->setInputName(idx, nodeName);
        // } else if(opName == "OUTPUT"){
        //     int idx = _outputId2idx.size();
        //     setOutputIdx(id, idx);
        //     dfg->setOutputName(idx, nodeName);
        // } else if(opName == "CONST"){
        if(opName == "CONST"){
            uint64_t value = 0;
            if(node->hasConst()){
                // std::cout << node->constValHex_str() << "\n";
                value = std::stoull(node->constValHex_str(), NULL, 16);
            }
            setConst(id, value); 
        } else{
            DFGNode* dfg_node;            
            if(opName == "INPUT" || opName == "OUTPUT" || opName == "LOAD" || opName == "STORE" || opName == "CLOAD" || opName == "CSTORE"){
                dfg->addIONode(id);
                DFGIONode* dfg_io_node = new DFGIONode();
                dfg_node = dfg_io_node;
                dfg_io_node->setMemRefName(node->getMemrefName());
                // if(nodeJson.contains("ref_name")){
                //     dfg_io_node->setMemRefName(nodeJson["ref_name"].get<std::string>());
                // }
                dfg_io_node->setMemOffset(0);
                dfg_io_node->setReducedMemOffset(atoi(node->getInitAddr().c_str())); //// The first addr iob access
                // if(nodeJson.contains("offset")){
                //     int offset = std::stoi(nodeJson["offset"].get<std::string>());
                //     std::string patStr = nodeJson["offset"].get<std::string>();
                //     std::vector<int> values = strSplit2Int(patStr, ',');
                //     dfg_io_node->setMemOffset(values[0]);
                //     if(values.size() > 1){
                //         dfg_io_node->setReducedMemOffset(values[1]);
                //     }else{
                //         dfg_io_node->setReducedMemOffset(0);
                //     }
                // }else{
                //     dfg_io_node->setMemOffset(0);
                //     dfg_io_node->setReducedMemOffset(0);
                // }
                dfg_io_node->setMemSize(node->getMemrefSize());

                /// set dfg io node
                dfg_io_node->setPingpong(node->isPingpong());

                // if(nodeJson.contains("size")){
                //     int size = std::stoi(nodeJson["size"].get<std::string>());
                //     dfg_io_node->setMemSize(size);
                // }
                // std::vector<int> values = strSplit2Int(node->getLinearAccess(), ',');
                std::vector<std::string> values = strSplit2Str(node->getLinearAccess(), ',');
                std::vector<std::pair<std::string, std::string>> VarPattern;
                for(size_t i = 0; i + 1 < values.size(); i += 2){
                    if(values[i].empty() || values[i + 1].empty())
                        continue;
                    int value0 = 0x4fe, value1 = 0x4fe;
                    std::string var0 = "__const__", var1 = "__const__";

                    if(isInteger(values[i]))
                        value0 = std::stoi(values[i]);
                    else
                        var0 = values[i];

                    if(isInteger(values[i+1]))
                        value1 = std::stoi(values[i+1]);
                    else
                        var1 = values[i+1];

                    dfg_io_node->addPatternLevel(value0, value1);
                    VarPattern.push_back(std::pair(var0, var1));
                }
                if(!IsPatternConstant(VarPattern)){
                    dfg->VariableConfigNodes[dfg_node] = VariableConfig(VariableConfig::NodeT::IONode);
                    dfg->VariableConfigNodes[dfg_node].pattern = VarPattern;
                    dfg->VariableConfigNodes[dfg_node].memOffset = "__const__";
                    dfg->VariableConfigNodes[dfg_node].reducedmemOffset = "__const__";
                    dfg->VariableConfigNodes[dfg_node].print();
                }
                // if(nodeJson.contains("pattern")){
                //     // std::string patStr = nodeJson["pattern"].get<std::string>();
                //     std::vector<int> values = strSplit2Int(patStr, ',');
                //     for(int i = 0; i < values.size(); i += 2){
                //         dfg_io_node->addPatternLevel(values[i], values[i+1]);
                //     }
                // }
                // if(nodeJson.contains("cycles")){
                //     int cycles = std::stoi(nodeJson["cycles"].get<std::string>());
                //     dfg_io_node->addPatternLevel(0, cycles);
                // }
                // dfg_node = dfg_io_node;
            }else{
                dfg_node = new DFGNode();
                if(node->isAcc()){
                    // std::string patStr = nodeJson["acc_params"].get<std::string>();
                    // std::vector<uint64_t> values = strSplit2Uint64(node->getACCInfo_str(), ',');
                    // assert(values.size() >= 3);
                    // dfg_node->setInitVal(values[0]);
                    // dfg_node->setCycles((int)values[1]);
                    // dfg_node->setInterval((int)values[2]);
                    // dfg_node->setRepeats((int)values[3]);
                    // dfg_node->setIsAccFirst(true);

                    std::vector<std::string> values = strSplit2Str(node->getACCInfo_str(), ',');
                    std::vector<std::string> VarACC;
                    // std::vector<int> values_int;
                    assert(values.size() >= 3);
                    if(isInteger(values[0])){
                        dfg_node->setInitVal(std::stoi(values[0]));
                        VarACC.push_back("__const__");
                    }
                    else{
                        dfg_node->setInitVal(0x4fe);
                        VarACC.push_back(values[0]);
                    }

                    if(isInteger(values[1])){
                        dfg_node->setCycles(std::stoi(values[1]));
                        VarACC.push_back("__const__");
                    }
                    else{
                        dfg_node->setCycles(0x4fe);
                        VarACC.push_back(values[1]);
                    }

                    if(isInteger(values[2])){
                        dfg_node->setInterval(std::stoi(values[2]));
                        VarACC.push_back("__const__");
                    }
                    else{
                        dfg_node->setInterval(0x4fe);
                        VarACC.push_back(values[2]);
                    }

                    if(isInteger(values[3])){
                        dfg_node->setRepeats(std::stoi(values[3]));
                        VarACC.push_back("__const__");
                    }
                    else{
                        dfg_node->setRepeats(0x4fe);
                        VarACC.push_back(values[3]);
                    }
                    ///// acc first

                    if(opName == "ISEL" || node->isLoopIndexAcc()){
                        dfg_node->setIsAccFirst(false);
                    }
                    else{
                        dfg_node->setIsAccFirst(true);
                    }
                    if(node->isLoopIndexAcc()){
                        dfg_node->setLoopIndexAcc();
                    }
                        
                    if(!IsAccConstant(VarACC)){
                        dfg->VariableConfigNodes[dfg_node] = VariableConfig(VariableConfig::NodeT::ACCNode);
                        dfg->VariableConfigNodes[dfg_node].initVal = VarACC[0];
                        dfg->VariableConfigNodes[dfg_node].cycles = VarACC[1];
                        dfg->VariableConfigNodes[dfg_node].interval = VarACC[2];
                        dfg->VariableConfigNodes[dfg_node].repeats = VarACC[3];
                    }

                    // int first = std::stoi(nodeJson["acc_first"].get<std::string>());
                    // if(first){
                    //     dfg_node->setIsAccFirst(true);
                    // }else{
                    //     dfg_node->setIsAccFirst(false);
                    // }
                }
            }
            dfg_node->setId(id);
            dfg_node->setName(NodeName);
            dfg_node->setOperation(opName);
            dfg->addNode(dfg_node);
            // if(std::find(this->addsub.begin(), this->addsub.end(), opName) != this->addsub.end()){
			//     this->optypecount.numaddsub +=1;
		    // }else if(std::find(this->logic.begin(), this->logic.end(), opName) != this->logic.end()){
			//     this->optypecount.numlogic +=1;
		    // }else if(std::find(this->multiplier.begin(), this->multiplier.end(), opName) != this->multiplier.end()){
			//     this->optypecount.nummul +=1;
		    // }else if(std::find(this->comp.begin(), this->comp.end(), opName) != this->comp.end()){
			//     this->optypecount.numcomp +=1;
		    // }else{
			//     std::cout << "Operand not find"<< opName << std::endl;
		    // }
        }
    }
    // parse edges
    std::map<std::pair<LLVMCDFGNode*, LLVMCDFGNode*>, std::vector<int>> visited_edgeidx;    
    std::map<std::pair<LLVMCDFGNode*, LLVMCDFGNode*>, size_t> rawEdgePortIdx;
    std::map<int, LLVMCDFGEdge*> edges = CDFG->edges();
    std::map<int, std::map<int, int>> logicalPortUse;
    std::map<int, int> immediateCount;
    for(auto &elem : edges){
        int edge_id = elem.first;
        LLVMCDFGEdge* edge = elem.second;
        int srcId = edge->src()->id() + 1;
        int dstId = edge->dst()->id() + 1;
        std::vector<int> dstPorts;
        int srcPort, dstPort; // default one output for each node

        const std::string destinationType = edge->dst()->getTypeName();
        const bool isCStoreDestination = destinationType == "CSTORE";
        const bool isStoreDestination = destinationType == "store";
        const bool hasExactIOContract =
            isCStoreDestination || isStoreDestination;
        if(hasExactIOContract){
            auto inputInfo = edge->dst()->inputInfoMap().find(edge->src());
            auto edgePair = std::make_pair(edge->src(), edge->dst());
            size_t& portIdx = rawEdgePortIdx[edgePair];
            if(inputInfo == edge->dst()->inputInfoMap().end() ||
               portIdx >= inputInfo->second.size()){
                std::cout << "Invalid "
                          << (isCStoreDestination ? "CSTORE" : "STORE")
                          << " DFG node " << dstId
                          << ": missing logical operand path for raw edge "
                          << edge_id << std::endl;
                exit(1);
            }
            dstPort = inputInfo->second[portIdx++].idx;
        }else{
            dstPorts = edge->dst()->getInputIndices(edge->src());
            if(dstPorts.size() > 1){
                for(int _ = 0; _ < dstPorts.size(); _++){
                    auto& visitedPorts = visited_edgeidx[std::make_pair(edge->src(), edge->dst())];

                    if (std::find(visitedPorts.begin(), visitedPorts.end(), dstPorts[_]) == visitedPorts.end()) {
                        visitedPorts.push_back(dstPorts[_]);
                        dstPort = dstPorts[_];
                        break;
                    }
                }
            }
            else{
                dstPort = dstPorts[0];
            }
        }

        srcPort = 0; // default one output for each node

        bool isBackEdge = edge->src()->isOutputBackEdge(edge->dst());
        if(hasExactIOContract){
            CStoreContract::recordNonMemoryLogicalPort(logicalPortUse[dstId],
                                                       dstPort,
                                                       edge->type() == EDGE_TYPE_MEM);
            // Count source-level constants before materialization. The v1 DFG
            // contract intentionally accepts at most one even though that one
            // is subsequently represented by a routed PASS for an IOB.
            if(edge->type() != EDGE_TYPE_MEM && isConst(srcId))
                ++immediateCount[dstId];
        }
        // if(isBackEdge){continue;}
        // if(edgeJson.contains("operand")){
        //     dstPort = std::stoi(edgeJson["operand"].get<std::string>());
        // }else if(edgeJson.contains("headport")){
        //     auto str = edgeJson["headport"].get<std::string>(); // in0, in1...
        //     dstPort = std::stoi(str.substr(2, 1));
        // // }else if(outputIdx(dstId) < 0){ // not annotate dst-port, not output port
        // //     DFGNode* node = dfg->node(dstId);
        // //     dstPort = node->numInputs();
        // }
        // if(edgeJson.contains("tailport")){
        //     auto str = edgeJson["tailport"].get<std::string>(); // out0, out1...
        //     srcPort = std::stoi(str.substr(3, 1));
        // }else{ // default one output for each node
        //     srcPort = 0;
        // }
        if(isConst(srcId)){ // merge const node into the node connected to it
            DFGNode* node = dfg->node(dstId);
            if(requiresRoutedConstant(node, dstPort)){
                DFGNode *constant = materializeRoutedConstant(
                    dfg, nextMaterializedNodeId, constValue(srcId));
                DFGEdge* DFGedge = new DFGEdge(edge_id);
                DFGedge->setEdge(constant->id(), 0, dstId, dstPort);
                DFGedge->setType(edge->type());
                dfg->addEdge(DFGedge);
                if(isBackEdge){
                    DFGedge->setBackEdge(true);
                    DFGedge->setIterDist(edge->IterDist());
                }
            }else{
                node->setImm(constValue(srcId));
                node->setImmIdx(dstPort);
            }
        } else{
            // int edgeId = edgeJson["_gvid"].get<int>();
            DFGEdge* DFGedge = new DFGEdge(edge_id);
            // if(inputIdx(srcId) >= 0){ // INPUT                        
            //     edge->setEdge(0, inputIdx(srcId), dstId, dstPort);                    
            // } else if(outputIdx(dstId) >= 0){ // output
            //     edge->setEdge(srcId, srcPort, 0, outputIdx(dstId));
            // } else{
            DFGedge->setEdge(srcId, srcPort, dstId, dstPort);
            // }
            DFGedge->setType(edge->type());
            dfg->addEdge(DFGedge);
            if(isBackEdge){
                int iterdist = edge->IterDist();
                DFGedge->setBackEdge(isBackEdge);
                DFGedge->setIterDist(iterdist);
            }
        }         
    }
    // // add const operand for nodes with only one operand, should be avoid
    // for(auto &elem : dfg->nodes()){
    //     DFGNode* node = elem.second;
    //     if(!dfg->isIONode(elem.first) && node->numInputs() < 2){ // just for verification, not all cases
    //         node->setImm(0);
    //         node->setImmIdx(1);
    //     }
    // }
    for(auto& elem : dfg->nodes()){
        DFGNode* node = elem.second;
        const bool isCStore = node->operation() == "CSTORE";
        const bool isStore = node->operation() == "STORE";
        if(!isCStore && !isStore){
            continue;
        }
        const std::string operation = isCStore ? "CSTORE" : "STORE";
        const int expectedOperands = isCStore ? 3 : 2;
        const std::string prefix = "Invalid " + operation + " DFG node " +
                                   std::to_string(node->id()) + ": ";
        if(Operations::numOperands(operation) != expectedOperands ||
           Operations::numRes(operation) != 0){
            std::cout << prefix << "operation spec must have "
                      << expectedOperands << " operands and 0 results"
                      << std::endl;
            exit(1);
        }
        const std::string immediateViolation =
            CStoreContract::immediateViolation(immediateCount[node->id()]);
        if(!immediateViolation.empty()){
            std::cout << prefix << immediateViolation << std::endl;
            exit(1);
        }
        const std::string portViolation =
            CStoreContract::logicalPortViolation(logicalPortUse[node->id()],
                                                 expectedOperands);
        if(!portViolation.empty()){
            std::cout << prefix << portViolation << std::endl;
            exit(1);
        }
    }
    dfg->printVariableConfigNodes();
    return dfg;
}


DFG* DFGIR::parseDFG(std::string filename, std::string format){
    // if(format == "dot"){
    //     return parseDFGDot(filename);
    // }else{
    return parseDFGJson(filename);
    // }
}
