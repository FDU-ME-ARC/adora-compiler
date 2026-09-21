
#include "ir/adg_ir.h"

#include <cctype>
#include <limits>

namespace {

bool rawIOBCapableOfCStore(const json &attrs,
                          const std::map<int, std::string> &iobModeNames) {
    auto operations = attrs.find("operations");
    if(operations != attrs.end()){
        if(!operations->is_array())
            return false;
        for(const auto &operation : *operations){
            if(operation.is_string() && operation.get<std::string>() == "CSTORE")
                return true;
        }
        return false;
    }

    auto mode = attrs.find("iob_mode");
    if(mode == attrs.end() || !mode->is_number_integer())
        return false;
    auto name = iobModeNames.find(mode->get<int>());
    return name != iobModeNames.end() && name->second != "FIFO_MODE" &&
           name->second != "SRAM_MODE";
}

[[noreturn]] void invalidRawCStoreIOB(int moduleId, const json &attrs,
                                      const std::string &message) {
    std::cout << "Invalid CSTORE IOB module " << moduleId;
    auto index = attrs.find("iob_index");
    if(index != attrs.end() && index->is_number_integer())
        std::cout << " (iob_index " << index->get<int>() << ")";
    std::cout << ": " << message << std::endl;
    exit(1);
}

void validateRawCStoreIOB(int moduleId, const json &attrs) {
    auto index = attrs.find("iob_index");
    if(index == attrs.end())
        invalidRawCStoreIOB(moduleId, attrs, "missing iob_index");
    if(!index->is_number_integer())
        invalidRawCStoreIOB(moduleId, attrs,
                            "iob_index must be an integer");

    auto connections = attrs.find("connections");
    if(connections == attrs.end() || !connections->is_object())
        invalidRawCStoreIOB(moduleId, attrs,
                            "connections must be an object");
    const std::string maxEdgeId =
        std::to_string(std::numeric_limits<int>::max());
    for(const auto &element : connections->items()){
        const std::string &edgeId = element.key();
        if(edgeId.empty() ||
           !std::all_of(edgeId.begin(), edgeId.end(), [](unsigned char ch) {
               return std::isdigit(ch);
           }) ||
           edgeId.size() > maxEdgeId.size() ||
           (edgeId.size() == maxEdgeId.size() && edgeId > maxEdgeId))
            invalidRawCStoreIOB(moduleId, attrs,
                                "connection id must be a nonnegative integer");
        const json &edge = element.value();
        if(!edge.is_array())
            invalidRawCStoreIOB(moduleId, attrs,
                                "connection " + edgeId + " must be an array");
        if(edge.size() != 6)
            invalidRawCStoreIOB(
                moduleId, attrs,
                "connection " + edgeId + " must contain 6 fields");
        for(int field : {0, 2, 3, 5}){
            if(!edge[field].is_number_integer())
                invalidRawCStoreIOB(
                    moduleId, attrs,
                    "connection " + edgeId +
                    " endpoint/port fields must be integers");
        }
        if(!edge[1].is_string() || !edge[4].is_string())
            invalidRawCStoreIOB(
                moduleId, attrs,
                "connection " + edgeId + " endpoint types must be strings");
    }

    auto controller = attrs.find("io_controller_cfg_id");
    if(controller == attrs.end())
        invalidRawCStoreIOB(moduleId, attrs, "missing io_controller_cfg_id");
    if(!controller->is_object())
        invalidRawCStoreIOB(moduleId, attrs,
                            "expected io_controller_cfg_id object");

    auto useEn = controller->find("UseEn");
    if(useEn == controller->end())
        invalidRawCStoreIOB(moduleId, attrs, "missing UseEn");
    if(!useEn->is_number_integer())
        invalidRawCStoreIOB(moduleId, attrs, "UseEn must be an integer");

    auto numOperands = attrs.find("num_operands");
    if(numOperands == attrs.end() || !numOperands->is_number_integer() ||
       numOperands->get<int>() != 3)
        invalidRawCStoreIOB(moduleId, attrs, "expected num_operands=3");

    for(const char *field : {"BaseAddr", "II", "Latency", "IsStore"}){
        auto value = controller->find(field);
        if(value == controller->end() || !value->is_number_integer())
            invalidRawCStoreIOB(
                moduleId, attrs,
                std::string("expected integer io_controller_cfg_id.") + field);
    }
    auto useAddr = controller->find("UseAddr");
    if(useAddr != controller->end() && !useAddr->is_number_integer())
        invalidRawCStoreIOB(moduleId, attrs, "UseAddr must be an integer");

    auto nestLevels = attrs.find("ag_nest_levels");
    if(nestLevels == attrs.end() || !nestLevels->is_number_integer() ||
       nestLevels->get<int>() < 0)
        invalidRawCStoreIOB(moduleId, attrs,
                            "expected nonnegative integer ag_nest_levels");
    for(int level = 0; level < nestLevels->get<int>(); ++level){
        for(const std::string &field :
            {"Stride" + std::to_string(level),
             "Cycles" + std::to_string(level)}){
            auto value = controller->find(field);
            if(value == controller->end() || !value->is_number_integer())
                invalidRawCStoreIOB(
                    moduleId, attrs,
                    "expected integer io_controller_cfg_id." + field);
        }
    }
}

[[noreturn]] void invalidRawCStoreIOBInstance(
    int nodeId, int moduleId, const std::string &message) {
    std::cout << "Invalid CSTORE IOB instance " << nodeId
              << " (module " << moduleId << "): " << message << std::endl;
    exit(1);
}

void validateRawCStoreIOBInstance(int nodeId, int moduleId,
                                  const json &nodeJson) {
    auto index = nodeJson.find("iob_index");
    if(index == nodeJson.end())
        invalidRawCStoreIOBInstance(nodeId, moduleId, "missing iob_index");
    if(!index->is_number_integer())
        invalidRawCStoreIOBInstance(nodeId, moduleId,
                                    "iob_index must be an integer");
}

} // namespace


ADGIR::ADGIR(std::string filename)
{
    std::ifstream ifs(filename);
    if(!ifs){
        std::cout << "Cannnot open ADG file: " << filename << std::endl;
        exit(1);
    }
    json adgJson;
    ifs >> adgJson;
    _adg = parseADG(adgJson);
}

ADGIR::~ADGIR()
{
    if(_adg){
        delete _adg;
    }
}


// parse ADG json object
ADG* ADGIR::parseADG(json& adgJson, const std::string& owner){
    // std::cout << "Parse ADG..." << std::endl;
    ADG* adg = new ADG();
    adg->setBitWidth(adgJson["data_width"].get<int>());    
    // adg->setNumInputs(adgJson["num_input"].get<int>());
    // adg->setNumOutputs(adgJson["num_output"].get<int>()); 
    if(adgJson.contains("cgra_tile_num")){
        adg->setTileNum(adgJson["cgra_tile_num"].get<int>());
    } 
    if(adgJson.contains("cfg_spad_data_width")){
        adg->setCfgSpadDataWidth(adgJson["cfg_spad_data_width"].get<int>());
    }
    if(adgJson.contains("cfg_data_width")){
        adg->setCfgDataWidth(adgJson["cfg_data_width"].get<int>());
        adg->setCfgAddrWidth(adgJson["cfg_addr_width"].get<int>()); // they are together
        adg->setCfgBlkOffset(adgJson["cfg_blk_offset"].get<int>());
    }
    if(adgJson.contains("iob_ag_nest_levels")){
        adg->setIobAgNestLevels(adgJson["iob_ag_nest_levels"].get<int>());
    }
    if(adgJson.contains("iob_mode_names")){
        for(auto& elem : adgJson["iob_mode_names"].items()){
            int mode = std::stoi(elem.key());
            std::string name = elem.value();
            _iobModeNames[mode] = name;
        }
    }
    if(adgJson.contains("iob_to_spad_banks")){
        for(auto& elem : adgJson["iob_to_spad_banks"].items()){
            int iobId = std::stoi(elem.key());
            std::vector<int> banks;
            for(auto& bank : elem.value()){
                banks.push_back(bank.get<int>());
            }
            adg->setIobToSpadBanks(iobId, banks);
        }
    }
    if(adgJson.contains("iob_spad_bank_size")){
        adg->setIobSpadBankSize(adgJson["iob_spad_bank_size"].get<int>());
    }
    if(adgJson.contains("cfg_spad_size")){
        adg->setCfgSpadSize(adgJson["cfg_spad_size"].get<int>());
    }
    std::map<int, std::pair<ADGNode*, bool>> modules; // // <moduleId, <ADGNode*, used>>
    for(auto& nodeJson : adgJson["sub_modules"]){
        ADGNode* node = parseADGNode(nodeJson);
        modules[node->id()] = std::make_pair(node, false);
    }
    for(auto& nodeJson : adgJson["instances"]){
        ADGNode* node = parseADGNode(nodeJson, modules);
        int nodeId = nodeJson["id"].get<int>();
        if(node){ // not store sub-module of "This" type
            adg->addNode(nodeId, node);
        }else{ // "This" sub-module
            adg->setId(nodeId);
        }
    }
    parseADGEdges(adg, adgJson["connections"], owner);
    postProcess(adg);  
    return adg; 
}


// parse ADGNode from sub-modules json object 
ADGNode* ADGIR::parseADGNode(json& nodeJson){
    // std::cout << "Parse ADG node" << std::endl;
    std::string type = nodeJson["type"].get<std::string>();
    // if(type == "This"){
    //     return nullptr;
    // }
    int nodeId = nodeJson["id"].get<int>();
    ADGNode* adg_node;
    if(type == "GPE" || type == "GIB" || type == "IOB"){
        auto& attrs = nodeJson["attributes"];
        if(type == "IOB" && rawIOBCapableOfCStore(attrs, _iobModeNames))
            validateRawCStoreIOB(nodeId, attrs);
        if(type == "GPE" || type == "IOB"){
            FUNode *fu_node;
            if(type == "GPE"){
                GPENode* node = new GPENode(nodeId);
                // node->setNumRfReg(attrs["num_rf_reg"].get<int>());                
                for(auto& op : attrs["operations"]){
                    node->addOperation(op.get<std::string>());
                }
                if(attrs.contains("affine_ctrl_reg_cfg_id")){
                    auto& iocCfgId = attrs["affine_ctrl_reg_cfg_id"];
                    node->cfgIdMap["InitVal"] = iocCfgId["InitVal"].get<int>();
                    node->cfgIdMap["Cycles"] = iocCfgId["Cycles"].get<int>();
                    node->cfgIdMap["WI"] = iocCfgId["WI"].get<int>();
                    node->cfgIdMap["Latency"] = iocCfgId["Latency"].get<int>();
                    node->cfgIdMap["Repeats"] = iocCfgId["Repeats"].get<int>();
                    node->cfgIdMap["SkipFirst"] = iocCfgId["SkipFirst"].get<int>();
                }
                fu_node = node;
            }else{
                IOBNode* node  = new IOBNode(nodeId);
                auto& iocCfgId = attrs["io_controller_cfg_id"];
                node->cfgIdMap["BaseAddr"] = iocCfgId["BaseAddr"].get<int>();
                node->cfgIdMap["II"] = iocCfgId["II"].get<int>();
                node->cfgIdMap["Latency"] = iocCfgId["Latency"].get<int>();
                node->cfgIdMap["IsStore"] = iocCfgId["IsStore"].get<int>();
                if(iocCfgId.contains("UseAddr")){
                    node->cfgIdMap["UseAddr"] = iocCfgId["UseAddr"].get<int>();
                }
                if(iocCfgId.contains("UseEn")){
                    node->cfgIdMap["UseEn"] = iocCfgId["UseEn"].get<int>();
                }
                int agNestLevels = attrs["ag_nest_levels"].get<int>();
                for(int i = 0; i < agNestLevels; i++){
                    std::string strideName = "Stride" + std::to_string(i);
                    node->cfgIdMap[strideName] = iocCfgId[strideName].get<int>();
                    std::string cyclesName = "Cycles" + std::to_string(i);
                    node->cfgIdMap[cyclesName] = iocCfgId[cyclesName].get<int>();
                }
                if(attrs.contains("operations")){
                    if(attrs["operations"].is_array()){
                        for(auto& op : attrs["operations"]){
                            if(op.is_string()){
                                node->addOperation(op.get<std::string>());
                            }
                        }
                    }
                }else{
                    int iobMode = attrs["iob_mode"].get<int>();
                    std::string modeName = _iobModeNames[iobMode];
                    if(modeName == "FIFO_MODE"){
                        node->addOperation("INPUT");
                        node->addOperation("OUTPUT");
                    }else if(modeName == "SRAM_MODE"){ // SRAM_MODE
                        node->addOperation("INPUT");
                        node->addOperation("OUTPUT");
                        node->addOperation("LOAD");
                        node->addOperation("STORE");
                    }else{ // COND_LS_MODE
                        node->addOperation("INPUT");
                        node->addOperation("OUTPUT");
                        node->addOperation("LOAD");
                        node->addOperation("STORE");
                        node->addOperation("CLOAD");
                        node->addOperation("CSTORE");
                    }
                }
                fu_node = node;
            }
            if(attrs.contains("max_delay")){
                fu_node->setMaxDelay(attrs["max_delay"].get<int>());
            }else{
                fu_node->setMaxDelay(0);
            }
            fu_node->setNumOperands(attrs["num_operands"].get<int>());
            adg_node = fu_node;
        }else if(type == "GIB"){
            GIBNode* node  = new GIBNode(nodeId);            
            adg_node = node;
        }
        // adg_node->setType(type);
        // adg_node->setBitWidth(bitWidth);
        adg_node->setCfgBlkIdx(attrs["cfg_blk_index"].get<int>());
        std::string subADGOwner;
        if(type == "IOB" && dynamic_cast<IOBNode*>(adg_node)->opCapable("CSTORE")){
            subADGOwner = "CSTORE IOB module " + std::to_string(nodeId) +
                          " (iob_index " + std::to_string(attrs["iob_index"].get<int>()) + ")";
        }
        ADG* subADG = parseADG(attrs, subADGOwner); // parse sub-adg
        adg_node->setSubADG(subADG);
        if(attrs.count("configuration")){
            for(auto& elem : attrs["configuration"].items()){
                int subModuleId = std::stoi(elem.key());
                auto& info = elem.value();
                CfgDataLoc cfg;
                cfg.high = info[1].get<int>();
                cfg.low = info[2].get<int>();
                adg_node->addConfigInfo(subModuleId, cfg);
            }
        }
    }else{ // common components: ALU, Muxn, DMR, RDU, Const
        adg_node = new ADGNode(nodeId);        
    } 
    adg_node->setType(type);
    return adg_node;
}


// parse ADGNode from instances json object, 
// modules<moduleId, <ADGNode*, used>>,  
ADGNode* ADGIR::parseADGNode(json& nodeJson, std::map<int, std::pair<ADGNode*, bool>>& modules){
    std::string type = nodeJson["type"].get<std::string>();
    if(type == "This"){
        return nullptr;
    }
    int nodeId = nodeJson["id"].get<int>();
    int moduleId = nodeJson["module_id"].get<int>();
    ADGNode* adg_node;
    ADGNode* module = modules[moduleId].first;
    if(type == "IOB"){
        IOBNode *iobModule = dynamic_cast<IOBNode*>(module);
        if(iobModule && iobModule->opCapable("CSTORE"))
            validateRawCStoreIOBInstance(nodeId, moduleId, nodeJson);
    }
    bool renewNode = modules[moduleId].second; // used, need to re-new ADGNode
    if(type == "GPE" || type == "GIB" || type == "IOB"){                
        if(renewNode){ // re-new ADGNode
            if(type == "GPE"){
                GPENode* node = new GPENode(nodeId);
                *node = *(dynamic_cast<GPENode*>(module));
                adg_node = node;
            }else if(type == "IOB"){
                IOBNode* node  = new IOBNode(nodeId);
                *node = *(dynamic_cast<IOBNode*>(module));
                adg_node = node;
            }else{
                GIBNode* node  = new GIBNode(nodeId);
                *node = *(dynamic_cast<GIBNode*>(module));
                adg_node = node;
            }
            ADG* subADG = new ADG(); 
            *subADG = *(adg_node->subADG()); // COPY Sub-ADG
            adg_node->setSubADG(subADG);
        }else{ // reuse the ADGNode in modules
            adg_node = module;
            modules[moduleId].second = true;
        }
        if(type == "GPE"){
            GPENode *gpe_node = dynamic_cast<GPENode*>(adg_node);
            // gpe_node->setMaxDelay(nodeJson["max_delay"].get<int>());
        }else if(type == "IOB"){
            IOBNode *iob_node = dynamic_cast<IOBNode*>(adg_node);
            iob_node->setIndex(nodeJson["iob_index"].get<int>());
            // iob_node->setMaxDelay(nodeJson["max_delay"].get<int>());
        }else{
            GIBNode *gib_node = dynamic_cast<GIBNode*>(adg_node);
            gib_node->setTrackReged(nodeJson["track_reged"].get<bool>());
        }
        adg_node->setCfgBlkIdx(nodeJson["cfg_blk_index"].get<int>());  
        adg_node->setX(nodeJson["x"].get<int>());     
        adg_node->setY(nodeJson["y"].get<int>());  
        if(nodeJson.contains("tile"))
            adg_node->setTile(nodeJson["tile"].get<int>());     
    }else{ // common components: ALU, Muxn, DMR, RDU, Const
        if(renewNode){ // re-new ADGNode
            adg_node = new ADGNode(nodeId);
            *adg_node = *module;  
        }else{ // reuse the ADGNode in modules
            adg_node = module;
            modules[moduleId].second = true;
        }      
    } 
    adg_node->setId(nodeId);
    adg_node->setName(type+std::to_string(nodeId));
    // adg_node->setType(type);
    return adg_node;
}


// parse ADGEdge json object
void ADGIR::parseADGEdges(ADG* adg, json& edgeJson, const std::string& owner){
    // std::cout << "Parse ADG Edge" << std::endl;
    for(auto& elem : edgeJson.items()){
        int edgeId = std::stoi(elem.key());
        auto& edge = elem.value();
        int srcId = edge[0].get<int>();
        // std::string srcType = edge[1].get<std::string>();
        int srcPort = edge[2].get<int>();
        int dstId = edge[3].get<int>();
        // std::string dstType = edge[4].get<std::string>();
        int dstPort = edge[5].get<int>();
        bool missingSrc = srcId != adg->id() && !adg->node(srcId);
        bool missingDst = dstId != adg->id() && !adg->node(dstId);
        if(missingSrc || missingDst){
            const std::string endpoint = missingSrc ? "source" : "destination";
            const int endpointId = missingSrc ? srcId : dstId;
            if(!owner.empty()){
                std::cout << "Invalid " << owner << ": malformed endpoint "
                          << endpoint << " node " << endpointId
                          << " in edge " << edgeId << std::endl;
            }else{
                std::cout << "Invalid ADG edge " << edgeId
                          << ": malformed endpoint " << endpoint
                          << " node " << endpointId << std::endl;
            }
            exit(1);
        }
        ADGEdge* adg_edge = new ADGEdge(srcId, dstId);
        adg_edge->setId(edgeId);
        adg_edge->setSrcId(srcId);
        adg_edge->setDstId(dstId);
        adg_edge->setSrcPortIdx(srcPort);
        adg_edge->setDstPortIdx(dstPort);
        adg->addEdge(edgeId, adg_edge);
    }
}


// analyze the connections among the internal sub-modules for GPENode, fill _operandInputs 
void ADGIR::analyzeIntraConnect(GPENode* node){
    ADG* subAdg = node->subADG();
    for(auto& elem : subAdg->inputs()){
        auto input = elem.second.begin(); // one input only connected to one sub-module
        ADGNode* subNode = subAdg->node(input->first);
        int opeIdx = input->second; // operand index
        while (subNode->type() != "ALU"){
            if(subNode->outputs().size() == 1){ // only one output
                opeIdx = 0;
            }
            auto out = subNode->output(opeIdx).begin(); 
            subNode = subAdg->node(out->first);
            opeIdx = out->second;
        }
        // opeIdx is ALU operand index now
        node->addOperandInputs(opeIdx, elem.first);
    }
}


// analyze the connections among the internal sub-modules for IOBNode, fill _operandInputs 
void ADGIR::analyzeIntraConnect(IOBNode* node){
    ADG* subAdg = node->subADG();
    for(auto& elem : subAdg->inputs()){
        if(elem.second.empty()){
            continue;
        }
        std::vector<std::pair<int, int>> pending(elem.second.begin(), elem.second.end());
        std::set<std::pair<int, int>> visited;
        while(!pending.empty()){
            auto current = pending.back();
            pending.pop_back();
            ADGNode* subNode = subAdg->node(current.first);
            int opeIdx = current.second;
            if(!subNode || !visited.emplace(current).second){
                continue;
            }
            if(subNode->type() == "IOController"){
                if(opeIdx >= 0 && opeIdx < node->numOperands()){
                    node->addOperandInputs(opeIdx, elem.first);
                }
                continue;
            }
            if(subNode->outputs().size() == 1){ // only one output
                opeIdx = 0;
            }
            auto outputs = subNode->output(opeIdx);
            for(auto& output : outputs){
                pending.push_back(output);
            }
        }
    }
}

static void validateCStoreIOB(IOBNode* node){
    if(!node->opCapable("CSTORE")){
        return;
    }
    const std::string prefix = "Invalid CSTORE IOB " + node->name() +
                               " (id=" + std::to_string(node->id()) + "): ";
    if(node->numOperands() != 3){
        std::cout << prefix << "expected num_operands=3" << std::endl;
        exit(1);
    }
    if(!node->cfgIdMap.count("UseEn")){
        std::cout << prefix << "missing UseEn" << std::endl;
        exit(1);
    }
    for(int operand = 0; operand < 3; ++operand){
        if(node->operandInputs(operand).empty()){
            std::cout << prefix << "operand " << operand
                      << " has no physical input path" << std::endl;
            exit(1);
        }
    }
    for(int first = 0; first < 3; ++first){
        for(int second = first + 1; second < 3; ++second){
            for(int input : node->operandInputs(first)){
                if(node->operandInputs(second).count(input)){
                    std::cout << prefix << "physical input sets overlap" << std::endl;
                    exit(1);
                }
            }
        }
    }
}


// analyze the connections among the internal sub-modules for GIBNode
// fill _out2ins, _in2outs 
void ADGIR::analyzeIntraConnect(GIBNode* node){
    ADG* subAdg = node->subADG();
    for(auto& ielem : subAdg->inputs()){
        int inPort = ielem.first;
        for(auto& subNode : ielem.second){
            int outPort;
            if(subNode.first == subAdg->id()){ // input directly connected to output
                outPort = subNode.second;
            } else {
                ADGNode* subNodePtr = subAdg->node(subNode.first);
                outPort = subNodePtr->output(0).begin()->second; // only one layer of Muxn
            }
            node->addIn2outs(inPort, outPort);
            node->addOut2ins(outPort, inPort);
        }
    }
}



// analyze if there are registers in the output ports of GIB
// fill _outReged
void ADGIR::analyzeOutReg(ADG* adg, GIBNode* node){
    for(auto& elem : node->outputs()){
        int id = elem.second.begin()->first; // GIB output port only connected to one node
        if(node->trackReged() && (adg->node(id)->type() == "GIB")){ // this edge is track
            node->setOutReged(elem.first, true);
        }else{
            node->setOutReged(elem.first, false);
        }
    }
}


// post-process the ADG nodes
void ADGIR::postProcess(ADG* adg){
    int numGpeNodes = 0;
    int numIobNodes = 0;
    for(auto& node : adg->nodes()){
        auto nodePtr = node.second;
        if(nodePtr->type() == "GPE"){
            numGpeNodes++;
            analyzeIntraConnect(dynamic_cast<GPENode*>(nodePtr));
        } else if(nodePtr->type() == "IOB"){
            numIobNodes++;
            IOBNode* iob = dynamic_cast<IOBNode*>(nodePtr);
            analyzeIntraConnect(iob);
            validateCStoreIOB(iob);
        } else if(nodePtr->type() == "GIB"){
            analyzeIntraConnect(dynamic_cast<GIBNode*>(nodePtr));
            analyzeOutReg(adg, dynamic_cast<GIBNode*>(nodePtr));
        }
    }
    adg->setNumGpeNodes(numGpeNodes);
    adg->setNumIobNodes(numIobNodes);
}
