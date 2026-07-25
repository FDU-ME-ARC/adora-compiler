
#include "ir/adg_ir.h"


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
    _adg->setMaxLUTInput(_maxLUTInput);
}

ADGIR::~ADGIR()
{
    if(_adg){
        delete _adg;
    }
}


// parse ADG json object
ADG* ADGIR::parseADG(json& adgJson){
    // std::cout << "Parse ADG..." << std::endl;
    ADG* adg = new ADG();
    adg->setBitWidth(adgJson["data_width"].get<int>());    
    if(adgJson.contains("fine_grained") && adgJson["fine_grained"].get<bool>()){
        adg->addBitWidth(1);
    }
    if(adgJson.contains("fg_cfg_base_block")){
        adg->setFgCfgBaseBlock(adgJson["fg_cfg_base_block"].get<int>());
    }
    if(adgJson.contains("fg_cfg_block_count")){
        adg->setFgCfgBlockCount(adgJson["fg_cfg_block_count"].get<int>());
    }
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
    parseADGEdges(adg, adgJson["connections"]);
    if(adgJson.contains("fine_grained_networks")){
        parseFineGrainedNetworks(adg, adgJson["fine_grained_networks"]);
    }
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
    if(type == "GPE" || type == "GIB" || type == "CGGIB" || type == "FGGIB" || type == "IOB"){
        auto& attrs = nodeJson["attributes"];
        if(type == "GPE" || type == "IOB"){
            FUNode *fu_node;
            if(type == "GPE"){
                GPENode* node = new GPENode(nodeId);
                int numInputLUT = attrs.value("num_input_lut", 0);
                node->setNumInputLUT(numInputLUT);
                node->sethasLUT(numInputLUT > 0);
                _maxLUTInput = std::max(_maxLUTInput, numInputLUT);
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
                if(iocCfgId.contains("UsePredicate")){
                    node->cfgIdMap["UsePredicate"] = iocCfgId["UsePredicate"].get<int>();
                }
                if(attrs.contains("iob_immediate_cfg_id")){
                    auto& immCfgId = attrs["iob_immediate_cfg_id"];
                    node->cfgIdMap["UseImm"] = immCfgId["UseImm"].get<int>();
                    node->cfgIdMap["ImmOperand"] =
                        immCfgId["ImmOperand"].get<int>();
                    node->cfgIdMap["ImmValue"] =
                        immCfgId["ImmValue"].get<int>();
                }
                int agNestLevels = attrs["ag_nest_levels"].get<int>();
                for(int i = 0; i < agNestLevels; i++){
                    std::string strideName = "Stride" + std::to_string(i);
                    node->cfgIdMap[strideName] = iocCfgId[strideName].get<int>();
                    std::string cyclesName = "Cycles" + std::to_string(i);
                    node->cfgIdMap[cyclesName] = iocCfgId[cyclesName].get<int>();
                }
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
                if(iocCfgId.contains("UsePredicate")){
                    node->addOperation("CINPUT");
                    node->addOperation("COUTPUT");
                    node->addOperation("CLOAD");
                    node->addOperation("CSTORE");
                }
                fu_node = node;
            }
            int cgWidth = attrs.value("data_width", 32);
            int cgMaxDelay = attrs.value("max_delay_cg", attrs.value("max_delay", 0));
            int fgMaxDelay = attrs.value("max_delay_fg", cgMaxDelay);
            int cgOperands = attrs.value("num_operand_cg", attrs.value("num_operands", 0));
            int fgInputs = attrs.value("num_input_fg", 0);
            int fgOperands = attrs.value("num_operand_fg", 0);
            if(fgOperands == 0 && fgInputs > 0){
                if(type == "GPE"){
                    auto* gpe = dynamic_cast<GPENode*>(fu_node);
                    fgOperands = std::max(2, gpe->numInputLUT());
                }else{
                    fgOperands = 1;
                }
            }
            fu_node->setMaxDelay(cgMaxDelay);
            fu_node->setNumOperands(cgOperands);
            fu_node->setMaxDelay(cgWidth, cgMaxDelay);
            fu_node->setNumOperands(cgWidth, cgOperands);
            fu_node->setBitWidth(cgWidth);
            if(fgInputs > 0 || attrs.value("num_output_fg", 0) > 0){
                fu_node->addBitWidth(1);
                fu_node->setMaxDelay(1, fgMaxDelay);
                fu_node->setNumOperands(1, fgOperands);
                int fgInputPort = 0;
                if(attrs.contains("num_input_per_fg")){
                    int operand = 0;
                    for(auto& countJson : attrs["num_input_per_fg"]){
                        int count = countJson.get<int>();
                        for(int i = 0; i < count; ++i){
                            fu_node->addOperandInput(1, operand, fgInputPort++);
                        }
                        ++operand;
                    }
                }else if(fgOperands > 0){
                    int inputsPerOperand = fgInputs / fgOperands;
                    for(int operand = 0; operand < fgOperands; ++operand){
                        for(int i = 0; i < inputsPerOperand; ++i){
                            fu_node->addOperandInput(1, operand, fgInputPort++);
                        }
                    }
                }
            }
            if(attrs.contains("fine_grained_configuration_ranges")){
                auto& ranges = attrs["fine_grained_configuration_ranges"];
                FineGrainedCfgInfo fgCfg;
                fgCfg.delay.low = ranges.value("delay_low", -1);
                fgCfg.delay.high = ranges.value("delay_high", -1);
                fgCfg.mux.low = ranges.value("mux_low", -1);
                fgCfg.mux.high = ranges.value("mux_high", -1);
                fgCfg.lut.low = ranges.value("lut_low", -1);
                fgCfg.lut.high = ranges.value("lut_high", -1);
                if(ranges.contains("mux_widths")){
                    for(auto& width : ranges["mux_widths"]){
                        fgCfg.muxWidths.push_back(width.get<int>());
                    }
                }
                fgCfg.valid = true;
                fu_node->setFineGrainedCfg(fgCfg);
            }
            adg_node = fu_node;
        }else if(type == "GIB" || type == "CGGIB" || type == "FGGIB"){
            GIBNode* node  = new GIBNode(nodeId);            
            adg_node = node;
            adg_node->setBitWidth(attrs.value("data_width", 32));
        }
        // adg_node->setType(type);
        // adg_node->setBitWidth(bitWidth);
        adg_node->setCfgBlkIdx(attrs["cfg_blk_index"].get<int>());
        ADG* subADG = parseADG(attrs); // parse sub-adg
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
void ADGIR::parseADGEdges(ADG* adg, json& edgeJson){
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
        int bitWidth = edge.size() > 6 ? edge[6].get<int>() : adg->bitWidth();
        ADGEdge* adg_edge = new ADGEdge(srcId, dstId);
        adg_edge->setId(edgeId);
        adg_edge->setSrcId(srcId);
        adg_edge->setDstId(dstId);
        adg_edge->setSrcPortIdx(srcPort);
        adg_edge->setDstPortIdx(dstPort);
        adg_edge->setBitWidth(bitWidth);
        adg->addEdge(edgeId, adg_edge);
    }
}

void ADGIR::parseFineGrainedNetworks(ADG* adg, json& networksJson){
    if(networksJson.empty()) return;

    int nextNodeId = 1;
    int nextEdgeId = 0;
    for(auto& elem : adg->nodes()) nextNodeId = std::max(nextNodeId, elem.first + 1);
    for(auto& elem : adg->edges()) nextEdgeId = std::max(nextEdgeId, elem.first + 1);

    std::map<int, std::vector<int>> tileGpes;
    std::map<int, std::vector<std::pair<int, int>>> tileIobs;
    for(auto& elem : adg->nodes()){
        ADGNode* node = elem.second;
        if(node->type() == "GPE"){
            tileGpes[node->tile()].push_back(elem.first);
        }else if(node->type() == "IOB"){
            auto* iob = dynamic_cast<IOBNode*>(node);
            tileIobs[node->tile()].push_back({iob->index(), elem.first});
        }
    }
    for(auto& elem : tileGpes){
        std::sort(elem.second.begin(), elem.second.end(), [adg](int lhs, int rhs){
            ADGNode* a = adg->node(lhs);
            ADGNode* b = adg->node(rhs);
            return std::make_tuple(a->x(), a->y(), lhs) <
                   std::make_tuple(b->x(), b->y(), rhs);
        });
    }
    for(auto& elem : tileIobs) std::sort(elem.second.begin(), elem.second.end());

    std::map<std::pair<int, int>, int> fgGibIds;
    for(auto& networkItem : networksJson.items()){
        int tile = std::stoi(networkItem.key());
        json& network = networkItem.value();
        if(!network.contains("gibs") || !network.contains("connections")){
            std::cerr << "Warning: fine_grained_networks[" << tile
                      << "] has no explicit connections; regenerate the ADG "
                         "with the FG topology exporter." << std::endl;
            continue;
        }
        for(auto& gibItem : network["gibs"].items()){
            int localId = std::stoi(gibItem.key());
            json wrapper = {
                {"id", nextNodeId},
                {"type", "GIB"},
                {"attributes", gibItem.value()}
            };
            ADGNode* node = parseADGNode(wrapper);
            node->setId(nextNodeId);
            node->setName("FGGIB" + std::to_string(nextNodeId));
            // Keep the legacy node type for existing AuFORA mapper paths; the
            // one-bit edge/node width distinguishes this as an FGGIB.
            node->setType("GIB");
            node->setTile(tile);
            auto& attrs = gibItem.value();
            if(attrs.contains("x")) node->setX(attrs["x"].get<int>());
            if(attrs.contains("y")) node->setY(attrs["y"].get<int>());
            dynamic_cast<GIBNode*>(node)->setTrackReged(attrs.value("track_reged", false));
            adg->addNode(nextNodeId, node);
            fgGibIds[{tile, localId}] = nextNodeId++;
        }
    }

    auto resolveNode = [&](int tile, const std::string& type, int localId) -> int {
        if(type == "FGGIB"){
            auto it = fgGibIds.find({tile, localId});
            return it == fgGibIds.end() ? -1 : it->second;
        }
        if(type == "GPE"){
            auto it = tileGpes.find(tile);
            return it != tileGpes.end() && localId >= 0 &&
                   localId < static_cast<int>(it->second.size())
                ? it->second[localId] : -1;
        }
        if(type == "IOB"){
            auto it = tileIobs.find(tile);
            return it != tileIobs.end() && localId >= 0 &&
                   localId < static_cast<int>(it->second.size())
                ? it->second[localId].second : -1;
        }
        return -1;
    };

    for(auto& networkItem : networksJson.items()){
        int tile = std::stoi(networkItem.key());
        if(!networkItem.value().contains("connections")) continue;
        json& connections = networkItem.value()["connections"];
        for(auto& edgeItem : connections.items()){
            auto& edge = edgeItem.value();
            if(edge.size() != 7){
                std::cerr << "Invalid FG edge in tile " << tile << std::endl;
                exit(1);
            }
            std::string srcType = edge[0].get<std::string>();
            int srcId = resolveNode(tile, srcType, edge[1].get<int>());
            int srcPort = edge[2].get<int>();
            std::string dstType = edge[3].get<std::string>();
            int dstId = resolveNode(tile, dstType, edge[4].get<int>());
            int dstPort = edge[5].get<int>();
            int bitWidth = edge[6].get<int>();
            if(srcId < 0 || dstId < 0 || bitWidth != 1){
                std::cerr << "Cannot resolve FG edge " << edge.dump()
                          << " in tile " << tile << std::endl;
                exit(1);
            }
            ADGEdge* adgEdge = new ADGEdge(srcId, dstId);
            adgEdge->setId(nextEdgeId);
            adgEdge->setEdge(bitWidth, srcId, srcPort, dstId, dstPort);
            adg->addEdge(nextEdgeId++, adgEdge);
        }
    }
    adg->addBitWidth(1);
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
        node->addOperandInput(node->bitWidth(), opeIdx, elem.first);
    }
}


// analyze the connections among the internal sub-modules for IOBNode, fill _operandInputs 
void ADGIR::analyzeIntraConnect(IOBNode* node){
    ADG* subAdg = node->subADG();
    for(auto& elem : subAdg->inputs()){
        auto input = elem.second.begin(); // one input only connected to one sub-module
        ADGNode* subNode = subAdg->node(input->first);
        int opeIdx = input->second; // operand index
        while (subNode->type() != "IOController"){
            if(subNode->outputs().size() == 1){ // only one output
                opeIdx = 0;
            }
            auto out = subNode->output(opeIdx).begin(); 
            subNode = subAdg->node(out->first);
            opeIdx = out->second;
        }
        // opeIdx is ALU operand index now
        node->addOperandInputs(opeIdx, elem.first);
        node->addOperandInput(node->bitWidth(), opeIdx, elem.first);
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
            analyzeIntraConnect(dynamic_cast<IOBNode*>(nodePtr));
        } else if(nodePtr->type() == "GIB"){
            analyzeIntraConnect(dynamic_cast<GIBNode*>(nodePtr));
            analyzeOutReg(adg, dynamic_cast<GIBNode*>(nodePtr));
        }
    }
    adg->setNumGpeNodes(numGpeNodes);
    adg->setNumIobNodes(numIobNodes);
}
