
#include "mapper/configuration.h"

#include <limits>

void Configuration::addCfgData(std::map<int, CfgData> &cfg, const CfgDataLoc &loc, uint32_t data){
    CfgData loc_data(loc.high - loc.low + 1, data);
    cfg[loc.low] = loc_data;
}

void Configuration::addCfgData(std::map<int, CfgData> &cfg, const CfgDataLoc &loc, uint64_t data){
    int len = loc.high - loc.low + 1;
    CfgData loc_data(len);
    uint64_t val = data & (((uint64_t)1 << len) - 1);
    while(len > 0){
        loc_data.data.push_back(uint32_t(val&0xffffffff));
        len -= 32;
        val >>= 32;
    }
    cfg[loc.low] = loc_data;
}

void Configuration::addCfgData(std::map<int, CfgData> &cfg, const CfgDataLoc &loc, const std::vector<uint32_t> &data){
    CfgData loc_data(loc.high - loc.low + 1, data);
    cfg[loc.low] = loc_data;
}

void Configuration::addIobDelayCfg(std::map<int, CfgData> &cfg, IOBNode *node,
                                   ADGNode *delayNode,
                                   const std::map<int, int> &delayUsed){
    if(delayUsed.empty()){
        return;
    }
    const std::string prefix = "Invalid IOB configuration " + node->name() +
                               " (id=" + std::to_string(node->id()) + "): ";
    if(!delayNode ||
       (delayNode->type() != "DelayPipe" && delayNode->type() != "RDU")){
        std::cout << prefix << "expected DelayPipe after input mux" << std::endl;
        exit(1);
    }
    auto config = node->configInfo().find(delayNode->id());
    if(config == node->configInfo().end()){
        std::cout << prefix << "missing DelayPipe configuration for module "
                  << delayNode->id() << std::endl;
        exit(1);
    }
    const CfgDataLoc &location = config->second;
    const int width = location.high - location.low + 1;
    const int operands = node->numOperands();
    if(width <= 0 || width > 32 || operands <= 0 || width % operands != 0){
        std::cout << prefix << "DelayPipe configuration width " << width
                  << " is incompatible with " << operands << " operands"
                  << std::endl;
        exit(1);
    }
    const int laneWidth = width / operands;
    const uint32_t laneMask = laneWidth == 32
                                  ? std::numeric_limits<uint32_t>::max()
                                  : (uint32_t(1) << laneWidth) - 1;
    uint32_t packed = 0;
    for(const auto &entry : delayUsed){
        const int lane = entry.first;
        const int delay = entry.second;
        if(lane < 0 || lane >= operands || delay < 0 ||
           uint64_t(delay) > laneMask){
            std::cout << prefix << "DelayPipe lane " << lane
                      << " cannot encode delay " << delay << std::endl;
            exit(1);
        }
        packed |= uint32_t(delay) << (laneWidth * lane);
    }
    addCfgData(cfg, location, packed);
}


std::map<int, int> Configuration::addAdditionalDelayForMERGEOp(DFGNode* dfgNode,std::map<int, int>& delayUsed){
    if(dfgNode->operation() == "MERGE4" || dfgNode->operation() == "INTLV4"){
        delayUsed[1] += 1;
        delayUsed[2] += 2;
        delayUsed[3] += 3;
    }
    else if(dfgNode->operation() == "MERGE3" || dfgNode->operation() == "INTLV3"){
        delayUsed[1] += 1;
        delayUsed[2] += 2;
    }  
    else if(dfgNode->operation() == "MERGE2" || dfgNode->operation() == "INTLV2"){
        delayUsed[1] += 1;
        delayUsed[2] += 2;
    }  
    // int totalDelay = 0;
    // for(auto elem : delayUsed){
    //     totalDelay += elem.second;
    // }
    // assert(totalDelay <= _mapping->mappedNode(dfgNode)->id());

    return delayUsed;
}

int Configuration::addAdditionalLatencyForMERGEOp(DFGNode* dfgNode, int latency){
    if(dfgNode->operation() == "MERGE4" || dfgNode->operation() == "INTLV4"){
        return latency + 3;
    }
    else if(dfgNode->operation() == "MERGE3" || dfgNode->operation() == "INTLV3"){
        return latency + 2;
    }  
    else if(dfgNode->operation() == "MERGE2" || dfgNode->operation() == "INTLV2"){
        return latency + 1;
    }  
    // int totalDelay = 0;
    // for(auto elem : delayUsed){
    //     totalDelay += elem.second;
    // }
    // assert(totalDelay <= _mapping->mappedNode(dfgNode)->id());

    return latency;
}
// int Configuration::addAdditionalLatencyForOpsFolloingMERGE(DFGNode* dfgNode, int latency){
//     if(dfgNode->operation() == "OUTPUT"){
//         if(dfgNode->inputEdges().size() == 1){
//             int srcnodeid = _mapping->getDFG()->edge(dfgNode->inputEdge(0))->srcId();
//             return addAdditionalLatencyForMERGEOp(_mapping->getDFG()->node(srcnodeid), latency);
//         }
//     }
//     return latency;
// }

std::set<int> Configuration::getConfiguredTiles(){
    std::set<int> results;
    for(auto& elem : _mapping->getADG()->nodes()){
        ADGNode* adgnode = elem.second;
        if(_mapping->isMapped(adgnode)){
            results.insert(adgnode->tile());
        }
    }

    return results;
}

// get config data for GPE, return<LSB-location, CfgData>
std::map<int, CfgData> Configuration::getGpeCfgData(GPENode* node){
    if(!_mapping->isMapped(node)){
        return {};
    }
    int adgNodeId = node->id();
    ADG* subAdg = node->subADG();
    auto& adgNodeAttr = _mapping->adgNodeAttr(adgNodeId);
    DFGNode* dfgNode = adgNodeAttr.dfgNode;
    dfgNode->printDfgNode(); /// @jhlou 
    std::map<int, CfgData> cfg;
    // operation
    int opc = Operations::OPC(dfgNode->operation());
    int aluId = -1;
    int rduId;
    // bool flag = false;
    std::set<int> usedOperands;
    std::map<int, int> delayUsed;

    VariableConfig* varconfig = nullptr;
    if(IsADGNodeConfigVariable(node)){
        varconfig = getConfigVariable(node);
        varconfig->print();
    }

    for(auto& elem : dfgNode->inputEdges()){
        int eid = elem.second;
        auto& edgeAttr = _mapping->dfgEdgeAttr(eid);
        int inputIdx = edgeAttr.edgeLinks.rbegin()->srcPort; // last edgeLInk, dst port
        auto muxPair = subAdg->input(inputIdx).begin(); // one input only connected to one Mux
        int muxId = muxPair->first;
        int muxCfgData = muxPair->second;
        auto mux = subAdg->node(muxId);
        auto rduPair = mux->output(0).begin();
        rduId = rduPair->first; 
        int rduPort = rduPair->second;
        auto rdu = subAdg->node(rduId);   
        delayUsed[rduPort] = edgeAttr.delay; // delay cycles used by this port

        auto aluPair = rdu->output(rduPort).begin();  // RDU has the same input/output index
        aluId = aluPair->first;
        // int aluPort = aluPair->second;
        // usedOperands.emplace(aluPort); // operand index
        usedOperands.emplace(rduPort); // operand index
        addCfgData(cfg, node->configInfo(muxId), (uint32_t)muxCfgData);
        // CfgDataLoc muxCfgLoc = node->configInfo(muxId);
        // CfgData muxCfg((muxCfgLoc.high - muxCfgLoc.low + 1), (uint32_t)muxCfgData);
        // cfg[muxCfgLoc.low] = muxCfg;    
    }

    //// @jhlou: for merge op, add additional delay 
    delayUsed = addAdditionalDelayForMERGEOp(dfgNode, delayUsed);

    if(aluId == -1){ // in case that some node has no input
        auto muxPair = subAdg->input(0).begin(); // one input only connected to one Mux
        int muxId = muxPair->first;
        auto mux = subAdg->node(muxId);
        rduId = mux->output(0).begin()->first; 
        auto rdu = subAdg->node(rduId);   
        auto aluPair = rdu->output(0).begin();   
        aluId = aluPair->first;
    }else{
        // RDU
        CfgDataLoc rduCfgLoc = node->configInfo(rduId);
        uint32_t delayCfg = 0;
        int eachDelayWidth = (rduCfgLoc.high - rduCfgLoc.low + 1) / node->numOperands();
        for(auto& elem : delayUsed){
            delayCfg |= elem.second << (eachDelayWidth * elem.first);
        }
        addCfgData(cfg, rduCfgLoc, (uint32_t)delayCfg);
        // CfgData rduCfg(rduCfgLoc.high - rduCfgLoc.low + 1, delayCfg);
        // cfg[rduCfgLoc.low] = rduCfg;  
    }
    addCfgData(cfg, node->configInfo(aluId), (uint32_t)opc);
    // CfgDataLoc aluCfgLoc = node->configInfo(aluId);
    // CfgData aluCfg(aluCfgLoc.high - aluCfgLoc.low + 1, (uint32_t)opc);
    // cfg[aluCfgLoc.low] = aluCfg;
    
    // accumulative node
    if(dfgNode->accumulative()){
        node->print();
        //// Init Value
        int initValId = node->cfgIdMap["InitVal"];
        addCfgData(cfg, node->configInfo(initValId), (uint64_t)(dfgNode->initVal()));
        // CfgDataLoc InitValCfgLoc = node->configInfo(initValId);                
        // int len = InitValCfgLoc.high - InitValCfgLoc.low + 1;
        // CfgData InitValCfg(len);
        // uint64_t InitVal = dfgNode->initVal() & (((uint64_t)1 << len) - 1);
        // while(len > 0){
        //     InitValCfg.data.push_back(uint32_t(InitVal&0xffffffff));
        //     len -= 32;
        //     InitVal >> 32;
        // }
        // cfg[InitValCfgLoc.low] = InitValCfg;
        if(varconfig != nullptr && !IsConstStrExpr(varconfig->initVal)){ /// Init
            const uint32_t key = node->configInfo(initValId).low;
            std::pair<uint32_t, std::string> value = 
                        std::make_pair((uint32_t)node->configInfo(initValId).high - node->configInfo(initValId).low + 1, varconfig->initVal);
            varconfig->LSBToLenExpr[key] = value;
        }

        /// II
        int II = _mapping->II();
        int WI = dfgNode->isLoopIndexAcc() ? 1 : dfgNode->interval() * II;
        int wiId = node->cfgIdMap["WI"];
        addCfgData(cfg, node->configInfo(wiId), (uint32_t)WI);
        // CfgDataLoc wiCfgLoc = node->configInfo(wiId);
        // int wiCfgLen = wiCfgLoc.high - wiCfgLoc.low + 1;
        // CfgData wiCfg(wiCfgLen);
        // wiCfg.data.push_back((uint32_t)WI); 
        // cfg[wiCfgLoc.low] = wiCfg;

        auto& dfgNodeAttr = _mapping->dfgNodeAttr(dfgNode->id());
        int latency = dfgNode->isLoopIndexAcc() ? 0 :
            dfgNodeAttr.lat - dfgNode->opLatency();
        int latencyId = node->cfgIdMap["Latency"];

        //// @jhlou: for nodes following merge op, add additional latency(acr counting 3 more cycles) 
        if(!dfgNode->isLoopIndexAcc()){
            latency = addAdditionalLatencyForMERGEOp(dfgNode, latency);
        }

        addCfgData(cfg, node->configInfo(latencyId), (uint32_t)latency);
        // CfgDataLoc latencyCfgLoc = node->configInfo(latencyId);
        // int latencyCfgLen = latencyCfgLoc.high - latencyCfgLoc.low + 1;
        // CfgData latencyCfg(latencyCfgLen);
        // latencyCfg.data.push_back((uint32_t)latency); 
        // cfg[latencyCfgLoc.low] = latencyCfg;

        int cycles = dfgNode->cycles();
        int cyclesId = node->cfgIdMap["Cycles"];
        addCfgData(cfg, node->configInfo(cyclesId), (uint32_t)cycles);
        // CfgDataLoc cyclesCfgLoc = node->configInfo(cyclesId);
        // int cyclesCfgLen = cyclesCfgLoc.high - cyclesCfgLoc.low + 1;
        // CfgData cyclesCfg(cyclesCfgLen);
        // cyclesCfg.data.push_back((uint32_t)cycles); 
        // cfg[cyclesCfgLoc.low] = cyclesCfg;
        if(varconfig != nullptr && !IsConstStrExpr(varconfig->cycles)){ /// cycles
            const uint32_t key = node->configInfo(cyclesId).low;
            std::pair<uint32_t, std::string> value = 
                std::make_pair((uint32_t)node->configInfo(cyclesId).high - node->configInfo(cyclesId).low + 1, varconfig->cycles);
            varconfig->LSBToLenExpr[key] = value;
        }

        int repeats = dfgNode->isLoopIndexAcc() ? 1 : dfgNode->repeats();
        int repeatsId = node->cfgIdMap["Repeats"];
        addCfgData(cfg, node->configInfo(repeatsId), (uint32_t)repeats);
        // CfgDataLoc repeatsCfgLoc = node->configInfo(repeatsId);
        // int repeatsCfgLen = repeatsCfgLoc.high - repeatsCfgLoc.low + 1;
        // CfgData repeatsCfg(repeatsCfgLen);
        // repeatsCfg.data.push_back((uint32_t)repeats); 
        // cfg[repeatsCfgLoc.low] = repeatsCfg;
        if(varconfig != nullptr && !IsConstStrExpr(varconfig->repeats)){ /// cycles
            const uint32_t key = node->configInfo(repeatsId).low;
            std::pair<uint32_t, std::string> value = 
                std::make_pair((uint32_t)node->configInfo(repeatsId).high - node->configInfo(repeatsId).low + 1, varconfig->repeats);
            varconfig->LSBToLenExpr[key] = value;
        }

        bool skipfisrt = dfgNode->isLoopIndexAcc() || !dfgNode->isAccFirst();
        int skipfisrtId = node->cfgIdMap["SkipFirst"];
        addCfgData(cfg, node->configInfo(skipfisrtId), (uint32_t)skipfisrt);
        // CfgDataLoc skipfisrtCfgLoc = node->configInfo(skipfisrtId);
        // int skipfisrtCfgLen = skipfisrtCfgLoc.high - skipfisrtCfgLoc.low + 1;
        // CfgData skipfisrtCfg(skipfisrtCfgLen);
        // skipfisrtCfg.data.push_back((uint32_t)skipfisrt); 
        // cfg[skipfisrtCfgLoc.low] = skipfisrtCfg;
    }
    // Constant
    if(dfgNode->hasImm()){
        // find unused operand
        int i = 0;
        for(; i < dfgNode->numInputs(); i++){
            if(!usedOperands.count(i)){ 
                break;
            }
        }
        assert(i < dfgNode->numInputs());
        // auto alu = subAdg->node(aluId); 
        auto rdu = subAdg->node(rduId); // used default delay 
        int muxId = rdu->input(i).first;
        for(auto& elem : subAdg->node(muxId)->inputs()){
            int id = elem.second.first;
            if(id == subAdg->id()) continue;
            if(subAdg->node(id)->type() == "Const"){
                addCfgData(cfg, node->configInfo(muxId), (uint32_t)elem.first);
                addCfgData(cfg, node->configInfo(id), (uint64_t)(dfgNode->imm()));
                // CfgDataLoc muxCfgLoc = node->configInfo(muxId);
                // CfgData muxCfg(muxCfgLoc.high - muxCfgLoc.low + 1, (uint32_t)elem.first);
                // cfg[muxCfgLoc.low] = muxCfg;
                // CfgDataLoc constCfgLoc = node->configInfo(id);                
                // int len = constCfgLoc.high - constCfgLoc.low + 1;
                // CfgData constCfg(len);
                // uint64_t imm = dfgNode->imm() & (((uint64_t)1 << len) - 1);
                // while(len > 0){
                //     constCfg.data.push_back(uint32_t(imm&0xffffffff));
                //     len -= 32;
                //     imm >> 32;
                // }
                // cfg[constCfgLoc.low] = constCfg;
                break;
            }
        }  
    }
    return cfg;
}

// get config data for IOB, return<LSB-location, CfgData>
std::map<int, CfgData> Configuration::getIobCfgData(IOBNode* node){
    if(!_mapping->isMapped(node)){
        return {};
    }
    int adgNodeId = node->id();
    ADG* subAdg = node->subADG();
    auto& adgNodeAttr = _mapping->adgNodeAttr(adgNodeId);
    DFGNode* dfgNode = adgNodeAttr.dfgNode;
    auto& dfgNodeAttr = _mapping->dfgNodeAttr(dfgNode->id());
    DFGIONode* dfgIONode = dynamic_cast<DFGIONode*>(dfgNode);
    std::map<int, CfgData> cfg;
    // int ioctrlId = subAdg->output(0).first; // IOB has only one output connected to IOController
    // CfgDataLoc ioctrlCfgLoc = node->configInfo(ioctrlId);
    // int ioctrlCfgLen = ioctrlCfgLoc.high - ioctrlCfgLoc.low + 1;
    // CfgData ioctrlCfg(ioctrlCfgLen);
    // while(ioctrlCfgLen > 0){ // config data of IOController is set by the host
    //     ioctrlCfg.data.push_back((uint32_t)0);
    //     ioctrlCfgLen -= 32;
    // }
    // cfg[ioctrlCfgLoc.low] = ioctrlCfg;
    int isStore = 0;
    auto dfg = _mapping->getDFG();
    // dfg->printVariableConfigNodes();
    bool isUsedAsOB = dfg->getOutNodes().count(dfgNode->id());
    if(isUsedAsOB){ // IOB used as OB
        isStore = 1;     
    }else{ // IOB used as IB
        isStore = 0;
    }
    int II = _mapping->II();
    int latency = dfgNodeAttr.lat - dfgNode->opLatency(); // substract load/store latency
    
    //// @jhlou: for nodes following merge op, add additional latency(acr counting 3 more cycles) 
    // latency = addAdditionalLatencyForOpsFolloingMERGE(dfgNode, latency);
    if(dfgNode->additionalStartDelay() != 0){
        latency += dfgNode->additionalStartDelay();
    }

    int dataBytes = _mapping->getADG()->bitWidth() / 8;
    int baseAddr = _dfgIoSpadAddrs[dfgNode->id()];
    int offset = dfgIONode->reducedMemOffset() / dataBytes;
    int baseAddrId = node->cfgIdMap["BaseAddr"];    
    // CfgDataLoc baseAddrCfgLoc = node->configInfo(baseAddrId);
    // int baseAddrCfgLen = baseAddrCfgLoc.high - baseAddrCfgLoc.low + 1;
    // CfgData baseAddrCfg(baseAddrCfgLen);    
    // baseAddrCfg.data.push_back((uint32_t)(baseAddr+offset)); 
    // cfg[baseAddrCfgLoc.low] = baseAddrCfg;
    addCfgData(cfg, node->configInfo(baseAddrId), (uint32_t)(baseAddr+offset));
    int dfgNestedLevels = dfgIONode->getNestedLevels();
    int iobNestedLevels = _mapping->getADG()->iobAgNestLevels();
    assert(dfgNestedLevels <= iobNestedLevels);
    auto& pattern = dfgIONode->pattern();
    // dumpCfgData(std::cout);
    // IsADGNodeConfigVariable(node);
    VariableConfig* varconfig = nullptr;
    if(IsADGNodeConfigVariable(node)){
        varconfig = getConfigVariable(node);
        varconfig->print();
    }
    for(int i = 0; i < iobNestedLevels; i++){
        int stride = 0;
        int cycles = 0;
        if(i < dfgNestedLevels){
            stride = pattern[i].first;
            assert(stride % dataBytes == 0);
            stride = stride / dataBytes;
            cycles = pattern[i].second;
        }
        std::string strideName = "Stride" + std::to_string(i);
        int strideId = node->cfgIdMap[strideName];
        CfgDataLoc strideCfgLoc = node->configInfo(strideId);
        int strideCfgLen = strideCfgLoc.high - strideCfgLoc.low + 1;
        uint32_t strideAlign = stride & ((1 << strideCfgLen) - 1);
        // CfgData strideCfg(strideCfgLen);        
        // strideCfg.data.push_back(strideAlign); 
        // cfg[strideCfgLoc.low] = strideCfg;
        addCfgData(cfg, strideCfgLoc, (uint32_t)strideAlign);

        /// Handle variable configuration
        // std::string str= varconfig->pattern[i].second;
        if(varconfig != nullptr && i < varconfig->pattern.size()  
                                && !IsConstStrExpr(varconfig->pattern[i].first)){ /// stride
            const uint32_t key = strideCfgLoc.low;
            std::pair<uint32_t, std::string> value = 
                        std::make_pair((uint32_t)strideAlign, varconfig->pattern[i].first);
            varconfig->LSBToLenExpr[key] = value;
        }

        std::string cyclesName = "Cycles" + std::to_string(i);
        int cyclesId = node->cfgIdMap[cyclesName];
        addCfgData(cfg, node->configInfo(cyclesId), (uint32_t)cycles);

        /// Handle variable configuration
        // str= varconfig->pattern[i].second;
        if(varconfig != nullptr && i < varconfig->pattern.size() 
                                && !IsConstStrExpr(varconfig->pattern[i].second)){ /// cycles
            const uint32_t key(node->configInfo(cyclesId).low);
            std::string str= varconfig->pattern[i].second;
            const std::pair<uint32_t, std::string> value = 
                    std::make_pair(node->configInfo(cyclesId).high - node->configInfo(cyclesId).low + 1, str);
            varconfig->LSBToLenExpr[key] = value;
        }
        // CfgDataLoc cyclesCfgLoc = node->configInfo(cyclesId);
        // int cyclesCfgLen = cyclesCfgLoc.high - cyclesCfgLoc.low + 1;
        // CfgData cyclesCfg(cyclesCfgLen);
        // cyclesCfg.data.push_back((uint32_t)cycles); 
        // cfg[cyclesCfgLoc.low] = cyclesCfg;
    }
    // dumpCfgData(std::cout);
    int iiId = node->cfgIdMap["II"];
    addCfgData(cfg, node->configInfo(iiId), (uint32_t)II);
    // CfgDataLoc iiCfgLoc = node->configInfo(iiId);
    // int iiCfgLen = iiCfgLoc.high - iiCfgLoc.low + 1;
    // CfgData iiCfg(iiCfgLen);
    // iiCfg.data.push_back((uint32_t)II); 
    // cfg[iiCfgLoc.low] = iiCfg;
    int latencyId = node->cfgIdMap["Latency"];
    addCfgData(cfg, node->configInfo(latencyId), (uint32_t)latency);
    // CfgDataLoc latencyCfgLoc = node->configInfo(latencyId);
    // int latencyCfgLen = latencyCfgLoc.high - latencyCfgLoc.low + 1;
    // CfgData latencyCfg(latencyCfgLen);
    // latencyCfg.data.push_back((uint32_t)latency); 
    // cfg[latencyCfgLoc.low] = latencyCfg;
    int isStoreId = node->cfgIdMap["IsStore"];
    addCfgData(cfg, node->configInfo(isStoreId), (uint32_t)isStore);
    // CfgDataLoc isStoreCfgLoc = node->configInfo(isStoreId);
    // int isStoreCfgLen = isStoreCfgLoc.high - isStoreCfgLoc.low + 1;
    // CfgData isStoreCfg(isStoreCfgLen);
    // isStoreCfg.data.push_back((uint32_t)isStore); 
    // cfg[isStoreCfgLoc.low] = isStoreCfg;
    auto op = dfgNode->operation();

    // dumpCfgData(std::cout);
    if(node->cfgIdMap.count("UseAddr")){        
        int useAddr = op == "LOAD" || op == "STORE" || op == "CLOAD" || op == "CSTORE";
        int useAddrId = node->cfgIdMap["UseAddr"];
        addCfgData(cfg, node->configInfo(useAddrId), (uint32_t)useAddr);
        // CfgDataLoc useAddrCfgLoc = node->configInfo(useAddrId);
        // int useAddrCfgLen = useAddrCfgLoc.high - useAddrCfgLoc.low + 1;
        // CfgData useAddrCfg(useAddrCfgLen);
        // useAddrCfg.data.push_back((uint32_t)useAddr); 
        // cfg[useAddrCfgLoc.low] = useAddrCfg;
    }
    if(node->cfgIdMap.count("UseEn")){        
        int useEn = op == "CLOAD" || op == "CSTORE";
        int useEnId = node->cfgIdMap["UseEn"];
        addCfgData(cfg, node->configInfo(useEnId), (uint32_t)useEn);
        // CfgDataLoc useEnCfgLoc = node->configInfo(useEnId);
        // int useEnCfgLen = useEnCfgLoc.high - useEnCfgLoc.low + 1;
        // CfgData useEnCfg(useEnCfgLen);
        // useEnCfg.data.push_back((uint32_t)useEn); 
        // cfg[useEnCfgLoc.low] = useEnCfg;
    }
    if(op != "INPUT"){ // only INPUT node donot use Mux     
        ADGNode* delayNode = nullptr;
        std::map<int, int> delayUsed;
        for(auto& elem : dfgNode->inputEdges()){
            int eid = elem.second;
            // auto edge = dfg->edge(eid);
            // std::cout << "eid: " << eid << ", " << dfg->node(edge->srcId())->name() << " -> " << dfg->node(edge->dstId())->name() << std::endl;
            auto& edgeAttr = _mapping->dfgEdgeAttr(eid);
            int inputIdx = edgeAttr.edgeLinks.rbegin()->srcPort; // last edgeLInk, dst port
            auto muxPair = subAdg->input(inputIdx).begin(); // one input only connected to one Mux
            int muxId = muxPair->first;
            int muxCfgData = muxPair->second;
            auto mux = subAdg->node(muxId);
            addCfgData(cfg, node->configInfo(muxId), (uint32_t)muxCfgData);
            // CfgDataLoc muxCfgLoc = node->configInfo(muxId);
            // CfgData muxCfg((muxCfgLoc.high - muxCfgLoc.low + 1), (uint32_t)muxCfgData);
            // cfg[muxCfgLoc.low] = muxCfg;
            auto delayPair = mux->output(0).begin();
            auto currentDelayNode = subAdg->node(delayPair->first);
            if(delayNode && delayNode->id() != currentDelayNode->id()){
                std::cout << "Invalid IOB configuration " << node->name()
                          << " (id=" << node->id()
                          << "): input muxes do not share one DelayPipe"
                          << std::endl;
                exit(1);
            }     
            delayNode = currentDelayNode;
            delayUsed[delayPair->second] = edgeAttr.delay;
        }  
        addIobDelayCfg(cfg, node, delayNode, delayUsed);
    }
    // dumpCfgData(std::cout);
    return cfg;
}


// get config data for GIB, return<LSB-location, CfgData>
std::map<int, CfgData> Configuration::getGibCfgData(GIBNode* node){
    int adgNodeId = node->id();
    ADG* subAdg = node->subADG();
    auto& adgNodeAttr = _mapping->adgNodeAttr(adgNodeId);
    auto& passEdges = adgNodeAttr.dfgEdgePass;
    if(passEdges.empty()){
        return {};
    }
    std::map<int, CfgData> cfg;
    for(auto& elem : passEdges){
        int muxId = subAdg->output(elem.dstPort).first; // one output connected to one mux
        if(muxId == subAdg->id()){ // actually connected to input port
            continue;
        }
        auto mux = subAdg->node(muxId);        
        // find srcPort
        for(auto in : mux->inputs()){
            if(in.second.second == elem.srcPort){ 
                addCfgData(cfg, node->configInfo(muxId), (uint32_t)in.first);
                // CfgDataLoc muxCfgLoc = node->configInfo(muxId);
                // CfgData muxCfg(muxCfgLoc.high - muxCfgLoc.low + 1, (uint32_t)in.first);
                // cfg[muxCfgLoc.low] = muxCfg;
                break;
            }
        }
    }
    return cfg;
}

/// Get the variable config of adg node
VariableConfig* Configuration::getConfigVariable(ADGNode* node){
    if(!_mapping->isMapped(node)){
        return nullptr;
    }
    int adgNodeId = node->id();
    ADG* subAdg = node->subADG();
    auto& adgNodeAttr = _mapping->adgNodeAttr(adgNodeId);
    DFGNode* dfgNode = adgNodeAttr.dfgNode;
    if(_mapping->getDFG()->VariableConfigNodes.find(dfgNode) 
            == _mapping->getDFG()->VariableConfigNodes.end())
        return nullptr;
    return &(_mapping->getDFG()->VariableConfigNodes[dfgNode]);
}

/// Check whether a node has variable configuration
bool Configuration::IsADGNodeConfigVariable(ADGNode* node){
    // getConfigVariable(node)->print();
    if(getConfigVariable(node) != nullptr)
        return getConfigVariable(node)->IsVariable();
    
    return  false;
}

// /// A stupid tool function(shit mountain)
// static std::string getVarConfigFromAddr(std::map<int, std::vector<int>> lsb2addr,int addr_tofind, VariableConfig* vcfg){
//     if(vcfg == nullptr)
//         return "";
//     std::string var;
//     for(auto& elem: lsb2addr){
//         for(auto& addr :  elem.second){
//             if(addr == addr_tofind){
//                 return vcfg->LSBToLenExpr[elem.first].second;
//             }
//         }
//     }
//     assert(false && "Can't run to this position");
// }
int Configuration::findMask32(uint32_t value, int begin, int l) {
    if(value == 0 || l == 0 || begin >= 32) 
        return -1;
    uint32_t mask = (1U << l) - 1; 

    for (int i = begin; i <= 32; i++) {
        if ((value & (mask << i)) == (mask << i)) {
            return i;      
        }
    }
    return -1; 
}

int Configuration::findMask16(uint16_t value, short begin, short l) {
    if(value == 0 || l == 0 || begin >= 16) 
        return -1;
    uint16_t mask = (1U << l) - 1; 

    for (uint16_t i = begin; i <= 16; i++) {
        uint16_t a = (value & (uint16_t)(mask << i));
        uint16_t b = (uint16_t)(mask << i);
        if ((value & (uint16_t)(mask << i)) == (uint16_t)(mask << i)) {
            return i;      
        }
    }
    return -1; 
}

// get config data for ADG node
void Configuration::getNodeCfgData(ADGNode* node, std::vector<CfgDataPacket>& cfg){
    std::map<int, CfgData> cfgMap;
    if(node->type() == "GPE"){
        cfgMap = getGpeCfgData(dynamic_cast<GPENode*>(node));
    }else if(node->type() == "GIB"){
        cfgMap = getGibCfgData(dynamic_cast<GIBNode*>(node));
     }else if(node->type() == "IOB"){
        // cfgMap = getIobCfgData(dynamic_cast<IOBNode*>(node)); // jhlou
        cfgMap = getIobPingpongCfgData(dynamic_cast<IOBNode*>(node)); // jhlou
    }
    if(cfgMap.empty()){
        return;
    }
    // if(node->type() == "IOB") {
    //     std::cout << "IOB!" <<std::endl;
    //     cfgMap.clear();
    //     cfgMap = getIobCfgData(dynamic_cast<IOBNode*>(node));
    //     for(auto& elem : cfgMap){ // std::map auto-sort keys
    //         int lsb = elem.first;
    //         int len = elem.second.len;
    //         auto& data = elem.second.data;
    //         std::cout << std::hex;
    //         std::cout << "lsb:" << lsb ;
    //         std::cout << ",len:" << len ;
    //         std::cout << ",data:" << len ;
    //         for(auto data_ : data){
    //             std::cout << " " << data_;
    //         }

    //         std::cout << std::dec << std::endl;
    //     }
    // }

    ADG* adg = _mapping->getADG();
    int cfgDataWidth = adg->cfgDataWidth();
    int totalLen = cfgMap.rbegin()->first + cfgMap.rbegin()->second.len;
    int num = (totalLen+31)/32;
    std::vector<uint32_t> cfgDataVec(num, 0);
    std::set<uint32_t> addrs;

    if(IsADGNodeConfigVariable(node)){ 
        std::vector<uint32_t> VarcfgMaskVec(num, 0);
        VariableConfig* varconfig= getConfigVariable(node);
        std::map<std::string, unsigned> varconfigs_len;
        for(auto& elem : cfgMap){ // std::map auto-sort keys
            int lsb = elem.first;
            int len = elem.second.len;
            auto& data = elem.second.data;
            // cache valid address
            uint32_t targetAddr = lsb/cfgDataWidth;
            int addrNum = (len + (lsb%cfgDataWidth) + cfgDataWidth - 1)/cfgDataWidth;
            for(int i = 0; i < addrNum; i++){
                addrs.emplace(targetAddr+i);
            } 
            uint64_t temp_mask;
            if(MapHasKey(varconfig->LSBToLenExpr, lsb)){
                assert(varconfig->LSBToLenExpr[lsb].first == len);
                temp_mask = (1U << len) - 1;/// 64 bit
                varconfigs_len[varconfig->LSBToLenExpr[lsb].second] = len;
            }
            else{
                temp_mask = 0;
            }

            // cache data from 0 to MSB   
            int targetIdx = lsb/32;
            int offset = lsb%32;
            uint64_t tmpData = data[0];
            int dataIdx = 0;
            int dataLenLeft = 32;

            while(len > 0){
                if(len <= 32 - offset){
                    len = 0;
                    cfgDataVec[targetIdx] |= (tmpData << offset);
                    VarcfgMaskVec[targetIdx] |= (temp_mask << offset);
                }
                else{                          
                    dataLenLeft -= 32 - offset; 
                    cfgDataVec[targetIdx] |= (tmpData << offset);     
                    VarcfgMaskVec[targetIdx] |= (temp_mask << offset);           

                    targetIdx++;
                    dataIdx++;
                    tmpData >>= 32 - offset;
                    temp_mask >>= 32 - offset;
                    if(dataIdx < data.size()){
                        tmpData |= data[dataIdx] << dataLenLeft;
                        temp_mask |= 0;
                        dataLenLeft += 32;
                    }
                    len -= 32 - offset;
                    offset = 0;
                }
            }
        }
        // construct CfgDataPacket
        int cfgBlkOffset = adg->cfgBlkOffset();
        int cfgBlkIdx = node->cfgBlkIdx();
        // int x = node->x();
        uint32_t highAddr = uint32_t(cfgBlkIdx << cfgBlkOffset);
        int n;
        int mask;
        if(cfgDataWidth >= 32){
            assert(cfgDataWidth%32 == 0);
            n = cfgDataWidth/32;
        }else{
            assert(32%cfgDataWidth == 0);
            n = 32/cfgDataWidth;
            mask = (1 << cfgDataWidth) - 1;
        }
        std::vector<CfgDataPacket> cdp_masks;
        auto Iter = varconfigs_len.begin();
        assert(varconfigs_len.size() != 0);
        std::string varname = Iter->first;
        uint32_t begin = 0, len = Iter->second, len_to_find = len;
        for(auto addr : addrs){
            CfgDataPacket cdp(highAddr|addr);
            CfgDataPacket cdp_mask(highAddr|addr);

            if(cfgDataWidth >= 32){
                int size = cfgDataVec.size();
                for(int i = 0; i < n; i++){
                    int idx = addr*n+i;
                    uint32_t data = (idx < size)? cfgDataVec[idx] : 0;
                    uint32_t VarcfgMask = (idx < size)? VarcfgMaskVec[idx] : 0;
                    cdp.data.push_back(data);
                    cdp_mask.data.push_back(VarcfgMask);
                }
 
            }else{
                uint32_t data = (cfgDataVec[addr/n] >> ((addr%n)*cfgDataWidth)) & mask;
                uint32_t VarcfgMask = (VarcfgMaskVec[addr/n] >> ((addr%n)*cfgDataWidth)) & mask;
                cdp.data.push_back(data);
                cdp_mask.data.push_back(VarcfgMask);
            }
            
            //// find the mask and that's where the var config should replace
            for(int j = 0; j < cdp_mask.data.size(); j++){
                uint32_t data = cdp_mask.data[j];
                uint16_t data0 = data & 0xffff;
                uint16_t data1 = (data & 0xffff0000) >> 16 ;
                // while(begin < 31){
                //     int pos = findMask32(data, begin, len_to_find);
                //     if(pos != -1) {
                //         ConfigReplaceInfo CRinfo;
                //         CRinfo.addr = cdp_mask.addr;
                //         CRinfo.Idx0 = cfg.size();
                //         CRinfo.Idx1 = j;
                //         if(pos == 0){
                //             //// mask is like 000111
                //             CRinfo.rshift = len_to_find;
                //         }
                //         else{//// mask is like 111000
                //             CRinfo.lshift = pos;
                //         }
                //         if(pos + len <= 32){
                //             //// contained in this data
                //             Iter++;
                //             if(Iter != varconfigs_len.end()){
                //                 len = Iter->second;
                //                 varname = Iter->first;
                //                 len_to_find = len;
                //             }
                //             else{
                //                 len_to_find = 0;
                //             }
                //         }
                //         else{
                //             len_to_find = len + pos - 32;
                //         }
                //         begin = pos + len;
                //     }
                //     else {
                //         begin = 32;
                //     }
                // }
                //// low bits of 32bit data
                while(begin < 15){
                    int pos = findMask16(data0, begin, len_to_find);
                    if(pos != -1) {
                        ConfigReplaceInfo CRinfo;
                        CRinfo.addr = cdp_mask.addr;
                        CRinfo.Idx0 = cfg.size();
                        CRinfo.Idx1 = 0;
                        if(pos == 0){
                            //// mask is like 000111
                            CRinfo.rshift = len_to_find;
                        }
                        else{//// mask is like 111000
                            CRinfo.lshift = pos;
                        }
                        if(pos + len_to_find <= 16){
                            //// contained in this data
                            Iter++;
                            if(Iter != varconfigs_len.end()){
                                len = Iter->second;
                                varname = Iter->first;
                                len_to_find = len;
                            }
                            else{
                                len_to_find = 0;
                            }
                        }
                        else{
                            len_to_find = len_to_find + pos - 16;
                        }
                        begin = pos + len_to_find;
                        VarReplaceInfo[varname].push_back(CRinfo);
                    }
                    else {
                        begin = 16;
                    }
                }
                begin = 0;
                
                //// high bits of 32bit data
                while(begin < 15){
                    int pos = findMask16(data1, begin, len_to_find);
                    if(pos != -1) {
                        ConfigReplaceInfo CRinfo;
                        CRinfo.addr = cdp_mask.addr;
                        CRinfo.Idx0 = cfg.size();
                        CRinfo.Idx1 = 1; //// high bits
                        if(pos == 0){
                            //// mask is like 000111
                            CRinfo.rshift = len_to_find;
                        }
                        else{//// mask is like 111000
                            CRinfo.lshift = pos;
                        }
                        if(pos + len <= 16){
                            //// contained in this data
                            Iter++;
                            if(Iter != varconfigs_len.end()){
                                len = Iter->second;
                                varname = Iter->first;
                                len_to_find = len;
                            }
                            else{
                                len_to_find = 0;
                            }
                        }
                        else{
                            len_to_find = len + pos - 16;
                        }
                        begin = pos + len;
                        VarReplaceInfo[varname].push_back(CRinfo);
                    }
                    else {
                        begin = 16;
                    }
                }
                begin = 0;
            }

            if(node->type() == "IOB") {
                cdp.print();
                cdp_mask.print();
            }
            cfg.push_back(cdp);
        }
    }
    else{
        //// do not contain variable configuration
        for(auto& elem : cfgMap){ // std::map auto-sort keys
            int lsb = elem.first;
            int len = elem.second.len;
            auto& data = elem.second.data;
            // cache valid address
            uint32_t targetAddr = lsb/cfgDataWidth;
            int addrNum = (len + (lsb%cfgDataWidth) + cfgDataWidth - 1)/cfgDataWidth;
            for(int i = 0; i < addrNum; i++){
                addrs.emplace(targetAddr+i);
            } 

            // cache data from 0 to MSB   
            int targetIdx = lsb/32;
            int offset = lsb%32;
            uint64_t tmpData = data[0];
            int dataIdx = 0;
            int dataLenLeft = 32;

            while(len > 0){
                if(len <= 32 - offset){
                    len = 0;
                    cfgDataVec[targetIdx] |= (tmpData << offset);
                }
                else{                          
                    dataLenLeft -= 32 - offset; 
                    cfgDataVec[targetIdx] |= (tmpData << offset);           

                    targetIdx++;
                    dataIdx++;
                    tmpData >>= 32 - offset;
                    if(dataIdx < data.size()){
                        tmpData |= data[dataIdx] << dataLenLeft;
                        dataLenLeft += 32;
                    }
                    len -= 32 - offset;
                    offset = 0;
                }
            }
        }
        // construct CfgDataPacket
        int cfgBlkOffset = adg->cfgBlkOffset();
        int cfgBlkIdx = node->cfgBlkIdx();
        // int x = node->x();
        uint32_t highAddr = uint32_t(cfgBlkIdx << cfgBlkOffset);
        int n;
        int mask;
        if(cfgDataWidth >= 32){
            assert(cfgDataWidth%32 == 0);
            n = cfgDataWidth/32;
        }else{
            assert(32%cfgDataWidth == 0);
            n = 32/cfgDataWidth;
            mask = (1 << cfgDataWidth) - 1;
        }
        for(auto addr : addrs){
            CfgDataPacket cdp(highAddr|addr);
            if(cfgDataWidth >= 32){
                int size = cfgDataVec.size();
                for(int i = 0; i < n; i++){
                    int idx = addr*n+i;
                    uint32_t data = (idx < size)? cfgDataVec[idx] : 0;
                    cdp.data.push_back(data);
                }
 
            }else{
                uint32_t data = (cfgDataVec[addr/n] >> ((addr%n)*cfgDataWidth)) & mask;
                cdp.data.push_back(data);
            }
            cfg.push_back(cdp);
        }
    }
    // printVarReplaceInfo();
}

// get config data for ADG
void Configuration::getCfgData(std::vector<CfgDataPacket>& cfg){
    cfg.clear();
    for(auto& elem : _mapping->getADG()->nodes()){
        getNodeCfgData(elem.second, cfg);
    }
}


void Configuration::printVarReplaceInfo(){
    std::cout << "------- Print VarReplaceInfo -------" << std::endl;
    for(auto& elem : VarReplaceInfo){
        std::cout << " variable config name: " << elem.first << std::endl;
        for(auto& info: elem.second) {
            info.print();
        }
    }
    std::cout << "------- End Print -------" << std::endl;
}



// dump config data
void Configuration::dumpCfgData(std::ostream& os){
    std::vector<CfgDataPacket> cfg;
    getCfgData(cfg);
    ADG* adg = _mapping->getADG();
    int cfgAddrWidth = adg->cfgAddrWidth();
    int cfgDataWidth = adg->cfgDataWidth();
    int addrWidthHex = (cfgAddrWidth+3)/4;
    int dataWidthHex = std::min(cfgDataWidth/4, 8);
    os << std::hex;
    for(auto& cdp : cfg){
        os << std::setw(addrWidthHex) << std::setfill('0') << (cdp.addr) << " ";
        for(int i = cdp.data.size() - 1; i >= 0; i--){
            os << std::setw(dataWidthHex) << std::setfill('0') << cdp.data[i];
        }
        os << std::endl;
    }
    os << std::dec;
}
