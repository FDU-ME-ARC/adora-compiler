#include "mapper/configuration.h"
// @jhlou: Get Pingpong related configuration

void addPingpongCfgData(std::map<int, CfgData> &cfg, const CfgDataLoc &loc, uint32_t data){
    CfgData loc_data(loc.high - loc.low + 1, data);
    loc_data.isRuntimeCfg = true;

    cfg[loc.low] = loc_data;
}

// get config data for IOB, return<LSB-location, CfgData>
std::map<int, CfgData> Configuration::getIobPingpongCfgData(IOBNode* node, bool en_pingpong, int pingpong_phase){
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
    
    //// @jhlou: for merge op, add additional latency(acr counting 3 more cycles) 
    // latency = addAdditionalLatencyForOpsFolloingMERGE(dfgNode, latency);
    if(dfgNode->additionalStartDelay() != 0){
        latency += dfgNode->additionalStartDelay();
    }

    int dataBytes = _mapping->getADG()->bitWidth() / 8;
    int baseAddr = _dfgIoSpadAddrs[dfgNode->id()];
    int offset = dfgIONode->reducedMemOffset() / dataBytes;
    int baseAddrId = node->cfgIdMap["BaseAddr"];    

    int memsize = dfgIONode->memSize();
    if(!en_pingpong)
        addCfgData(cfg, node->configInfo(baseAddrId), (uint32_t)(baseAddr+offset));
    else{
        addPingpongCfgData(cfg, node->configInfo(baseAddrId), (uint32_t)(baseAddr+offset+memsize*pingpong_phase));
    }


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
            printf("stride:%d\n", stride);
            stride = pattern[i].first;
            if(stride % dataBytes != 0){
                std::cout << "Error in data bytes:" << dataBytes << "\n";
                assert(stride % dataBytes == 0);
            }

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

// get config data for ADG node
void Configuration::getNodePingpongCfgData(ADGNode* node, std::vector<CfgDataPacket>& cfg, int pingpong_phase){
    if(!_mapping->isMapped(node)){
        return;
    }
    std::map<int, CfgData> cfgMap;

    int adgNodeId = node->id();
    ADG* subAdg = node->subADG();
    auto& adgNodeAttr = _mapping->adgNodeAttr(adgNodeId);
    DFGNode* dfgNode = adgNodeAttr.dfgNode;
    auto& dfgNodeAttr = _mapping->dfgNodeAttr(dfgNode->id());

    if(node->type() == "GPE"){
        return ;
    }else if(node->type() == "GIB"){
        return ;
    }else if(node->type() == "IOB"){
        DFGIONode* dfgIONode = dynamic_cast<DFGIONode*>(dfgNode);
        if(dfgIONode->isPingpong()){
            cfgMap = getIobPingpongCfgData(dynamic_cast<IOBNode*>(node), /*en_pingpong*/true, /*pingpong_phase*/pingpong_phase);
        }
    }
    if(cfgMap.empty()){
        return;
    }

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
            bool isRuntimeCFG = elem.second.isRuntimeCfg;
            if(!isRuntimeCFG){
                continue;
            }

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
            bool isRuntimeCFG = elem.second.isRuntimeCfg;
            // if(!isRuntimeCFG){
            //     continue;
            // }

            // cache valid address
            uint32_t targetAddr = lsb/cfgDataWidth;
            int addrNum = (len + (lsb%cfgDataWidth) + cfgDataWidth - 1)/cfgDataWidth;
            
            if(isRuntimeCFG){
                for(int i = 0; i < addrNum; i++){
                    addrs.emplace(targetAddr+i);
                } 
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


void Configuration::getPingpongCfgData(std::vector<CfgDataPacket>& cfg_ping, std::vector<CfgDataPacket>& cfg_pong){
    cfg_ping.clear();
    cfg_pong.clear();
    for(auto& elem : _mapping->getADG()->nodes()){
        getNodePingpongCfgData(elem.second, cfg_ping, 0);
        getNodePingpongCfgData(elem.second, cfg_pong, 1);
    }
}
