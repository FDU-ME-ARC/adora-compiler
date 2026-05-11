#include "mapper/io_scheduler.h"
#include "mapper/agent_trace.h"
#include "mapper/online_ranker.h"

#include <sstream>
#include <vector>

namespace {

struct SpadBankCand {
    int bank_id;
    int iob_idx;
    int start;
    int end;
    int status;   // 0: both free; 1: old avail, older not; 2: older avail, old not; 3: both not
    int cand_dep; // expected dfgIoInfo.dep if this candidate wins
    bool is_default; // true iff this is the bank the greedy path selected
};

static std::string dumpBankWindow(const std::vector<spadBankStatus>& v) {
    std::ostringstream o;
    o << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) o << ",";
        o << "{\"iob\":" << v[i].iob
          << ",\"used\":" << v[i].used
          << ",\"start\":" << v[i].start
          << ",\"end\":" << v[i].end << "}";
    }
    o << "]";
    return o.str();
}

static std::string serialiseBankWindow(
    const std::vector<spadBankStatus>& cur,
    const std::vector<spadBankStatus>& old_,
    const std::vector<spadBankStatus>& older) {
    std::ostringstream o;
    o << "{\"cur\":" << dumpBankWindow(cur)
      << ",\"old\":" << dumpBankWindow(old_)
      << ",\"older\":" << dumpBankWindow(older) << "}";
    return o.str();
}

} // namespace

IOScheduler::IOScheduler(ADG *adg)
{
    _adg = adg;
    int banks = adg->numIobNodes();
    _cur_bank_status.assign(banks, {0, 0, 0, 0});
    _old_bank_status.assign(banks, {0, 0, 0, 0});
    _older_bank_status.assign(banks, {0, 0, 0, 0});
    _old_cfg_status = {0, 0};
}

IOScheduler::~IOScheduler()
{
}

void IOScheduler::ioSchedule(Mapping *mapping)
{
    // std::map<int, dfgIoInfo> res;
    _dfg_io_infos.clear();
    _ex_dep = 0;
    _iob_ens = 0;
    _dep_cost = 0;
    int bankNum = _adg->numIobNodes();
    _cur_bank_status.assign(bankNum, {0, 0, 0, 0});
    DFG* dfg = mapping->getDFG();
    auto outNodeIds = dfg->getOutNodes(); // OUTPUT/STORE nodes
    // LD_DEP_ST_LAST_SEC_TASK cost = 1
    long inNodeNum = dfg->ioNodes().size() - outNodeIds.size(); // LD_DEP_EX_LAST_TASK cost
    long inNodeNum_2 = inNodeNum * inNodeNum;   // EX_DEP_ST_LAST_TASK dep cost
    long inNodeNum_3 = inNodeNum_2 * inNodeNum; // LD_DEP_ST_LAST_TASK dep cost
    int sizeofBank = _adg->iobSpadBankSize();
    int dataByte = _adg->bitWidth() / 8;
    int depthofBank = sizeofBank / dataByte; // accomodate total data number
    for(auto& id : dfg->ioNodes()){
        dfgIoInfo ioInfo;
        // auto dfgIONode = dfg->node(id);
        bool isStore = false;
        if(outNodeIds.count(id)){ 
            isStore = true;
        }
        ioInfo.isStore = isStore;
        int memSize = dynamic_cast<DFGIONode*>(dfg->node(id))->memSize();
        int spadDataByte = _adg->cfgSpadDataWidth() / 8; // dual ports of cfg-spad have the same width 
        memSize = (memSize + spadDataByte - 1) / spadDataByte * spadDataByte; // align to spadDataByte
        auto& attr =  mapping->dfgNodeAttr(id);
        int iobId = attr.adgNode->id();
        int iobIdx = dynamic_cast<IOBNode*>(_adg->node(iobId))->index();
        _iob_ens |= 1 << iobIdx;
        std::vector<int> banks = _adg->iobToSpadBanks(iobIdx); // spad banks connected to this IOB
        int minBank = *(std::min_element(banks.begin(), banks.end()));
        std::vector<int> availBanks;
        for(int bank : banks){ // two IOs of the same DFG cannot access the same bank
            if(_cur_bank_status[bank].used == 0){
                availBanks.push_back(bank);
            }
        }
        assert(!availBanks.empty());
        bool allocated = false;
        int selBank;
        int selStart = 0;
        std::vector<std::pair<int, int>> bankStatus; // <status, start-addr>
        // 0: both available; 1: old available, older not; 2: older available, old not; 3: both not
        for(int bank : availBanks){            
            int oldUsed = _old_bank_status[bank].used;
            int olderUsed = _older_bank_status[bank].used;
            int oldStart = _old_bank_status[bank].start;
            int olderStart = _older_bank_status[bank].start;
            int oldEnd = _old_bank_status[bank].end;
            int olderEnd = _older_bank_status[bank].end;
            int oldStatus = 0; // 0: free; 1: used but available; 2: not available
            int oldSelStart = 0;
            if(oldUsed == 0){               
                oldStatus = 0;
                oldSelStart = 0;
            }else if((oldUsed == 1 && !isStore) || (oldUsed == 2 && isStore)){ // the same operation
                if(memSize <= sizeofBank - oldEnd){
                    oldStatus = 1;
                    oldSelStart = oldEnd;
                }else if(memSize <= oldStart){
                    oldStatus = 1;
                    oldSelStart = 0;
                }else{
                    oldStatus = 2;
                }
            }else if(oldUsed == 1 && isStore){ // old load, current store
                oldStatus = 0;
                oldSelStart = 0;
            }else{ // old store, current load; cannot access the same bank simultaneously
                oldStatus = 2;
            }

            int olderStatus = 0; // 0: free; 1: used but available; 2: not available
            int olderSelStart = 0;
            if(olderUsed == 2 && !isStore){ // the same operation
                olderStatus = 2;                
            }else{
                olderStatus = 0;
                olderSelStart = 0;
            }

            int status = 0; // 0: both available; 1: old available, older not; 2: older available, old not; 3: both not            
            if(oldStatus < 2 && olderStatus == 0){
                status = 0;
                selStart = oldSelStart;              
            }else if(oldStatus < 2 && olderStatus == 2){
                status = 1;
                selStart = oldSelStart;
            }else if(oldStatus == 2 && olderStatus == 0){
                status = 2;
                selStart = olderSelStart;
            }else{ // (oldStatus == 2 && olderStatus == 2)
                status = 3;
                selStart = 0;
            }            
            if(status == 0){
                selBank = bank;
                ioInfo.dep = 0;
                allocated = true;
                break;
            }
            bankStatus.push_back(std::make_pair(status, selStart));
        }            
        
        int availBankNum = bankStatus.size();
        if(!allocated){
            for(int i = 0; i < availBankNum; i++){
                if(bankStatus[i].first == 1){
                    selBank = availBanks[i];
                    selStart = bankStatus[i].second;
                    assert((!isStore) && _older_bank_status[selBank].used == 2);
                    ioInfo.dep = LD_DEP_ST_LAST_SEC_TASK;
                    _dep_cost += 1;               
                    allocated = true;
                    break;
                }
            }
        }
        if(!allocated && isStore){
            selBank = availBanks[0];
            selStart = bankStatus[0].second;
            ioInfo.dep = EX_DEP_ST_LAST_TASK;
            _ex_dep = EX_DEP_ST_LAST_TASK;
            allocated = true;
        }else if(!allocated){
            for(int i = 0; i < availBankNum; i++){
                selBank = availBanks[i];
                if(_old_bank_status[selBank].used == 1){                    
                    selStart = bankStatus[i].second;
                    ioInfo.dep = LD_DEP_EX_LAST_TASK;
                    _dep_cost += inNodeNum;        
                    allocated = true;
                    break;
                }
            }
            if(!allocated){
                selBank = availBanks[0];
                selStart = bankStatus[0].second;
                ioInfo.dep = LD_DEP_ST_LAST_TASK;
                _dep_cost += inNodeNum_3;
                allocated = true;
            }
        }
        assert(allocated);

        // --- runtime-online-v0 §1 allocate_spad_banks emission ---
        // The greedy above has resolved (selBank, selStart, ioInfo.dep). We
        // expose the same candidate set (availBanks with their computed
        // status/selStart) to the online ranker. For this landing the hook is
        // advisory telemetry: the greedy's pick stays authoritative so
        // conflict-free allocation is preserved by the mapper.
        if (OnlineRanker::enabled() && !availBanks.empty()) {
            std::vector<SpadBankCand> cands;
            cands.reserve(availBanks.size());
            // index 0 must be the greedy default so out-of-range agent
            // responses fall back to it.
            SpadBankCand def;
            def.bank_id = selBank;
            def.iob_idx = iobIdx;
            def.start = selStart;
            def.end = selStart + memSize;
            def.status = 0;
            def.cand_dep = ioInfo.dep;
            def.is_default = true;
            cands.push_back(def);
            for (size_t bi = 0; bi < availBanks.size(); ++bi) {
                int bank = availBanks[bi];
                if (bank == selBank) continue;
                SpadBankCand c;
                c.bank_id = bank;
                c.iob_idx = iobIdx;
                c.status = (bi < bankStatus.size()) ? bankStatus[bi].first : 0;
                c.start = (bi < bankStatus.size()) ? bankStatus[bi].second : 0;
                c.end = c.start + memSize;
                c.cand_dep = 0;
                c.is_default = false;
                cands.push_back(c);
            }

            std::ostringstream req;
            req << "{\"schema_version\":\"runtime-online-v0\","
                << "\"phase\":\"allocate_spad_banks\","
                << "\"request_id\":\"" << agentTraceJsonEscape(AgentTrace::runId())
                << ":allocate_spad_banks:" << _task_id << ":" << id
                << ":" << OnlineRanker::budgetUsed() << "\","
                << "\"task_id\":" << _task_id << ","
                << "\"dfg_io_id\":" << id << ","
                << "\"dfg_io_info\":{"
                << "\"is_store\":" << (isStore ? "true" : "false")
                << ",\"mem_size\":" << memSize
                << ",\"iob_idx\":" << iobIdx
                << "},"
                << "\"bank_status_window\":"
                << serialiseBankWindow(_cur_bank_status, _old_bank_status, _older_bank_status)
                << ",\"legal_candidates\":[";
            for (size_t i = 0; i < cands.size(); ++i) {
                if (i) req << ",";
                const auto& c = cands[i];
                req << "{\"action_type\":\"assign_spad_bank\","
                    << "\"bank_id\":" << c.bank_id
                    << ",\"iob_idx\":" << c.iob_idx
                    << ",\"base_offset\":" << c.start
                    << ",\"size_bytes\":" << (c.end - c.start)
                    << ",\"status\":" << c.status
                    << ",\"cand_dep\":" << c.cand_dep
                    << ",\"is_default\":" << (c.is_default ? "true" : "false")
                    << "}";
            }
            req << "]";
            auto adgCtx = OnlineRanker::adgContext();
            if (!adgCtx.adg_hash.empty()) {
                req << ",\"adg_ref\":{\"adg_hash\":\""
                    << agentTraceJsonEscape(adgCtx.adg_hash) << "\"}";
            }
            req << "}";

            auto decision = OnlineRanker::rank(req.str(), static_cast<int>(cands.size()));
            int chosenIdx = 0;
            if (decision.succeeded
                && decision.selected_index >= 0
                && decision.selected_index < static_cast<int>(cands.size())) {
                chosenIdx = decision.selected_index;
            }
            // Apply agent's choice: override greedy selection when agent picked a different bank.
            if (chosenIdx != 0) {
                selBank  = cands[chosenIdx].bank_id;
                selStart = cands[chosenIdx].start;
            }
            if (AgentTrace::enabled()) {
                std::ostringstream evt;
                evt << "{\"task_id\":" << _task_id
                    << ",\"dfg_io_id\":" << id
                    << ",\"used\":" << (decision.used ? "true" : "false")
                    << ",\"succeeded\":" << (decision.succeeded ? "true" : "false")
                    << ",\"selected_index\":" << decision.selected_index
                    << ",\"applied_index\":" << chosenIdx
                    << ",\"applied_bank_id\":" << cands[chosenIdx].bank_id
                    << ",\"default_bank_id\":" << cands[0].bank_id
                    << ",\"used_fallback\":" << (decision.used_fallback ? "true" : "false")
                    << ",\"advisory_only\":false"
                    << ",\"rationale\":\"" << agentTraceJsonEscape(decision.rationale) << "\""
                    << ",\"error\":\"" << agentTraceJsonEscape(decision.error) << "\"}";
                AgentTrace::emit("allocate_spad_banks", "runtime_decision", evt.str());
            }
        }
        // --- end hook ---

        ioInfo.addr = selBank * sizeofBank + selStart;
        ioInfo.iobAddr = ((selBank - minBank) * sizeofBank + selStart) / dataByte;    
        ioInfo.dep = 0;
        _dfg_io_infos[id] = ioInfo;        
        // std::cout << id << ": " << ioInfo.addr << std::endl;
        _cur_bank_status[selBank].used = isStore ? 2 : 1;
        _cur_bank_status[selBank].iob = iobIdx;
        _cur_bank_status[selBank].start = selStart;
        _cur_bank_status[selBank].end = selStart + memSize;        
    }
    if(_ex_dep > 0){
        _dep_cost += inNodeNum_2;
    }
    // return res;
}


// get the config data and dependence
void IOScheduler::genCfgData(Mapping *mapping, std::ostream &os)
{
    Configuration cfg(mapping);
    for(auto &elem : _dfg_io_infos){
        cfg.setDfgIoSpadAddr(elem.first, elem.second.iobAddr);
    }
    std::vector<CfgDataPacket> cfgData;
    cfg.getCfgData(cfgData);
    int cfgSpadDataByte = _adg->cfgSpadDataWidth() / 8;
    int cfgAddrWidth = _adg->cfgAddrWidth();
    int cfgDataWidth = _adg->cfgDataWidth();
    int alignWidth = (cfgAddrWidth > 16) ? 32 : 16;
    assert(alignWidth >= cfgAddrWidth && cfgDataWidth >= alignWidth);
    int cfgNum = 0;
    for(auto& cdp : cfgData){
        cfgNum += cdp.data.size() * 32 / cfgDataWidth;
    }

    if(cfgAddrWidth > 16){
        os << "\tvolatile unsigned int ";
    }else{
        os << "\tvolatile unsigned short ";
    }
    os << "cin[" << cfgNum << "][" << (1 + cfgDataWidth / alignWidth) << "] __attribute__((aligned(" << cfgSpadDataByte << "))) = {\n";
    os << std::hex;
    int alignWidthHex = alignWidth/4;
    for(auto& cdp : cfgData){
        os << "\t\t{";
        for(auto data : cdp.data){      
            if(alignWidth == 32){
                os << "0x" << std::setw(alignWidthHex) << std::setfill('0') << data << ", ";
            }else{
                os << "0x" << std::setw(alignWidthHex) << std::setfill('0') << (data & 0xffff) << ", ";
                os << "0x" << std::setw(alignWidthHex) << std::setfill('0') << (data >> 16) << ", ";
            }
            
        }
        os << "0x" << std::setw(alignWidthHex) << std::setfill('0') << (cdp.addr) << "},\n";
    }
    os << std::dec << "\t};\n\n";

    _cfg_num = cfgNum;
    _cfg_len = cfgNum * (alignWidth + cfgDataWidth) / 8; // length of config_addr and config_data in bytes
    int cfgSpadSize = _adg->cfgSpadSize();
    int cfgBaseAddr;
    _ld_cfg_dep = 0;
    if(_cfg_len <= cfgSpadSize - _old_cfg_status.end){
        cfgBaseAddr = _old_cfg_status.end;
    }else if(_cfg_len <= _old_cfg_status.start){
        cfgBaseAddr = 0;
    }else{ // cfg data space overlap last cfg data space
        cfgBaseAddr = 0;
        _ld_cfg_dep = LD_DEP_EX_LAST_TASK;
    }
    _old_cfg_status.start = cfgBaseAddr;
    _old_cfg_status.end = cfgBaseAddr + (_cfg_len +  cfgSpadDataByte - 1) / cfgSpadDataByte * cfgSpadDataByte;
    _old_cfg_status.end = std::min(_old_cfg_status.end, cfgSpadSize);
}


// generate CGRA instructions
std::pair<std::vector<std::string>, std::vector<std::string>> IOScheduler::genInstructions(Mapping *mapping, std::ostream &os)
{
    std::vector<std::string> ldArrayNames;
    std::vector<std::string> stArrayNames;
    int banks = _adg->numIobNodes();
    int sizeofBank = _adg->iobSpadBankSize();
    int cfgBaseAddrSpad = _old_cfg_status.start + banks * sizeofBank; // cfg spad on top of iob spad
    int cfgSpadDataByte = _adg->cfgSpadDataWidth() / 8;
    int cfgBaseAddrCtrl = _old_cfg_status.start / cfgSpadDataByte; // config base address the controller access
    os << "\tload_cfg((void*)cin, 0x" << std::hex << cfgBaseAddrSpad << std::dec << ", " 
       << _cfg_len << ", " << _task_id << ", " << _ld_cfg_dep << ");\n";
    std::vector<std::pair<int, int>> ld_data_deps; // <id, dep>
    std::vector<int> st_ids; // store node ids  
    for(auto &elem : _dfg_io_infos){
        if(elem.second.isStore){
            st_ids.push_back(elem.first);
        }else{
            int ori_dep = elem.second.dep;
            int sort_dep = 0;
            if(ori_dep == LD_DEP_ST_LAST_SEC_TASK){
                sort_dep = 1;
            }else if(ori_dep == LD_DEP_EX_LAST_TASK){
                sort_dep = 2;
            }else if(ori_dep == LD_DEP_ST_LAST_TASK){
                sort_dep = 3;
            }
            ld_data_deps.push_back(std::make_pair(elem.first, sort_dep));
        }        
    }
    // sort the load data commands according to array name to make same array access adjacent
    std::sort(ld_data_deps.begin(), ld_data_deps.end(), [&](std::pair<int, int> a, std::pair<int, int> b){
        DFGIONode* dfgIONodeA = dynamic_cast<DFGIONode*>(mapping->getDFG()->node(a.first));
        std::string memRefNameA = dfgIONodeA->memRefName(); 
        DFGIONode* dfgIONodeB = dynamic_cast<DFGIONode*>(mapping->getDFG()->node(b.first));
        std::string memRefNameB = dfgIONodeB->memRefName(); 
        return memRefNameA < memRefNameB;
    });
    // sort the load data commands according to dependence type
    std::stable_sort(ld_data_deps.begin(), ld_data_deps.end(), [](std::pair<int, int> a, std::pair<int, int> b){
        return a.second < b.second;
    });
    int i = 0;
    int ld_num = ld_data_deps.size();
    DFG *dfg = mapping->getDFG();
    for(auto &elem : ld_data_deps){
        DFGIONode* dfgIONode = dynamic_cast<DFGIONode*>(dfg->node(elem.first));
        int dataLen = dfgIONode->memSize();
        std::string memRefName = dfgIONode->memRefName(); 
        int offset = dfgIONode->memOffset();        
        // the next load command info
        int fused = 0; // fused with the next command
        if(i < ld_num - 1){
            DFGIONode* dfgIONodeNext = dynamic_cast<DFGIONode*>(dfg->node(ld_data_deps[i+1].first));
            int dataLenNext = dfgIONodeNext->memSize();
            std::string memRefNameNext = dfgIONodeNext->memRefName(); 
            int offsetNext = dfgIONodeNext->memOffset();
            if(memRefName == memRefNameNext && dataLen == dataLenNext && offset == offsetNext){
                fused = 1;
            }
        }
        if(offset > 0){
            memRefName = "(void*)" + memRefName + "+" + std::to_string(offset);
        }
        ldArrayNames.push_back(memRefName);
        // void* remoteAddr = ioName2AddrLen[memRefName].first;        
        auto& ioInfo = _dfg_io_infos[elem.first];
        os << "\tload_data(din_addr[" << i << "], 0x" << std::hex << ioInfo.addr << std::dec << ", " 
           << dataLen << ", " << fused << ", " << _task_id << ", " << ioInfo.dep << ");\n";
        i++;
    }
    os << "\tconfig(0x" << std::hex << cfgBaseAddrCtrl << std::dec << ", " << _cfg_num << ", " << _task_id << ", " << 0 << ");\n";
    os << "\texecute(0x" << std::hex << _iob_ens << std::dec << ", " << _task_id << ", " << _ex_dep << ");\n";
    i = 0;
    for(auto id : st_ids){
        DFGIONode* dfgIONode = dynamic_cast<DFGIONode*>(dfg->node(id));
        int dataLen = dfgIONode->memSize();
        std::string memRefName = dfgIONode->memRefName();
        int offset = dfgIONode->memOffset();
        if(offset > 0){
            memRefName = "(void*)" + memRefName + "+" + std::to_string(offset);
        }
        stArrayNames.push_back(memRefName);
        // void* remoteAddr = ioName2AddrLen[memRefName].first;
        // int dataLen = ioName2AddrLen[memRefName].second;
        auto& ioInfo = _dfg_io_infos[id];
        os << "\tstore(dout_addr[" << i << "], 0x" << std::hex << ioInfo.addr << std::dec << ", " 
           << dataLen << ", " << _task_id << ", " << 0 << ");\n";
        i++;
    }
    return std::make_pair(ldArrayNames, stArrayNames);
}



// analyze dependence, allocate spad space, dump execution call function
void IOScheduler::execute(Mapping *mapping, std::ostream &os_func, std::ostream &os_call)
{
    // dump execution call function head
    os_func << "void cgra_execute(void** din_addr, void** dout_addr)\n{\n";
    // analyze the dependence and allocate space in the spad
    ioSchedule(mapping);
    // update bank status
    int banks = _adg->numIobNodes();
    for (int i = 0; i < banks; i++) { 
        _older_bank_status[i] = _old_bank_status[i];
        _old_bank_status[i] = _cur_bank_status[i];        
    }
    /*
    Graphviz viz(mapping, "");
    // viz.drawDFG();
    // viz.drawADG();
    // viz.dumpDFGIO(); 
    viz.printDFGEdgePath();
    */

    // generate config data and dump to ostream
    genCfgData(mapping, os_func);
    // generate CGRA execution instructions and dump
    auto arrayNames = genInstructions(mapping, os_func);
    os_func << "}\n";
    int ldNum = arrayNames.first.size();
    int stNum = arrayNames.second.size();
    if(ldNum > 0){
        os_call << "void* cgra_din_addr[" << ldNum << "] = {";
        for(int i = 0; i < ldNum-1; i++){
            os_call << arrayNames.first[i] << ", ";
        }
        os_call << arrayNames.first[ldNum-1] << "};\n";
    }
    if(stNum > 0){
        os_call << "void* cgra_dout_addr[" << stNum << "] = {";
        for(int i = 0; i < stNum-1; i++){
            os_call << arrayNames.second[i] << ", ";
        }
        os_call << arrayNames.second[stNum-1] << "};\n";
    }
    os_call << "cgra_execute(cgra_din_addr, cgra_dout_addr);\n";
    _task_id++;
    // return arrayNames;
}