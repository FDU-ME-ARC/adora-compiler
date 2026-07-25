#include "adg/adg.h"

ADG::ADG(){}

ADG::~ADG()
{
    // std::cout << "delete adg" << std::endl;
    // std::cout << "numGpeNodes: " << _numGpeNodes << std::endl;
    // std::cout << "numIobNodes: " << _numIobNodes << std::endl;
    for(auto& elem : _nodes){
        auto node = elem.second;
        node->print();
        auto sub_adg = node->subADG();
        if(sub_adg){
            delete sub_adg;
        }
        delete node;
    }
    for(auto& elem : _edges){
        delete elem.second;
    }
}


ADGNode* ADG::node(int id){
    if(_nodes.count(id)){
        return _nodes[id];
    } else {
        return nullptr;
    }  
}


ADGEdge* ADG::edge(int id){
    if(_edges.count(id)){
        return _edges[id];
    } else {
        return nullptr;
    }  
}


void ADG::addNode(int id, ADGNode* node){
    _nodes[id] = node;
}


void ADG::addEdge(int id, ADGEdge* edge){
    _edges[id] = edge;
    int srcId = edge->srcId();
    int dstId = edge->dstId();
    int srcPort = edge->srcPortIdx();
    int dstPort = edge->dstPortIdx();
    int bitWidth = edge->bitWidth();
    addBitWidth(bitWidth);
    if(srcId == _id){ // source is input port
        addInput(srcPort, std::make_pair(dstId, dstPort));
    } else {
        ADGNode* src = node(srcId);
        assert(src);
        src->addOutput(srcPort, std::make_pair(dstId, dstPort));
        src->addOutput(bitWidth, srcPort, std::make_pair(dstId, dstPort));
    }
    if(dstId == _id){ // destination is output port
        addOutput(dstPort, std::make_pair(srcId, srcPort));
    } else{        
        ADGNode* dst = node(dstId);
        assert(dst);
        dst->addInput(dstPort, std::make_pair(srcId, srcPort));
        dst->addInput(bitWidth, dstPort, std::make_pair(srcId, srcPort));
    }
}


void ADG::delNode(int id){
    ADGNode *adgNode = node(id);
    _nodes.erase(id);
    delete adgNode;
}


void ADG::delEdge(int id){
    ADGEdge *adgEdge = edge(id);
    _edges.erase(id);
    delete adgEdge;
}


ADG& ADG::operator=(const ADG& that){
    if(this == &that) return *this;
    this->_id = that._id;
    this->_bitWidth = that._bitWidth;
    this->_bitWidths = that._bitWidths;
    this->_numGpeNodes = that._numGpeNodes;
    this->_numIobNodes = that._numIobNodes;
    this->_cfgDataWidth = that._cfgDataWidth;
    this->_cfgAddrWidth = that._cfgAddrWidth;
    this->_cfgBlkOffset = that._cfgBlkOffset;
    this->_maxLUTInput = that._maxLUTInput;
    this->_fgCfgBaseBlock = that._fgCfgBaseBlock;
    this->_fgCfgBlockCount = that._fgCfgBlockCount;
    // this->_loadLatency = that._loadLatency;
    // this->_storeLatency = that._storeLatency;
    this->_cfgSpadSize = that._cfgSpadSize;
    this->_iobAgNestLevels = that._iobAgNestLevels;
    this->_iobSpadBankSize = that._iobSpadBankSize;
    this->_iobToSpadBanks = that._iobToSpadBanks;
    this->_cfgBits = that._cfgBits;
    this->_inputs = that._inputs;
    this->_outputs = that._outputs;
    this->_inputsByWidth = that._inputsByWidth;
    this->_outputsByWidth = that._outputsByWidth;
    // this->_input_used = that._input_used;
    // this->_output_used = that._output_used;
    for(auto& elem : that._nodes){
        ADGNode* node = new ADGNode();
        *node = *(elem.second);
        this->_nodes[elem.first] = node;
    }
    for(auto& elem : that._edges){
        ADGEdge* edge = new ADGEdge();
        *edge = *(elem.second);
        this->_edges[elem.first] = edge;
    }
    return *this;
}

inline bool isTileInFirstN(int tileId, int neededTiles) {
    assert(tileId >= 0);
    assert(neededTiles > 0);
    return tileId < neededTiles;
}

ADGNode* subADGNodeClone(ADGNode* from) {
    ADGNode* to;
    // assert(from && to);

    // ---------- 1) GIBNode specific ----------
    if (from->type() == "GIB") {
        auto bTo   = new GIBNode();
        auto bFrom = dynamic_cast<GIBNode*>(from);
        assert(bFrom);
        *bTo = *bFrom;

        for (int o = 0; o < bFrom->numOutputs(); ++o) {
            bTo->setOutReged(o, bFrom->outReged(o));
        }
        // for (int i = 0; i < numInputs; ++i) {
        //     for (int out : bFrom->in2outs(i)) bTo->addIn2outs(i, out);
        // }
        
        to = dynamic_cast<ADGNode*>(bTo);
    }

    // ---------- 2) FUNode（GPE/IOB's father class） ----------
    else if (auto fFrom = dynamic_cast<FUNode*>(from)) {
        FUNode* fTo;
        // ---------- 3) GPENode specific ----------
        if (from->type() == "GPE") {
            auto gFrom = dynamic_cast<GPENode*>(from);
            GPENode* gTo = new GPENode();
            assert(gFrom && gTo);
            *gTo = *gFrom;

            fTo = dynamic_cast<FUNode*>(gTo);
        }

        // ---------- 4) IOBNode specific ----------
        else if (from->type() == "IOB") {
            auto iFrom = dynamic_cast<IOBNode*>(from);
            IOBNode* iTo = new IOBNode();
            assert(iFrom && iTo);
            *iTo = *iFrom;

            fTo = dynamic_cast<FUNode*>(iTo);
        }

        fTo->setMaxDelay(fFrom->maxDelay());
        fTo->setNumOperands(fFrom->numOperands());

        for (const auto& op : fFrom->operations()) {
            fTo->addOperation(op);
        }

        // for (int oi = 0; oi < fFrom->numOperands(); ++oi) {
        //     const auto& ins = fFrom->operandInputs(oi);
        //     for (int inPort : ins) {
        //         fTo->addOperandInputs(oi, inPort);
        //     }
        // }

        // cfgIdMap（public）
        fTo->cfgIdMap = fFrom->cfgIdMap;

        to = dynamic_cast<ADGNode*>(fTo);
    }

    // ---------- 5) General copy ----------
    to->setCfgBlkIdx(from->cfgBlkIdx());
    to->setTile(from->tile());
    to->setX(from->x());
    to->setY(from->y());
    to->setId(from->id());
    to->setName(from->name());
    to->setType(from->type());
    to->setBitWidth(from->bitWidth());

    for (const auto& kv : from->configInfo()) {
        to->addConfigInfo(kv.first, kv.second);
    }

    if (from->subADG()) {
        ADG* sub = new ADG();
        *sub = *(from->subADG());   
        to->setSubADG(sub);
    }
    return to; 
}

void subADGEdgeCopy(ADGEdge* from, ADGEdge* to){
    to->setId(from->id());
    to->setEdge(from->bitWidth(), from->srcId(), from->srcPortIdx(),
                from->dstId(), from->dstPortIdx());
}

ADG* ADG::inducedSubgraphByFirstNTiles(size_t n) {
    // assert(n <= this->_tileNum);
    // if (n == 1 && this->_tileNum == 1) return this;
    if (n == 9999999 && this->_tileNum > 0){
        n = this->_tileNum;
    } 
    // else{
    //     return new ADG();
    // }

    auto sub = new ADG();
    sub->_bitWidth       = this->_bitWidth;
    sub->_bitWidths      = this->_bitWidths;
    sub->_cfgDataWidth   = this->_cfgDataWidth;
    sub->_cfgAddrWidth   = this->_cfgAddrWidth;
    sub->_cfgBlkOffset   = this->_cfgBlkOffset;
    sub->_cfgSpadSize    = this->_cfgSpadSize;
    sub->_cfgSpadDataWidth = this->_cfgSpadDataWidth;
    sub->_iobAgNestLevels= this->_iobAgNestLevels;
    sub->_iobSpadBankSize= this->_iobSpadBankSize;
    sub->_iobToSpadBanks = this->_iobToSpadBanks;
    sub->_cfgBits        = this->_cfgBits;
    sub->_tileNum        = n;
    sub->_maxLUTInput    = this->_maxLUTInput;
    sub->_fgCfgBaseBlock = this->_fgCfgBaseBlock;
    sub->_fgCfgBlockCount= this->_fgCfgBlockCount;


    std::unordered_set<int> keep; 
    keep.reserve(_nodes.size());
    for (const auto& kv : this->nodes()) {
        const int nid = kv.first;
        ADGNode* nd = kv.second;
        int t = nd->tile();
        if ((t >= 0 && isTileInFirstN(t, n)) || t == -1) { /// -1 means not a multi-tile cgra
            ADGNode* clone = subADGNodeClone(nd);        
            sub->addNode(nid, clone);  
            keep.insert(nid);
        }
    }

    sub->_numGpeNodes = 0;
    sub->_numIobNodes = 0;
    for (const auto& kv : sub->_nodes) {
        ADGNode* nd = kv.second;
        if (nd->type() == "GPE") ++sub->_numGpeNodes;
        if (nd->type() == "IOB") ++sub->_numIobNodes;
    }

    for (const auto& kv : _edges) {
        const int eid = kv.first;
        ADGEdge* e = kv.second;

        int srcId = e->srcId();
        int dstId = e->dstId();

        if (keep.count(srcId) && keep.count(dstId)) {
            ADGEdge* ce = new ADGEdge();
            // *ce = *e;           
            subADGEdgeCopy(e, ce);      
            sub->addEdge(eid, ce);   
        }
    }

    /// prune nodes's connection To first n Tiles
    for(auto& nid : keep){
        ADGNode* nd = sub->node(nid);
        auto inputs = nd->inputs();  // <input-index, <node-id, node-port-idx>>
        for(auto pair : inputs){
            auto in_idx = pair.first;
            auto in_node = pair.second.first;
            if(keep.count(in_node) == 0){
                // do not belong to needed tile
                nd->delInput(in_idx);
            }
        }
        auto outputs = nd->outputs();  // <output-index, set<node-id, node-port-idx>>
        for(auto pair : outputs){
            auto out_idx = pair.first;
            auto out_nodes = pair.second;
            for(auto out_node : out_nodes){
                if(keep.count(out_node.first) == 0){
                    // do not belong to needed tile
                    nd->delOutput(out_idx, /*std::pair*/out_node);
                }
            }
        }
        for(int bitWidth : nd->bitWidths()){
            auto widthInputs = nd->inputs(bitWidth);
            for(auto pair : widthInputs){
                if(!keep.count(pair.second.first)) nd->delInput(bitWidth, pair.first);
            }
            auto widthOutputs = nd->outputs(bitWidth);
            for(auto pair : widthOutputs){
                for(auto outNode : pair.second){
                    if(!keep.count(outNode.first))
                        nd->delOutput(bitWidth, pair.first, outNode);
                }
            }
        }

        if(nd -> type() == "GPE" || nd -> type() == "IOB"){
            FUNode* fFrom = dynamic_cast<FUNode*>(this->node(nid));
            FUNode* f = dynamic_cast<FUNode*>(nd);
            for(int operand = 0; operand < fFrom->numOperands(); operand++){
                std::set<int> finputs = fFrom->operandInputs(operand);
                for(auto finput : finputs){
                    if(nd->inputs().count(finput)){
                        f->addOperandInputs(operand, finput);
                    }
                }
            }
        }
        else if(nd -> type() == "GIB"){
            auto gib = dynamic_cast<GIBNode*>(nd);
            auto bFrom = dynamic_cast<GIBNode*>(this->node(nid));
            for (int o = 0; o < bFrom->numOutputs(); ++o) {
                for (int in : bFrom->out2ins(o)) gib->addOut2ins(o, in);
            }
            for (int i = 0; i < bFrom->numInputs(); ++i) {
                for (int out : bFrom->in2outs(i)) gib->addIn2outs(i, out);
            }
        }
    }

    // //// compare two adg:
    // std::cout << "original adg 2: " << std::endl;
    // this->print();
    // std::cout << "suv adg: " << std::endl;
    // sub->print();
    return sub;
}

void ADG::print(){
    printGraph();
    std::cout << "numGpeNodes: " << _numGpeNodes << std::endl;
    std::cout << "numIobNodes: " << _numIobNodes << std::endl;
    std::cout << "cfgDataWidth: " << _cfgDataWidth << std::endl;
    std::cout << "cfgAddrWidth: " << _cfgAddrWidth << std::endl;
    std::cout << "cfgBlkOffset: " << _cfgBlkOffset << std::endl;
    std::cout << "cfgSpadSize: " << _cfgSpadSize << std::endl;
    std::cout << "iobAgNestLevels: " << _iobAgNestLevels << std::endl;
    std::cout << "iobSpadBankSize: " << _iobSpadBankSize << std::endl;
    std::cout << "iobToSpadBanks: " << std::endl;   
    for(auto& elem : _iobToSpadBanks){
        std::cout << "IOB(" << elem.first << "): ";
        for(auto bank : elem.second)
            std::cout << bank << " ";
        std::cout << std::endl;
    }
    std::cout << "cfgBits: ";
    for(auto& elem : _cfgBits){
        std::cout << elem << " ";
    }
    std::cout << std::endl;
    for(auto& elem : _nodes){
        elem.second->print();
    }
}
