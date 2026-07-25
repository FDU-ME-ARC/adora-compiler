
#include "dfg/dfg.h"

DFG::DFG(){}

DFG::~DFG()
{
    for(auto& elem : _nodes){
        delete elem.second;
    }
    for(auto& elem : _edges){
        delete elem.second;
    }
}


DFGNode* DFG::node(int id){
    if(_nodes.count(id)){
        return _nodes[id];
    } else {
        return nullptr;
    }  
}


DFGEdge* DFG::edge(int id){
    if(_edges.count(id)){
        return _edges[id];
    } else {
        return nullptr;
    }  
}


void DFG::addNode(DFGNode* node){
    int id = node->id();
    _nodes[id] = node;
}


void DFG::addEdge(DFGEdge* edge){
    int id = edge->id();
    _edges[id] = edge;
    int srcId = edge->srcId();
    int dstId = edge->dstId();
    int srcPort = edge->srcPortIdx();
    int dstPort = edge->dstPortIdx();
    int bitWidth = edge->bitWidth();
    addBitWidth(bitWidth);
    if(srcId == _id){ // source is input port
        addInput(srcPort, std::make_pair(dstId, dstPort));
        addInputEdge(srcPort, id);
    } else {
        DFGNode* src = node(srcId);
        assert(src);
        src->addBitWidth(bitWidth);
        src->addOutput(srcPort, std::make_pair(dstId, dstPort));
        src->addOutputEdge(srcPort, id);
        src->addOutput(bitWidth, srcPort, std::make_pair(dstId, dstPort));
        src->addOutputEdge(bitWidth, srcPort, id);
    }
    if(dstId == _id){ // destination is output port
        addOutput(dstPort, std::make_pair(srcId, srcPort));
        addOutputEdge(dstPort, id);
    } else{        
        DFGNode* dst = node(dstId);
        assert(dst);
        dst->addBitWidth(bitWidth);
        dst->addInput(dstPort, std::make_pair(srcId, srcPort));
        dst->addInputEdge(dstPort, id);
        dst->addInput(bitWidth, dstPort, std::make_pair(srcId, srcPort));
        dst->addInputEdge(bitWidth, dstPort, id);
    }
}


void DFG::delNode(int id){
    DFGNode* dfgNode = node(id);
    for(auto& elem : dfgNode->inputEdges()){
        delEdge(elem.second);
    }
    for(auto& elem : dfgNode->outputEdges()){
        for(auto eid : elem.second){
            delEdge(eid);
        }        
    }
    _nodes.erase(id);
    delete dfgNode;
}


void DFG::delEdge(int id){
    DFGEdge* e = edge(id);
    int srcId = e->srcId();
    int dstId = e->dstId();
    int srcPortIdx = e->srcPortIdx();
    int dstPortIdx = e->dstPortIdx();
    if(srcId == _id){
        delInputEdge(srcPortIdx, id);
        delInput(srcPortIdx, std::make_pair(dstId, dstPortIdx));
    }else{
        DFGNode* srcNode = node(srcId);       
        srcNode->delOutputEdge(srcPortIdx, id);
        srcNode->delOutput(srcPortIdx, std::make_pair(dstId, dstPortIdx));
        srcNode->delOutputEdgefg(e->bitWidth(), srcPortIdx, id);
        srcNode->delOutput(e->bitWidth(), srcPortIdx, std::make_pair(dstId, dstPortIdx));
    }
    if(dstId == _id){
        delOutputEdge(dstPortIdx);
        delOutput(dstPortIdx);
    }else{
        DFGNode* dstNode = node(dstId);
        dstNode->delInputEdge(dstPortIdx);
        dstNode->delInput(dstPortIdx);
        dstNode->delInputEdge(e->bitWidth(), dstPortIdx);
        dstNode->delInput(e->bitWidth(), dstPortIdx);
    }
    _edges.erase(id);
    delete e;
}


// In nodes: INPUT/LOAD/CLOAD node
std::set<int> DFG::getInNodes(){
    std::set<int> inNodes;
    for(int ioNodeId : _ioNodes){
        DFGNode *ioNode = _nodes[ioNodeId];
        std::string opName = ioNode->operation();
        // if((opName == "INPUT") || (opName == "LOAD" && (ioNode->inputs().size() == 0))){
        if((opName == "INPUT") || (opName == "CINPUT") ||
           (opName == "LOAD") || (opName == "CLOAD")){
            inNodes.insert(ioNodeId);
        }
    }
    return inNodes;
}

// Out nodes: OUTPUT/STORE/CSTORE node
std::set<int> DFG::getOutNodes(){
    std::set<int> outNodes;
    for(int ioNodeId : _ioNodes){
        DFGNode *ioNode = _nodes[ioNodeId];
        std::string opName = ioNode->operation();
        if((opName == "OUTPUT") || (opName == "COUTPUT") ||
           (opName == "STORE") || (opName == "CSTORE")){
            outNodes.insert(ioNodeId);
        }
    }
    return outNodes;
}


// sort dfg nodes in topological order
// depth-first search
void DFG::dfs(DFGNode* node, std::map<int, bool>& visited){
    int nodeId = node->id();
    if(visited.count(nodeId) && visited[nodeId]){
        return; // already visited
    }
    visited[nodeId] = true;
    for(auto& in : node->inputs()){
        int inNodeId = in.second.first;
        if(inNodeId == _id){ // node connected to DFG input port
            continue;
        }
        dfs(_nodes[inNodeId], visited); // visit input node
    }
    _topoNodes.push_back(nodeId);
}

// sort dfg nodes in topological order
void DFG::topoSortNodes(){
    _topoNodes.clear();
    std::map<int, bool> visited; // node visited status
    for(auto& outNodeId : getOutNodes()){
        dfs(_nodes[outNodeId], visited); // visit output node
    }
}



// analyze I/O nodes accessing the same array
void DFG::analyzeMemDep(){
    std::map<std::string, std::vector<int>> name2nodes; // array-name, I/O node ID
    for(int id : _ioNodes){
        DFGIONode* ionode = dynamic_cast<DFGIONode*>(_nodes[id]);
        name2nodes[ionode->memRefName()].push_back(id);
    }
}


// ====== operators >>>>>>>>>>
// DFG copy
DFG& DFG::operator=(const DFG& that){
    if(this == &that) return *this;
    this->_id = that._id;
    this->_bitWidth = that._bitWidth;
    this->_bitWidths = that._bitWidths;
    this->_inputNames = that._inputNames;
    this->_outputNames = that._outputNames;
    this->_inputs = that._inputs;
    this->_outputs = that._outputs;
    this->_inputEdges = that._inputEdges;
    this->_outputEdges = that._outputEdges;
    this->_ioNodes = that._ioNodes;
    this->_lutNodes = that._lutNodes;
    this->_hasFineGrained = that._hasFineGrained;
    this->_cgWidth = that._cgWidth;
    this->_topoNodes = that._topoNodes;
    this->VariableConfigNodes = that.VariableConfigNodes;
    this->_backEdges = that._backEdges;
    this->_MII = that._MII;
    for(auto& elem : that._nodes){
        int id = elem.first;
        DFGNode* node;
        if(that._ioNodes.count(id)){
            DFGIONode *ioNode = new DFGIONode();
            *ioNode = *(dynamic_cast<DFGIONode*>(elem.second));
            node = ioNode;
        }else{
            node = new DFGNode();
            *node = *(elem.second);
        }        
        this->_nodes[id] = node;
    }
    // this->_edges = that._edges;
    for(auto& elem : that._edges){
        int id = elem.first;
        DFGEdge* edge = new DFGEdge();
        *edge = *(elem.second);
        this->_edges[id] = edge;
    }
    return *this;
}

void DFG::printVariableConfigNodes(){
    if(VariableConfigNodes.size() != 0){
        std::cout << "---------Print Variable Config Nodes------------\n";
        for(auto& elem : VariableConfigNodes){
            std::cout << "Node:\n";
            elem.first->print();
            std::cout << "VariableConfig:\n";
            elem.second.print();
            std::cout << std::endl;
        }
        std::cout << "---------End Print------------\n";
    }
}

void DFG::print(){
    printGraph();
    for(auto& elem : _nodes){
        elem.second->print();
    }
}



// detect loops based on backedge
void DFG::detectBackEdgeLoops(){
    _backEdges.clear();
    _backEdgesheadNodeId.clear();
     _backEdgeLoops.clear();
    for(auto elem : _edges){
        int curBackEdgeId = elem.first;
        DFGEdge *curBackEdge = elem.second;
        if(!curBackEdge->isBackEdge()){
            continue;
        }
        _backEdges.insert(curBackEdgeId);
        int headNodeId = curBackEdge->dstId(); // loop head
        int tailNodeId = curBackEdge->srcId(); // loop tail     
        std::vector<int> edgeStack;
        std::map<int, bool> visited;
        _backEdgesheadNodeId.insert(headNodeId);
        edgeStack.push_back(curBackEdgeId);
        while(!edgeStack.empty()){
            int topEid = *edgeStack.rbegin();
            DFGEdge *topEdge = _edges[topEid];
            int srcNodeId = topEdge->srcId();
            bool found = false;
            for(auto &elem : _nodes[srcNodeId]->inputEdges()){ // find next edges
                int eid = elem.second;
                DFGEdge *edge = _edges[eid];
                if(!edge->isBackEdge() && (!visited.count(eid) || !visited[eid])){
                    edgeStack.push_back(eid);                    
                    if(edge->srcId() == headNodeId){ // find a loop
                        auto loop = edgeStack;
                        std::reverse(loop.begin(), loop.end());                        
                        std::stringstream ss;
                        ss << "Detected a loop: ";
                        for(int loopedge : loop){
                            int loopSrcNodeId = _edges[loopedge]->srcId();
                            ss<<"eid_loop"<<loopedge;
                            ss << _nodes[loopSrcNodeId]->name() << " -> ";                           
                        }
                        ss << "<-";
                        spdlog::warn("{0}", ss.str()); 
                        loop.pop_back();
                        _backEdgeLoops[curBackEdgeId].push_back(loop); // record loop
                        visited[eid] = true;
                        edgeStack.pop_back();
                    }else{
                        found = true;
                        break;
                    }                    
                }
            }     
            if(!found){
                visited[*edgeStack.rbegin()] = true;
                edgeStack.pop_back();
            }  
        }
    }
    // set MII
    for(auto &elem: _backEdgeLoops){
        int iterDist = _edges[elem.first]->iterDist();    
        iterDist = std::max(iterDist, 1); // >= 1    
        for(auto &loop : elem.second){ // calculate loop latency only considering operation latency 
            int lat = _nodes[_edges[*loop.begin()]->srcId()]->opLatency();
            for(int eid : loop){
                //std::cout<<"eid"<<eid<<std::endl;
                lat += _nodes[_edges[eid]->dstId()]->opLatency();
            }
            _MII = std::max(_MII, (lat+iterDist-1)/iterDist);
        }
    }
    spdlog::warn("MII = {0}", _MII); 
}
