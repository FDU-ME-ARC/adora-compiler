#ifndef __DFG_H__
#define __DFG_H__

#include "dfg/dfg_node.h"
#include "dfg/dfg_edge.h"
#include "graph/graph.h"
#include "spdlog/spdlog.h"


class DFG : public Graph
{
private:
    bool _hasFineGrained = false;
    int _cgWidth = 32;
    std::map<int, DFGNode*> _nodes;   // <node-id, node>
    std::map<int, DFGEdge*> _edges;   // <edge-id, edge>
    std::set<int> _ioNodes; // IO Node IDs, including INPUT, OUTPUT, LOAD, STORE, COAD, CSTORE nodes
    std::set<int> _lutNodes;
    DFG(const DFG&) = delete; // disable the default copy construct function

    std::map<int, std::vector<std::vector<int>>> _backEdgeLoops;

    int _MII = 1; // Minimum II
    std::set<int> _backEdges;
    std::set<int> _backEdgesheadNodeId;
protected:
    // DFG nodes in topological order, DFG should be DAG
    std::vector<int> _topoNodes;

    // depth-first search, sort dfg nodes in topological order
    void dfs(DFGNode* node, std::map<int, bool>& visited);

public:
    DFG();
    ~DFG();

    const std::map<int, DFGNode*>& nodes(){ return _nodes; }
    const std::map<int, DFGEdge*>& edges(){ return _edges; }
    DFGNode* node(int id);
    DFGEdge* edge(int id);
    void addNode(DFGNode* node);
    void addEdge(DFGEdge* edge);
    void delNode(int id);
    void delEdge(int id);

    const std::set<int>& ioNodes(){ return _ioNodes; }
    void addIONode(int id){ _ioNodes.insert(id); }
    void delIONode(int id){ _ioNodes.erase(id); }
    bool isIONode(int id){ return _ioNodes.count(id); }
    const std::set<int>& lutNodes(){ return _lutNodes; }
    void addLUTNode(int id){ _lutNodes.insert(id); }
    void delLUTNode(int id){ _lutNodes.erase(id); }
    bool isLUTNode(int id){ return _lutNodes.count(id); }
    void setFineGrained(bool fineGrained){ _hasFineGrained = fineGrained; }
    bool hasFineGrained(){ return _hasFineGrained; }
    void setCGWidth(int width){ _cgWidth = width; setBitWidth(width); }
    int CGWidth(){ return _cgWidth; }
    // In nodes: INPUT/LOAD/CLOAD node
    std::set<int> getInNodes();
    // Out nodes: OUTPUT/STORE/CSTORE node
    std::set<int> getOutNodes();

    // DFG nodes in topological order
    const std::vector<int>& topoNodes(){ return _topoNodes; }
    // sort dfg nodes in topological order
    void topoSortNodes();

    // analyze I/O nodes accessing the same array
    void analyzeMemDep();

    const std::map<int, std::vector<std::vector<int>>>& backEdgeLoops(){ return _backEdgeLoops; }
  
    // detect loops based on backedge
    void detectBackEdgeLoops();

    int MII(){ return _MII; }
    void setMII(int II) { _MII = II; }
    const std::set<int>& backEdges(){return _backEdges; }
    const std::set<int>& backEdgesheadNodeId(){return _backEdgesheadNodeId; }

    // ====== operators >>>>>>>>>>
    // DFG copy
    DFG& operator=(const DFG& that);

    void print();

    std::map<DFGNode*, VariableConfig> VariableConfigNodes;
    void printVariableConfigNodes();
};



#endif
