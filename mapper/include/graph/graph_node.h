#ifndef __GRAPH_NODE_H__
#define __GRAPH_NODE_H__

#include <iostream>
#include <map>
#include <vector>
#include <set>
#include <assert.h>


class GraphNode
{
protected:
    int _id = -1;
    std::string _name;
    std::string _type;
    int _bitWidth = 32;
    std::set<int> _bitWidths;
    std::map<int, std::pair<int, int>> _inputs; // <input-index, <node-id, node-port-idx>>
    std::map<int, std::set<std::pair<int, int>>> _outputs; // <output-index, set<node-id, node-port-idx>>
    std::map<int, int> _inputEdges; // <input-index, edge-id>
    std::map<int, std::set<int>> _outputEdges; // <output-index, set<edge-id>>
    // Width-qualified connectivity used by the mixed-grained mapper.  The
    // legacy maps above remain as a coarse-grained compatibility view.
    std::map<int, std::map<int, std::pair<int, int>>> _inputsByWidth;
    std::map<int, std::map<int, std::set<std::pair<int, int>>>> _outputsByWidth;
    std::map<int, std::map<int, int>> _inputEdgesByWidth;
    std::map<int, std::map<int, std::set<int>>> _outputEdgesByWidth;
public:
    GraphNode(){}
    ~GraphNode(){}
    int id(){ return _id; }
    void setId(int id){ _id = id; }
    std::string name();
    void setName(std::string name){ _name = name; }
    std::string type(){ return _type; }
    void setType(std::string type){ _type = type; }
    int bitWidth(){ return _bitWidth; }
    void setBitWidth(int bitWidth){ _bitWidth = bitWidth; _bitWidths.insert(bitWidth); }
    const std::set<int>& bitWidths(){ return _bitWidths; }
    void setBitWidths(const std::set<int>& bitWidths){ _bitWidths = bitWidths; }
    void addBitWidth(int bitWidth){ _bitWidths.insert(bitWidth); }
    void delBitWidth(int bitWidth){ _bitWidths.erase(bitWidth); }
    virtual int numInputs(){ return _inputs.size(); }
    int numOutputs(){ return _outputs.size(); }
    virtual int numInputs(int bitWidth);
    int numOutputs(int bitWidth);

    const std::map<int, std::pair<int, int>>& inputs(){ return _inputs; }
    const std::map<int, std::set<std::pair<int, int>>>& outputs(){ return _outputs; }
    std::pair<int, int> input(int index); // return <node-id, node-port-idx>
    std::set<std::pair<int, int>> output(int index); // return set<node-id, node-port-idx>
    void addInput(int index, std::pair<int, int> node);  // add input
    void addOutput(int index, std::pair<int, int> node); // add output
    void delInput(int index);  // delete input
    void delOutput(int index, std::pair<int, int> node); // delete output
    void delOutput(int index); // delete output
    const std::map<int, std::pair<int, int>>& inputs(int bitWidth);
    const std::map<int, std::set<std::pair<int, int>>>& outputs(int bitWidth);
    std::pair<int, int> input(int bitWidth, int index);
    std::set<std::pair<int, int>> output(int bitWidth, int index);
    void addInput(int bitWidth, int index, std::pair<int, int> node);
    void addOutput(int bitWidth, int index, std::pair<int, int> node);
    void delInput(int bitWidth, int index);
    void delOutput(int bitWidth, int index, std::pair<int, int> node);
    void delOutput(int bitWidth, int index);

    const std::map<int, int>& inputEdges(){ return _inputEdges; }
    const std::map<int, std::set<int>>& outputEdges(){ return _outputEdges; }
    int inputEdge(int index); // return edge id
    std::set<int> outputEdge(int index); // return set<edge-id>
    void addInputEdge(int index, int edgeId);  // add input edge
    void addOutputEdge(int index, int edgeId); // add output edge
    void delInputEdge(int index); // delete input edge
    void delOutputEdge(int index, int edgeId); // delete output edge
    void delOutputEdge(int index); // delete output edge
    const std::map<int, int>& inputEdges(int bitWidth);
    const std::map<int, std::set<int>>& outputEdges(int bitWidth);
    int inputEdge(int bitWidth, int index);
    std::set<int> outputEdge(int bitWidth, int index);
    void addInputEdge(int bitWidth, int index, int edgeId);
    void addOutputEdge(int bitWidth, int index, int edgeId);
    void delInputEdge(int bitWidth, int index);
    void delOutputEdgefg(int bitWidth, int index, int edgeId);
    void delOutputEdgefg(int bitWidth, int index);

    virtual void printGraphNode();
};





#endif
