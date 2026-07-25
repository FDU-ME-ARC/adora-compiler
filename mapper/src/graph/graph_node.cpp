
#include "graph/graph_node.h"

// ===================================================
//   GraphNode functions
// ===================================================

std::string GraphNode::name(){ 
    if(!_name.empty()){
        return _name; 
    }else{
        return _type + std::to_string(_id);
    }
}

int GraphNode::numInputs(int bitWidth){
    return _inputsByWidth.count(bitWidth) ? _inputsByWidth[bitWidth].size() : 0;
}

int GraphNode::numOutputs(int bitWidth){
    return _outputsByWidth.count(bitWidth) ? _outputsByWidth[bitWidth].size() : 0;
}

const std::map<int, std::pair<int, int>>& GraphNode::inputs(int bitWidth){
    static const std::map<int, std::pair<int, int>> empty;
    auto it = _inputsByWidth.find(bitWidth);
    return it == _inputsByWidth.end() ? empty : it->second;
}

const std::map<int, std::set<std::pair<int, int>>>& GraphNode::outputs(int bitWidth){
    static const std::map<int, std::set<std::pair<int, int>>> empty;
    auto it = _outputsByWidth.find(bitWidth);
    return it == _outputsByWidth.end() ? empty : it->second;
}

std::pair<int, int> GraphNode::input(int bitWidth, int index){
    auto widthIt = _inputsByWidth.find(bitWidth);
    if(widthIt == _inputsByWidth.end()) return {};
    auto it = widthIt->second.find(index);
    return it == widthIt->second.end() ? std::pair<int, int>{} : it->second;
}

std::set<std::pair<int, int>> GraphNode::output(int bitWidth, int index){
    auto widthIt = _outputsByWidth.find(bitWidth);
    if(widthIt == _outputsByWidth.end()) return {};
    auto it = widthIt->second.find(index);
    return it == widthIt->second.end() ? std::set<std::pair<int, int>>{} : it->second;
}

void GraphNode::addInput(int bitWidth, int index, std::pair<int, int> node){
    _bitWidths.insert(bitWidth);
    _inputsByWidth[bitWidth][index] = node;
}

void GraphNode::addOutput(int bitWidth, int index, std::pair<int, int> node){
    _bitWidths.insert(bitWidth);
    _outputsByWidth[bitWidth][index].emplace(node);
}

void GraphNode::delInput(int bitWidth, int index){
    if(_inputsByWidth.count(bitWidth)) _inputsByWidth[bitWidth].erase(index);
}

void GraphNode::delOutput(int bitWidth, int index, std::pair<int, int> node){
    if(_outputsByWidth.count(bitWidth) && _outputsByWidth[bitWidth].count(index))
        _outputsByWidth[bitWidth][index].erase(node);
}

void GraphNode::delOutput(int bitWidth, int index){
    if(_outputsByWidth.count(bitWidth)) _outputsByWidth[bitWidth].erase(index);
}


// return <node-id, node-port-idx>
std::pair<int, int> GraphNode::input(int index){
    if(_inputs.count(index)){
        return _inputs[index];
    }else{
        return {}; // std::make_pair(-1, -1);
    }
}

// return set<node-id, node-port-idx>
std::set<std::pair<int, int>> GraphNode::output(int index){
    if(_outputs.count(index)){
        return _outputs[index];
    }else{
        return {}; // return empty set
    }
}


// add input
void GraphNode::addInput(int index, std::pair<int, int> node){
    _inputs[index] = node;
}

// add output
void GraphNode::addOutput(int index, std::pair<int, int> node){
    _outputs[index].emplace(node);
}

// delete input
void GraphNode::delInput(int index){
    _inputs.erase(index);
}

// delete output
void GraphNode::delOutput(int index, std::pair<int, int> node){
    _outputs[index].erase(node);
}

// delete output
void GraphNode::delOutput(int index){
    _outputs.erase(index);
}

// return edge id
int GraphNode::inputEdge(int index){
    if(_inputEdges.count(index)){
        return _inputEdges[index];
    }else{
        return -1; 
    }
}

// return set<edge-id>
std::set<int> GraphNode::outputEdge(int index){
    if(_outputEdges.count(index)){
        return _outputEdges[index];
    }else{
        return {}; // return empty set
    }
}

// add input edge
void GraphNode::addInputEdge(int index, int edgeId){
    _inputEdges[index] = edgeId;
}

// add output edge
void GraphNode::addOutputEdge(int index, int edgeId){
    _outputEdges[index].emplace(edgeId);
}

// delete input edge
void GraphNode::delInputEdge(int index){
    _inputEdges.erase(index);
}

// delete output edge
void GraphNode::delOutputEdge(int index, int edgeId){
    assert(_outputEdges.count(index));
    _outputEdges[index].erase(edgeId);
}

// delete output edge
void GraphNode::delOutputEdge(int index){
    _outputEdges.erase(index);
}

const std::map<int, int>& GraphNode::inputEdges(int bitWidth){
    static const std::map<int, int> empty;
    auto it = _inputEdgesByWidth.find(bitWidth);
    return it == _inputEdgesByWidth.end() ? empty : it->second;
}

const std::map<int, std::set<int>>& GraphNode::outputEdges(int bitWidth){
    static const std::map<int, std::set<int>> empty;
    auto it = _outputEdgesByWidth.find(bitWidth);
    return it == _outputEdgesByWidth.end() ? empty : it->second;
}

int GraphNode::inputEdge(int bitWidth, int index){
    auto widthIt = _inputEdgesByWidth.find(bitWidth);
    if(widthIt == _inputEdgesByWidth.end()) return -1;
    auto it = widthIt->second.find(index);
    return it == widthIt->second.end() ? -1 : it->second;
}

std::set<int> GraphNode::outputEdge(int bitWidth, int index){
    auto widthIt = _outputEdgesByWidth.find(bitWidth);
    if(widthIt == _outputEdgesByWidth.end()) return {};
    auto it = widthIt->second.find(index);
    return it == widthIt->second.end() ? std::set<int>{} : it->second;
}

void GraphNode::addInputEdge(int bitWidth, int index, int edgeId){
    _bitWidths.insert(bitWidth);
    _inputEdgesByWidth[bitWidth][index] = edgeId;
}

void GraphNode::addOutputEdge(int bitWidth, int index, int edgeId){
    _bitWidths.insert(bitWidth);
    _outputEdgesByWidth[bitWidth][index].emplace(edgeId);
}

void GraphNode::delInputEdge(int bitWidth, int index){
    if(_inputEdgesByWidth.count(bitWidth)) _inputEdgesByWidth[bitWidth].erase(index);
}

void GraphNode::delOutputEdgefg(int bitWidth, int index, int edgeId){
    if(_outputEdgesByWidth.count(bitWidth) && _outputEdgesByWidth[bitWidth].count(index))
        _outputEdgesByWidth[bitWidth][index].erase(edgeId);
}

void GraphNode::delOutputEdgefg(int bitWidth, int index){
    if(_outputEdgesByWidth.count(bitWidth)) _outputEdgesByWidth[bitWidth].erase(index);
}


void GraphNode::printGraphNode(){
    // std::cout << "=====================================\n";
    // std::cout << "id: " << _id << std::endl;
    // std::cout << "type: " << _type << std::endl;
    // std::cout << "name: " << _name << std::endl;
    // std::cout << "bitWidth: " << _bitWidth << std::endl;
    // std::cout << "numInputs: " << numInputs() << std::endl;
    // std::cout << "numOutputs: " << numOutputs() << std::endl;
    // std::cout << "inputs: " << std::endl;
    // for(auto& elem : _inputs){
    //     std::cout << elem.first << ": (" << elem.second.first << ", " << elem.second.second << ")\n";
    // }
    // std::cout << "outputs: " << std::endl;
    // for(auto& elem : _outputs){
    //     std::cout << elem.first << ": ";
    //     auto& s = elem.second;
    //     for(auto it = s.begin(); it != s.end(); it++){
    //         std::cout << "(" << it->first << ", " << it->second << ") ";
    //     }
    //     std::cout << std::endl;
    // }
}
