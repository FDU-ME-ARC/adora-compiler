#ifndef __GRAPH_EDGE_H__
#define __GRAPH_EDGE_H__

#include <iostream>
#include <map>
#include <vector>
#include <set>
#include <assert.h>


class GraphEdge
{
private:
    int _id;
    // Width is carried by every edge so coarse-grained data and one-bit
    // predicate/LUT traffic can coexist in the same graph.
    int _bitWidth = 32;
    int _srcPortIdx; // source node I/O port index
    int _dstPortIdx; // destination node I/O port index
    int _srcId;   // source node ID
    int _dstId;   // destination node ID
public:
    GraphEdge(){}
    GraphEdge(int edgeId){ _id = edgeId; }
    GraphEdge(int srcId, int dstId) : _srcId(srcId), _dstId(dstId) {}
    ~GraphEdge(){}
    int id(){ return _id; }
    void setId(int id){ _id = id; }
    int bitWidth(){ return _bitWidth; }
    void setBitWidth(int bitWidth){ _bitWidth = bitWidth; }
    int srcPortIdx(){ return _srcPortIdx; }
    void setSrcPortIdx(int srcPortIdx){ _srcPortIdx = srcPortIdx; }
    int dstPortIdx(){ return _dstPortIdx; }
    void setDstPortIdx(int dstPortIdx){ _dstPortIdx = dstPortIdx; }
    int srcId(){ return _srcId; }
    void setSrcId(int srcId){ _srcId = srcId; }
    int dstId(){ return _dstId; }
    void setDstId(int dstId){ _dstId = dstId; }
    void setEdge(int srcId, int dstId){
        _srcId = srcId;
        _dstId = dstId;
    }
    void setEdge(int srcId, int srcPort, int dstId, int dstPort){
        _srcId = srcId;
        _srcPortIdx = srcPort;
        _dstId = dstId;
        _dstPortIdx = dstPort;
    }
    void setEdge(int bitWidth, int srcId, int srcPort, int dstId, int dstPort){
        _bitWidth = bitWidth;
        setEdge(srcId, srcPort, dstId, dstPort);
    }
};





#endif
