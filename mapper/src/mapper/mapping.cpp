
#include "mapper/mapping.h"


// reset mapping intermediate result and status
void Mapping::reset(){
    _dfgNodeAttr.clear();
    _dfgEdgeAttr.clear();
    // _dfgInputAttr.clear();
    // _dfgOutputAttr.clear();
    _adgNodeAttr.clear();    
    // _adgLinkAttr.clear();
    _fuDelayAttr.clear();
    _vioDfgEdges.clear();
    // _totalViolation = 0;
    _numNodeMapped = 0;
}


// if this input port of this ADG node is used
bool Mapping::isAdgNodeInPortUsed(int nodeId, int portIdx){
    if(_adgNodeAttr.count(nodeId)){
        auto& status = _adgNodeAttr[nodeId].inPortUsed;
        if(status.count(portIdx)){
            return status[portIdx];
        }
    }
    return false;
}

bool Mapping::isAdgNodeInPortUsed(int nodeId, int bitWidth, int portIdx){
    if(_adgNodeAttr.count(nodeId)){
        auto& status = _adgNodeAttr[nodeId].inPortUsedByWidth[bitWidth];
        auto it = status.find(portIdx);
        return it != status.end() && it->second;
    }
    return false;
}


// if this output port of this ADG node is used
bool Mapping::isAdgNodeOutPortUsed(int nodeId, int portIdx){
    if(_adgNodeAttr.count(nodeId)){
        auto& status = _adgNodeAttr[nodeId].outPortUsed;
        if(status.count(portIdx)){
            return status[portIdx];
        }
    }
    return false;
}

bool Mapping::isAdgNodeOutPortUsed(int nodeId, int bitWidth, int portIdx){
    if(_adgNodeAttr.count(nodeId)){
        auto& status = _adgNodeAttr[nodeId].outPortUsedByWidth[bitWidth];
        auto it = status.find(portIdx);
        return it != status.end() && it->second;
    }
    return false;
}


// if the DFG node is already mapped
bool Mapping::isMapped(DFGNode* dfgNode){
    if(_dfgNodeAttr.count(dfgNode->id())){
        return _dfgNodeAttr[dfgNode->id()].adgNode != nullptr;
    }
    return false;
    
}

// if the ADG node is already mapped
bool Mapping::isMapped(ADGNode* adgNode){
    if(_adgNodeAttr.count(adgNode->id())){
        return _adgNodeAttr[adgNode->id()].dfgNode != nullptr;
    }
    return false;
}

// if the IOB node is available 
bool Mapping::isIOBFree(ADGNode* adgNode){
    return !isMapped(adgNode);
    // // assert(adgNode->type() == "IOB");
    // int id = adgNode->id();
    // if(_adgNodeAttr.count(id)){
    //     for(auto& elem : _adgNodeAttr[id].inPortUsed){
    //         if(elem.second){ // used as IB
    //             return false;
    //         }
    //     }
    //     for(auto& elem : _adgNodeAttr[id].outPortUsed){
    //         if(elem.second){ // used as OB
    //             return false;
    //         }
    //     }
    // }
    // return true;
}


// // if the IOB node is already used as IB 
// bool Mapping::isUsedAsIB(ADGNode* adgNode){
//     // assert(adgNode->type() == "IOB");
//     int id = adgNode->id();
//     if(_adgNodeAttr.count(id)){
//         for(auto& elem : _adgNodeAttr[id].inPortUsed){
//             if(elem.second){ // used as IB
//                 return true;
//             }
//         }
//     }
//     return false;
// }

// // if the IOB node is already used as OB 
// bool Mapping::isUsedAsOB(ADGNode* adgNode){
//     // assert(adgNode->type() == "IOB");
//     int id = adgNode->id();
//     if(_adgNodeAttr.count(id)){
//         for(auto& elem : _adgNodeAttr[id].outPortUsed){
//             if(elem.second){ // used as OB
//                 return true;
//             }
//         }
//     }
//     return false;
// }

// // if the IB node is available 
// bool Mapping::isIBAvail(ADGNode* adgNode){
//     assert(adgNode->type() == "IB");
//     int id = adgNode->id();
//     if(_adgNodeAttr.count(id)){
//         auto& inPortUsed = _adgNodeAttr[id].inPortUsed;
//         for(int i = 0; i < adgNode->numInputs(); i++){
//             if(!inPortUsed.count(i)){
//                 return true;
//             }else if(!inPortUsed[i]){
//                 return true;
//             }
//         }
//         return false;
//     }
//     return true;
// }

// // if the OB node is available 
// bool Mapping::isOBAvail(ADGNode* adgNode){
//     assert(adgNode->type() == "OB");
//     int id = adgNode->id();
//     if(_adgNodeAttr.count(id)){
//         auto& outPortUsed = _adgNodeAttr[id].outPortUsed;
//         for(int i = 0; i < adgNode->numOutputs(); i++){
//             if(!outPortUsed.count(i) || !outPortUsed[i]){
//                 return true;
//             }
//         }
//         return false;
//     }
//     return true;
// }

// // if the DFG input port is mapped
// bool Mapping::isDfgInputMapped(int idx){
//     if(_dfgInputAttr.count(idx) && !_dfgInputAttr[idx].routedEdgeIds.empty()){
//         return true;
//     }
//     return false;
// }

// // if the DFG output port is mapped
// bool Mapping::isDfgOutputMapped(int idx){
//     if(_dfgOutputAttr.count(idx) && !_dfgOutputAttr[idx].routedEdgeIds.empty()){
//         return true;
//     }
//     return false;
// }

// occupied ADG node of this DFG node
ADGNode* Mapping::mappedNode(DFGNode* dfgNode){
    int dfgNodeId = dfgNode->id();
    if(_dfgNodeAttr.count(dfgNodeId)){
        return _dfgNodeAttr[dfgNodeId].adgNode;
    }else{
        return nullptr;
    } 
}


// mapped DFG node of this ADG node
DFGNode* Mapping::mappedNode(ADGNode* adgNode){
    int adgNodeId = adgNode->id();
    if(_adgNodeAttr.count(adgNodeId)){
        return _adgNodeAttr[adgNodeId].dfgNode;
    }else{
        return nullptr;
    }
}


// just map DFG node to ADG node without routing the input/output edges
bool Mapping::mapDfgNodeNoRoute(DFGNode* dfgNode, ADGNode* adgNode){
    int dfgNodeId = dfgNode->id();
    // if(_dfgNodeAttr.count(dfgNodeId)){
    //     if(_dfgNodeAttr[dfgNodeId].adgNode == adgNode){
    //         return true;
    //     } else if(_dfgNodeAttr[dfgNodeId].adgNode != nullptr){
    //         return false;
    //     }
    // } 
    _dfgNodeAttr[dfgNodeId].adgNode = adgNode;
    // constant are only processed in configuration stage since it does not affect mapping
    // one operand is immediate and the operands are not commutative
    // if(dfgNode->hasImm() && !dfgNode->commutative()){ // assign constant port
    //     GPENode* gpeNode = dynamic_cast<GPENode*>(adgNode);
    //     for(auto inPort : gpeNode->operandInputs(dfgNode->immIdx())){
    //         if(adgAttr.inPortUsed.count(inPort) && adgAttr.inPortUsed[inPort] == true){ // already used
    //             return false;
    //         }
    //         adgAttr.inPortUsed[inPort] = true; // set all the input ports connetced to this operand as used
    //     }
    // }
    _adgNodeAttr[adgNode->id()].dfgNode = dfgNode;
    _numNodeMapped++;
    spdlog::debug("Map DFG node {0} to ADG node {1}", dfgNode->name(), adgNode->name()); 
    return true;
}


// map DFG node to ADG node and route the input/output edges connected to already-mapped DFG nodes
// map the connected input/output nodes alongside
bool Mapping::mapDfgNode(DFGNode* dfgNode, ADGNode* targetAdgNode){
    bool succeed = true;
    // if(_dfg->isIONode(dfgNode->id())){ // pre-map IO node to prevent duplicate mapping during mapping the connected IO nodes
    //     succeed = mapDfgNodeNoRoute(dfgNode, targetAdgNode);
    // }
    // assert(succeed);
    std::vector<DFGEdge*> routedEdges;
    // std::vector<DFGNode*> ioNodes; // newly mapped IO nodes
    // std::vector<DFGEdge*> inEdges;  // edges connected to DFG input node
    // std::vector<DFGEdge*> outEdges; // edges connected to DFG output node
    // route the src edges whose src nodes have been mapped or connected to DFG input node
    for(auto& edgePair : dfgNode->inputEdges()){
        DFGEdge* edge = _dfg->edge(edgePair.second);
        int inNodeId = edge->srcId();
        DFGNode* inNode = _dfg->node(inNodeId);
        ADGNode* adgNode = mappedNode(inNode);
        bool routed = false;
        if(adgNode){
            succeed = routeDfgEdge(edge, adgNode, targetAdgNode); // route edge between targetAdgNode and adgNode       
            routed = true;          
        // }else if(_dfg->isIONode(inNodeId)){ // connected to DFG input node which has not been placed
        //     succeed = routeDfgEdge(edge, targetAdgNode, true); // route edge between targetAdgNode and ADG IOB
        //     ioNodes.push_back(inNode); // cache newly mapped IO nodes
        //     // inEdges.push_back(edge);
        //     routed = true; 
        }
        if(!succeed){
            break;
        }else if(routed){
            routedEdges.push_back(edge); // cache the routed edge
        }
    }
    if(succeed){
        // route the dst edge whose dst nodes have been mapped or connected to DFG output port
        for(auto& elem : dfgNode->outputEdges()){
            for(int outEdgeId : elem.second){
                DFGEdge* edge = _dfg->edge(outEdgeId);
                int outNodeId = edge->dstId();
                DFGNode* outNode = _dfg->node(outNodeId);
                ADGNode* adgNode = mappedNode(outNode);
                bool routed = false;
                if(adgNode){
                    succeed = routeDfgEdge(edge, targetAdgNode, adgNode); // route edge between targetAdgNode and adgNode  
                    routed = true;                      
                // }else if(_dfg->isIONode(outNodeId)){ // connected to DFG output node which has not been placed
                //     succeed = routeDfgEdge(edge, targetAdgNode, false); // route edge between targetAdgNode and ADG IOB
                //     ioNodes.push_back(outNode); // cache newly mapped IO nodes
                //     // outEdges.push_back(edge);
                //     routed = true; 
                }
                if(!succeed){
                    break;
                }else if(routed){
                    routedEdges.push_back(edge); // cache the routed edge
                }
            } 
            if(!succeed) break;  
        }
    }
    // if(succeed){
    //     int i = 0;
    //     while(i < ioNodes.size()){
    //         DFGNode *ioNode = ioNodes[i++];
    //         ADGNode* newTargetAdgNode = mappedNode(ioNode);
    //         // route the src edges whose src nodes connected to DFG input node
    //         for(auto& edgePair : ioNode->inputEdges()){
    //             DFGEdge* edge = _dfg->edge(edgePair.second);
    //             int inNodeId = edge->srcId();
    //             DFGNode* inNode = _dfg->node(inNodeId);
    //             ADGNode* adgNode = mappedNode(inNode);
    //             bool routed = false;
    //             if(_dfg->isIONode(inNodeId) && !isRouted(edgePair.second)){ // connected to DFG input node
    //                 if(adgNode){ // already mapped
    //                     succeed = routeDfgEdge(edge, adgNode, newTargetAdgNode); // route edge between newTargetAdgNode and adgNode 
    //                 }else{ // unmapped
    //                     succeed = routeDfgEdge(edge, newTargetAdgNode, true); // route edge between newTargetAdgNode and ADG IOB
    //                     ioNodes.push_back(inNode); // cache newly mapped IO nodes
    //                 }                    
    //                 routed = true; 
    //             }
    //             if(!succeed){
    //                 break;
    //             }else if(routed){
    //                 routedEdges.push_back(edge); // cache the routed edge
    //             }
    //         }
    //         if(!succeed) break; 
    //         // route the dst edge whose dst nodes connected to unmapped DFG output port
    //         for(auto& elem : ioNode->outputEdges()){
    //             for(int outEdgeId : elem.second){
    //                 DFGEdge* edge = _dfg->edge(outEdgeId);
    //                 int outNodeId = edge->dstId();
    //                 DFGNode* outNode = _dfg->node(outNodeId);
    //                 ADGNode* adgNode = mappedNode(outNode);
    //                 bool routed = false;
    //                 if(adgNode == nullptr && _dfg->isIONode(outNodeId) && !isRouted(outEdgeId)){ // connected to DFG output node
    //                     if(adgNode){ // already mapped
    //                         succeed = routeDfgEdge(edge, newTargetAdgNode, adgNode); // route edge between newTargetAdgNode and adgNode 
    //                     }else{
    //                         succeed = routeDfgEdge(edge, newTargetAdgNode, false); // route edge between targetAdgNode and ADG IOB
    //                         ioNodes.push_back(outNode); // cache newly mapped IO nodes
    //                     }                        
    //                     routed = true; 
    //                 }
    //                 if(!succeed){
    //                     break;
    //                 }else if(routed){
    //                     routedEdges.push_back(edge); // cache the routed edge
    //                 }
    //             } 
    //             if(!succeed) break;  
    //         } 
    //         if(!succeed) break;  
    //     }

    // }
    if(!succeed){
        for(auto re : routedEdges){ // unroute all the routed edges
            unrouteDfgEdge(re);
        }
        return false;
    }
    // if(succeed && !_dfg->isIONode(dfgNode->id())){
    // map the DFG node to this targetAdgNode
    bool res = mapDfgNodeNoRoute(dfgNode, targetAdgNode);
    assert(res);
    // }
    
    return succeed;
}


// unmap DFG node without unrouting the input/output edges
void Mapping::unmapDfgNodeNoUnroute(DFGNode* dfgNode){
    ADGNode* adgNode = mappedNode(dfgNode);    
    if(adgNode == nullptr) return;
    _adgNodeAttr.erase(adgNode->id());
    _dfgNodeAttr.erase(dfgNode->id());
    _numNodeMapped--;
    spdlog::debug("Unmap DFG node {0}, id: {1}", dfgNode->name(), dfgNode->id());   
    
}


// unmap DFG node, unroute its input/output edges and unmap the to-be-free input/output nodes if any
void Mapping::unmapDfgNode(DFGNode* dfgNode){
    // std::vector<DFGNode*> unmapInNodes;
    // unroute the input edges of this node
    for(auto& elem : dfgNode->inputEdges()){ // input-idx, edge-id
        DFGEdge* edge = _dfg->edge(elem.second);
        unrouteDfgEdge(edge);  
        // int inNodeId = edge->srcId();      
        // DFGNode* inNode = _dfg->node(inNodeId);    
        // if(_dfg->isIONode(inNodeId) && isMapped(inNode)){ // connected to mapped DFG input node
        //     unmapInNodes.push_back(inNode);
        // }
    }
    // unroute the output edges of this node
    for(auto& elem : dfgNode->outputEdges()){ // input-idx, edge-id
        for(auto& outEdgeId : elem.second){   // edge-id
            DFGEdge* edge = _dfg->edge(outEdgeId);
            unrouteDfgEdge(edge);  
            // int outNodeId = edge->dstId();          
            // if(_dfg->isIONode(outNodeId)){ // connected to DFG output node
            //     unmapDfgNodeNoUnroute(_dfg->node(outNodeId));
            // }
        }
    }
    // unmap this DFG node
    unmapDfgNodeNoUnroute(dfgNode);
    // // unmap the connected input nodes
    // for(auto& inNode : unmapInNodes){
    //     bool mapped = false;
    //     for(auto& elem : inNode->outputs()){
    //         for(auto& out : elem.second){
    //             if(isMapped(_dfg->node(out.first))){ // there are still routed edges connected to the dfg input node
    //                 mapped = true;
    //                 break;
    //             }
    //         }
    //         if(mapped) break;
    //     }
    //     if(!mapped){
    //         unmapDfgNodeNoUnroute(inNode);
    //     }
    // }
}


// // if the ADG link is already routed
// bool Mapping::isRouted(ADGLink* link){
//     return !_adgLinkAttr[link->id()].dfgEdges.empty();
// }
// if the DFG Edge is already routed
bool Mapping::isRouted(int dfgEdgeId){
    if(_dfgEdgeAttr.count(dfgEdgeId)){
        if(!_dfgEdgeAttr[dfgEdgeId].edgeLinks.empty()){
            return true;
        }
    }
    return false;
}

// routed DFG edge links, including inter-node and intra-node links
const std::vector<EdgeLinkAttr>& Mapping::routedEdgeLinks(DFGEdge* edge){
    assert(_dfgEdgeAttr.count(edge->id()));
    return _dfgEdgeAttr[edge->id()].edgeLinks;
}

// /* route DFG edge to ADG link
// *  one link can route multiple edges, but they should have the same srcId and srcPortIdx 
// */
// bool Mapping::routeDfgEdge(DFGEdge* edge, ADGLink* link){
//     if(_adgLinkAttr.count(link->id())){
//         for(auto& linkEdge : _adgLinkAttr[link->id()].dfgEdges){
//             if(linkEdge->srcId() != edge->srcId() || linkEdge->srcPortIdx() != edge->srcPortIdx()){
//                 return false; 
//             }
//         }
//     }
//     _adgLinkAttr[link->id()].dfgEdges.push_back(edge);
//     EdgeLinkAttr attr;
//     attr.isLink = true;
//     attr.edgeLink.adgLink = link;
//     _dfgEdgeAttr[edge->id()].edgeLinks.push_back(attr);
//     return true;
// }


// route DFG edge to passthrough GIB node
// one internal link can passthrough multiple edges, but they should have the same srcId and srcPortIdx
// isTry: just try to route, not change the routing status
bool Mapping::routeDfgEdgePass(DFGEdge* edge, ADGNode* passNode, int srcPort, int dstPort, bool isTry){
    assert(passNode->type() == "GIB");
    int bitWidth = edge->bitWidth();
    if(!passNode->bitWidths().count(bitWidth)) return false;
    GIBNode* gibNode = dynamic_cast<GIBNode*>(passNode);
    int passNodeId = passNode->id();
    int routeSrcPort = srcPort;
    int routeDstPort = dstPort;
    bool hasSameSrcEdge = false;
    if(_adgNodeAttr.count(passNodeId)){
        auto& passNodeAttr = _adgNodeAttr[passNodeId];
        // check if conflict with current routed edges
        for(auto& edgeLink : passNodeAttr.dfgEdgePass){
            auto passEdge = edgeLink.edge;
            if(passEdge->bitWidth() == bitWidth &&
               (edgeLink.srcPort == srcPort || edgeLink.dstPort == dstPort) &&  // occupy the same port
               (passEdge->srcId() != edge->srcId() || passEdge->srcPortIdx() != edge->srcPortIdx())){
                return false; 
            } else if(passEdge->bitWidth() == bitWidth &&
                      (edgeLink.srcPort == srcPort || edgeLink.dstPort == dstPort)){ // try to route same-source edge to same internal link
                routeSrcPort = edgeLink.srcPort; // default route link
                routeDstPort = edgeLink.dstPort;
                hasSameSrcEdge = true;
            } 
        }
        if(srcPort >= 0 && dstPort >= 0){ // manually assign srcPort and dstPort
            if(!gibNode->isInOutConnected(srcPort, dstPort)){
                return false;
            } else if(isAdgNodeOutPortUsed(passNodeId, bitWidth, dstPort) && (!hasSameSrcEdge || (routeSrcPort != srcPort) || (routeDstPort != dstPort))){
                return false; // if dstPort used, must have same-source edge with same srcPort and dstPort
            } else if(isAdgNodeInPortUsed(passNodeId, bitWidth, srcPort) && (!hasSameSrcEdge || (routeSrcPort != srcPort))){
                return false; // if srcPort used, must have same-source edge with same srcPort
            }   
            routeSrcPort = srcPort;
            routeDstPort = dstPort;         
        } else if(srcPort >= 0){ // auto-assign dstPort           
            if(isAdgNodeInPortUsed(passNodeId, bitWidth, srcPort) && (!hasSameSrcEdge || (routeSrcPort != srcPort))){
                return false; // no same-source edge or have different srcPort
            }
            if(!isAdgNodeInPortUsed(passNodeId, bitWidth, srcPort)){
                bool flag = false;
                for(auto port : gibNode->in2outs(srcPort)){
                    if(isAdgNodeOutPortUsed(passNodeId, bitWidth, port)){ // already used
                        continue;
                    }
                    // find one available port
                    routeSrcPort = srcPort;
                    routeDstPort = port;
                    flag = true;
                    break;
                }
                if(!flag){ // cannot find one available port
                    return false;
                }
            }           
            // if have same-source edge and same srcPort, select the same dstPort            
        } else if(dstPort >= 0){ // auto-assign srcPort            
            if(isAdgNodeOutPortUsed(passNodeId, bitWidth, dstPort) && (!hasSameSrcEdge || (routeDstPort != dstPort))){
                return false;
            }
            if(!isAdgNodeOutPortUsed(passNodeId, bitWidth, dstPort)){ // no same-source edge or have different dstPort
                bool flag = false;
                for(auto port : gibNode->out2ins(dstPort)){
                    if(isAdgNodeInPortUsed(passNodeId, bitWidth, port)){ // already used
                        continue;
                    }
                    // find one available port
                    routeSrcPort = port;
                    routeDstPort = dstPort;
                    flag = true;
                    break;                    
                }
                if(!flag){ // cannot find one available port
                    return false;
                }                
            }            
            // if have same-source edge with same dstPort, select the same srcPort and dstPort          
        } else { // auto-assign srcPort and dstPort
            if(!hasSameSrcEdge){
                bool outflag = false;
                for(auto& elem : passNode->outputs(bitWidth)){
                    int outPort = elem.first;
                    if(isAdgNodeOutPortUsed(passNodeId, bitWidth, outPort)){ // already used
                        continue;
                    }
                    bool inflag = false;
                    for(auto inPort : gibNode->out2ins(outPort)){
                        if(isAdgNodeInPortUsed(passNodeId, bitWidth, inPort)){ // already used
                            continue;
                        }
                        // find one available inport
                        routeSrcPort = inPort;
                        routeDstPort = outPort;
                        inflag = true;
                        break;                        
                    }
                    if(inflag){ // find one available port
                        outflag = true;
                        break;
                    }
                }
                if(!outflag){ // cannot find one available port
                    return false;
                }
            }
        }
    } else { // _adgNodeAttr.count(passNodeId) = 0; this passNode has not been used
        bool outflag = false;
        for(auto& elem : passNode->outputs(bitWidth)){
            int outPort = elem.first;
            auto inPorts = gibNode->out2ins(outPort);
            if(!inPorts.empty()){ // find one available inport
                routeSrcPort = *(inPorts.begin());
                routeDstPort = outPort;
                outflag = true;
                break;
            }
        }
        if(!outflag){ // cannot find one available port
            return false;
        }
    }

    if(!isTry){
        EdgeLinkAttr edgeLinkAttr;    
        edgeLinkAttr.srcPort = routeSrcPort;
        edgeLinkAttr.dstPort = routeDstPort;
        edgeLinkAttr.adgNode = passNode;
        _dfgEdgeAttr[edge->id()].edgeLinks.push_back(edgeLinkAttr);
        DfgEdgePassAttr edgePassAttr;
        edgePassAttr.edge = edge;
        edgePassAttr.srcPort = routeSrcPort;
        edgePassAttr.dstPort = routeDstPort;
        _adgNodeAttr[passNodeId].inPortUsed[routeSrcPort] = true;
        _adgNodeAttr[passNodeId].outPortUsed[routeDstPort] = true;
        _adgNodeAttr[passNodeId].inPortUsedByWidth[bitWidth][routeSrcPort] = true;
        _adgNodeAttr[passNodeId].outPortUsedByWidth[bitWidth][routeDstPort] = true;
        _adgNodeAttr[passNodeId].dfgEdgePass.push_back(edgePassAttr);        
    }
    return true;
}


// route DFG edge between srcNode and dstNode/OB
// find a routable path from srcNode to dstNode/OB by BFS
// dstNode = nullptr: OB, map output node
// dstPortRange: the input port index range of the dstNode
bool Mapping::routeDfgEdgeFromSrc(DFGEdge* edge, ADGNode* srcNode, ADGNode* dstNode, const std::set<int>& dstPortRange){
    struct VisitNodeInfo{
        // int inPortIdx;  // input port index of this node 
        int srcNodeId;  // src node ID
        int srcInPortIdx; // input port index of src node
        int srcOutPortIdx; // output port index of src node
    };
    // cache visited node information, <<node-id, inport-index>, VisitNodeInfo>
    std::map<std::pair<int, int>, VisitNodeInfo> visitNodes; 
    // cache visited nodes, <node, inport-index>
    std::queue<std::pair<ADGNode*, int>> nodeQue; 
    // Breadth first search for possible routing path
    // assign the index of the output port of the srcNode
    int dfgSrcPortIdx = edge->srcPortIdx();
    int srcNodePortIdx = _dfg->node(edge->srcId())->operation() == "LUT"
        ? 1 : dfgSrcPortIdx;
    int srcNodeId = edge->srcId();
    int bitWidth = edge->bitWidth();
    std::string dstNodeOp = _dfg->node(edge->dstId())->operation();
    // int dstNodePortIdx;
    // int mappedAdgOutPort;
    std::pair<int, int> finalDstNode; // <node-id, inport-index>
    bool success = false;
    nodeQue.push(std::make_pair(srcNode, -1));
    while(!nodeQue.empty()){
        auto queHead = nodeQue.front();
        ADGNode* adgNode = queHead.first;
        int inPortIdx = queHead.second;
        nodeQue.pop();
        VisitNodeInfo info;
        std::vector<int> outPortIdxs;
        if(inPortIdx == -1){ // srcNode
            outPortIdxs.push_back(srcNodePortIdx);
        }else{ // intermediate node, can only be GIB
            for(int outPortIdx : dynamic_cast<GIBNode*>(adgNode)->in2outs(inPortIdx)){
                if(!routeDfgEdgePass(edge, adgNode, inPortIdx, outPortIdx, true)){ // try to find an internal link to route the edge
                    continue;
                }
                outPortIdxs.push_back(outPortIdx); // collect all available outPort 
            }   
        } 
        // search this layer of nodes
        for(int outPortIdx : outPortIdxs){
            for(auto& elem : adgNode->output(bitWidth, outPortIdx)){
                int nextNodeId = elem.first;
                int nextSrcPort = elem.second;
                auto nextId = std::make_pair(nextNodeId, nextSrcPort);
                ADGNode* nextNode = _adg->node(nextNodeId);
                auto nextNodeType = nextNode->type();
                if((dstNode != nullptr) && (dstNode->id() == nextNodeId) && dstPortRange.count(nextSrcPort)){ // get to the dstNode
                    success = true;
                    finalDstNode = nextId;
                } else if((dstNode != nullptr) && // route to dstNode
                         ((dstNode->id() == nextNodeId) ||
                          (nextNodeType == "GPE") || // not use GPE node to route
                          (nextNodeType == "IOB") || // cannot route to the dstNode through IOB
                          visitNodes.count(nextId))){ // the <node-id, inport-index> already visited
                    continue;
                }else if((dstNode == nullptr) && nextNodeType == "IOB" && isIOBFree(nextNode) && 
                         dynamic_cast<IOBNode*>(nextNode)->opCapable(dstNodeOp)){ // route to OB
                    success = true;
                    finalDstNode = nextId;
                } else if((dstNode == nullptr) && // route to OB
                         ((nextNodeType == "IOB") || // already used
                          (nextNodeType == "GPE") || // not use GPE node to route
                          visitNodes.count(nextId))){ // the <node-id, inport-index> already visited
                    continue;
                } else if(isAdgNodeInPortUsed(nextNodeId, bitWidth, nextSrcPort)){ // the input port is already used
                    // if has the same srcId and srcPortIdx, nextId can still be used
                    auto& nextNodeAttr = _adgNodeAttr[nextNodeId];
                    bool conflict = false;
                    for(auto& edgeLink : nextNodeAttr.dfgEdgePass){
                        auto passEdge = edgeLink.edge;
                        if(passEdge->bitWidth() == bitWidth &&
                           edgeLink.srcPort == nextSrcPort &&
                           (passEdge->srcId() != srcNodeId || passEdge->srcPortIdx() != dfgSrcPortIdx)){
                            conflict = true; // the edge with different srcId or srcPortIdx occupied nextSrcPort
                            break;
                        }
                    }
                    if(conflict) continue;
                }
                nodeQue.push(std::make_pair(nextNode, nextSrcPort)); // cache in queue
                info.srcNodeId = adgNode->id(); //  current node ID
                info.srcInPortIdx = inPortIdx;  // input port index of src node
                info.srcOutPortIdx = outPortIdx;   //  output port index of current node 
                visitNodes[nextId] = info;
                if(success){
                    break;
                }  
            }
            if(success){
                break;
            }
        }
        if(success){
            break;
        }
    }
    if(!success){
        return false;
    }
    // route the found path
    auto routeNode = finalDstNode;
    auto& edgeLinks = _dfgEdgeAttr[edge->id()].edgeLinks;
    int dstPort = -1;
    while (true){
        int nodeId = routeNode.first;
        int srcPort = routeNode.second;
        // keep the DFG edge routing status
        EdgeLinkAttr edgeAttr;
        edgeAttr.srcPort = srcPort;
        edgeAttr.dstPort = dstPort;
        edgeAttr.adgNode = _adg->node(nodeId);        
        edgeLinks.push_back(edgeAttr);
        // keep the ADG Node routing status
        ADGNodeAttr& nodeAttr = _adgNodeAttr[nodeId];
        if(routeNode == finalDstNode){ // dstNode
         // std::cout << "Set ADG node " << routeNode.first << " inport " << srcPort << " used\n";
            nodeAttr.inPortUsed[srcPort] = true;  // only change the input port status
            nodeAttr.inPortUsedByWidth[bitWidth][srcPort] = true;
        } else if(nodeId == srcNode->id()){ // srcNode
            nodeAttr.outPortUsed[dstPort] = true; // only change the output port status
            nodeAttr.outPortUsedByWidth[bitWidth][dstPort] = true;
            break; // get to the srcNode
        } else{ // intermediate routing nodes
            nodeAttr.outPortUsed[dstPort] = true; 
            nodeAttr.inPortUsed[srcPort] = true;
            nodeAttr.outPortUsedByWidth[bitWidth][dstPort] = true;
            nodeAttr.inPortUsedByWidth[bitWidth][srcPort] = true;
            DfgEdgePassAttr passAttr;
            passAttr.edge = edge;
            passAttr.srcPort = srcPort;
            passAttr.dstPort = dstPort;
            nodeAttr.dfgEdgePass.push_back(passAttr);  
        }  
        dstPort = visitNodes[routeNode].srcOutPortIdx;
        routeNode = std::make_pair(visitNodes[routeNode].srcNodeId, visitNodes[routeNode].srcInPortIdx);
    }
    // reverse the passthrough nodes from srcNode to dstNode
    std::reverse(edgeLinks.begin(), edgeLinks.end());
    // map DFG output node
    if(dstNode == nullptr){
        bool res = mapDfgNodeNoRoute(_dfg->node(edge->dstId()), _adg->node(finalDstNode.first));
        assert(res);
    }
    return true;
}


// find a routable path from dstNode to srcNode/IB by BFS
// srcNode = nullptr: IB
bool Mapping::routeDfgEdgeFromDst(DFGEdge* edge, ADGNode* srcNode, ADGNode* dstNode, const std::set<int>& dstPortRange){
    struct VisitNodeInfo{
        int dstNodeId;     // dst node ID
        int dstInPortIdx;  // input port index of dst node
        int dstOutPortIdx; // output port index of dst node
    };
    // cache visited node information, <<node-id, outport-index>, VisitNodeInfo>
    std::map<std::pair<int, int>, VisitNodeInfo> visitNodes; 
    // cache visited nodes, <node, outport-index>
    std::queue<std::pair<ADGNode*, int>> nodeQue; 
    // Breadth first search for possible routing path
    // assign the index of the output port of the srcNode
    int dfgSrcPortIdx = edge->srcPortIdx();
    int srcNodeOutPortIdx = _dfg->node(edge->srcId())->operation() == "LUT"
        ? 1 : dfgSrcPortIdx;
    int srcNodeId = edge->srcId();
    int bitWidth = edge->bitWidth();
    std::string srcNodeOp = _dfg->node(srcNodeId)->operation();
    // int srcNodeInPortIdx = -1;
    // int mappedAdgInPort;
    std::pair<int, int> finalSrcNode; // <node-id, outport-index>
    bool success = false;
    DFGEdge* sameSrcEdge = nullptr; // the already-routed edge with the same srcId and srcPortIdx
    nodeQue.push(std::make_pair(dstNode, -1));
    while(!nodeQue.empty()){
        auto queHead = nodeQue.front();
        ADGNode* adgNode = queHead.first;
        int outPortIdx = queHead.second;
        nodeQue.pop();        
        std::vector<int> inPortIdxs;
        if(outPortIdx == -1){ // dstNode
            inPortIdxs.assign(dstPortRange.begin(), dstPortRange.end());
        }else{ // intermediate node, can only be GIB
            for(int inPortIdx : dynamic_cast<GIBNode*>(adgNode)->out2ins(outPortIdx)){
                if(!routeDfgEdgePass(edge, adgNode, inPortIdx, outPortIdx, true)){ // try to find an internal link to route the edge
                    continue;
                }
                inPortIdxs.push_back(inPortIdx); // collect all available inPort 
            }   
        } 
        // search this layer of nodes
        for(int inPortIdx : inPortIdxs){
            auto elem = adgNode->input(bitWidth, inPortIdx);
            int nextNodeId = elem.first;
            int nextDstPort = elem.second;
            auto nextId = std::make_pair(nextNodeId, nextDstPort);
            ADGNode* nextNode = _adg->node(nextNodeId);
            // if(nextNode == nullptr) { //  not in this tile
            //     continue;
            // }
            std::string nextNodeType = nextNode->type();
            if((srcNode != nullptr) && (srcNode->id() == nextNodeId) && srcNodeOutPortIdx == nextDstPort){ // get to the srcNode
                success = true;
                finalSrcNode = nextId;
            } else if((srcNode != nullptr) && // route to srcNode
                     ((srcNode->id() == nextNodeId) ||
                      (nextNodeType == "GPE") || // not use GPE node to route
                      (nextNodeType == "IOB") || // cannot route to the srcNode through IOB
                      visitNodes.count(nextId))){ // the <node-id, outport-index> already visited
                continue;
            }else if((srcNode == nullptr) && nextNodeType == "IOB" && isIOBFree(nextNode) && 
                     dynamic_cast<IOBNode*>(nextNode)->opCapable(srcNodeOp)){ // route to free IOB
                success = true;
                finalSrcNode = nextId;
            } else if((srcNode == nullptr) && // route to IB
                     ((nextNodeType == "IOB") || // already used as OB
                      (nextNodeType == "GPE") || // not use GPE node to route
                      visitNodes.count(nextId))){ // the <node-id, outport-index> already visited
                continue;
            } else if(isAdgNodeOutPortUsed(nextNodeId, bitWidth, nextDstPort)){ // the output port is already used
                // if has the same srcId and srcPortIdx, nextId can still be used
                auto& nextNodeAttr = _adgNodeAttr[nextNodeId];
                bool conflict = false;
                for(auto& edgeLink : nextNodeAttr.dfgEdgePass){
                    auto passEdge = edgeLink.edge;
                    if(passEdge->bitWidth() == bitWidth &&
                       edgeLink.dstPort == nextDstPort &&
                       (passEdge->srcId() != srcNodeId || passEdge->srcPortIdx() != dfgSrcPortIdx)){
                        conflict = true; // the edge with different srcId or srcPortIdx occupied nextDstPort
                        break;
                    }else if(passEdge->bitWidth() == bitWidth && edgeLink.dstPort == nextDstPort){
                        sameSrcEdge = passEdge; // the edge with the same srcId and srcPortIdx
                        finalSrcNode = nextId;
                        success = true; // REUSE part of the routing path of the edge
                        break;
                    }
                }
                if(conflict) continue;
            }
            nodeQue.push(std::make_pair(nextNode, nextDstPort)); // cache in queue
            VisitNodeInfo info;
            info.dstNodeId = adgNode->id(); //  current node ID
            info.dstInPortIdx = inPortIdx;  // input port index of src node
            info.dstOutPortIdx = outPortIdx;   //  output port index of current node 
            visitNodes[nextId] = info;
            if(success){
                break;
            }  
        }
        if(success){
            break;
        } 
    }
    if(!success){
        return false;
    }
    // route the found path
    auto routeNode = finalSrcNode;
    auto& edgeLinks = _dfgEdgeAttr[edge->id()].edgeLinks;
    int srcPort = -1; // (srcNode != nullptr)? -1 : srcNodeInPortIdx;
    // reuse part of the routing path of the edge
    if(sameSrcEdge != nullptr){
        auto& sameSrcEdgeLinks = _dfgEdgeAttr[sameSrcEdge->id()].edgeLinks;
        for(auto& link : sameSrcEdgeLinks){ // search from the source node
            // keep the DFG edge routing status
            edgeLinks.push_back(link);
            // keep the ADG Node routing status
            ADGNodeAttr& nodeAttr = _adgNodeAttr[link.adgNode->id()];
            DfgEdgePassAttr passAttr;
            passAttr.edge = edge;
            passAttr.srcPort = link.srcPort;
            passAttr.dstPort = link.dstPort;
            nodeAttr.dfgEdgePass.push_back(passAttr);
            if(link.adgNode->id() == finalSrcNode.first && link.dstPort == finalSrcNode.second){
                break;
            }
        }
        srcPort = visitNodes[finalSrcNode].dstInPortIdx;
        routeNode = std::make_pair(visitNodes[finalSrcNode].dstNodeId, visitNodes[finalSrcNode].dstOutPortIdx);
    }
    while (true){
        int nodeId = routeNode.first;
        int dstPort = routeNode.second;
        // keep the DFG edge routing status
        EdgeLinkAttr edgeAttr;
        edgeAttr.srcPort = srcPort;
        edgeAttr.dstPort = dstPort;
        edgeAttr.adgNode = _adg->node(nodeId);        
        edgeLinks.push_back(edgeAttr);
        // keep the ADG Node routing status
        ADGNodeAttr& nodeAttr = _adgNodeAttr[nodeId];
        if(routeNode == finalSrcNode){ // srcNode
            nodeAttr.outPortUsed[dstPort] = true; // only change the output port status 
            nodeAttr.outPortUsedByWidth[bitWidth][dstPort] = true;
        } else if(nodeId == dstNode->id()){ // dstNode
        // std::cout << "Set ADG node " << nodeId << " inport " << srcPort << " used\n";
            nodeAttr.inPortUsed[srcPort] = true;  // only change the input port status 
            nodeAttr.inPortUsedByWidth[bitWidth][srcPort] = true;
            break; // get to the dstNode
        } else{
            nodeAttr.inPortUsed[srcPort] = true;
            nodeAttr.outPortUsed[dstPort] = true;
            nodeAttr.inPortUsedByWidth[bitWidth][srcPort] = true;
            nodeAttr.outPortUsedByWidth[bitWidth][dstPort] = true;
            DfgEdgePassAttr passAttr;
            passAttr.edge = edge;
            passAttr.srcPort = srcPort;
            passAttr.dstPort = dstPort;
            nodeAttr.dfgEdgePass.push_back(passAttr);
        }        
        srcPort = visitNodes[routeNode].dstInPortIdx;
        routeNode = std::make_pair(visitNodes[routeNode].dstNodeId, visitNodes[routeNode].dstOutPortIdx);
    }
    // map DFG input node
    if(srcNode == nullptr){
        bool res = mapDfgNodeNoRoute(_dfg->node(srcNodeId), _adg->node(finalSrcNode.first));
        assert(res);
    }
    
    return true;
}


// find the available input ports in the dstNode to route edge
std::set<int> Mapping::availDstPorts(DFGEdge* edge, ADGNode* dstNode){
    DFGNode* dstDfgNode = _dfg->node(edge->dstId());
    int bitWidth = edge->bitWidth();
    int edgeDstPort = edge->dstPortIdx();
    int opereandNum = dstDfgNode->numInputs(bitWidth);
    FUNode* dstFuNode = dynamic_cast<FUNode*>(dstNode);
    std::set<int> dstPortRange; // the input port index range of the dstNode
    std::vector<int> opIdxs; // operand indexes
    if(dstDfgNode->commutative()){ // operands are commutative, use all the operand indexes
        for(int opIdx = 0; opIdx < opereandNum; opIdx++){
            opIdxs.push_back(opIdx);
        }
    } else{ // operands are not commutative, use the edgeDstPort as the operand index
        int opIdx = edgeDstPort;
        // Mixed-width I/O operations use global DFG operand indexes (for
        // example, COUTPUT's one-bit predicate is operand B/index 1), while
        // the ADG stores operands independently for each bit width.  An IOB
        // with one fine-grained operand therefore exposes it as local index 0.
        if(bitWidth == 1 &&
           dynamic_cast<IOBNode*>(dstFuNode) &&
           dstFuNode->numOperands(1) == 1){
            opIdx = 0;
        }
        opIdxs.push_back(opIdx);
    }
    // select all the ports connected to available operand
    for(int opIdx : opIdxs){
        if(dstDfgNode->operation() == "LUT"){
            auto* gpeNode = dynamic_cast<GPENode*>(dstFuNode);
            if(!gpeNode || !gpeNode->hasLUT()) continue;
            opIdx = gpeNode->getOperandIdxLUT(opIdx);
        }
        bool operandUsed = false; // if this operand is used
        auto& inPorts = dstFuNode->operandInputs(bitWidth, opIdx);
        for(int inPort : inPorts){ // input ports connected to the operand with index of opIdx
            if(isAdgNodeInPortUsed(dstFuNode->id(), bitWidth, inPort)){
                operandUsed = true;
                break;
            }
        }
        if(!operandUsed){ 
            for(int inPort : inPorts){ // all the ports are available
                dstPortRange.emplace(inPort);
            }                
        }
    }
    return dstPortRange;
}


// route DFG edge between srcNode and dstNode
// find a routable path from srcNode to dstNode by BFS
bool Mapping::routeDfgEdge(DFGEdge* edge, ADGNode* srcNode, ADGNode* dstNode){
    int bitWidth = edge->bitWidth();
    if((srcNode && !srcNode->bitWidths().count(bitWidth)) ||
       (dstNode && !dstNode->bitWidths().count(bitWidth))){
        return false;
    }
    std::set<int> dstPortRange = availDstPorts(edge, dstNode); // the input port index range of the dstNode
    if(dstPortRange.empty()){ // no available input port in the dstNode
        return false;
    }
    return routeDfgEdgeFromDst(edge, srcNode, dstNode, dstPortRange);
}


// route DFG edge between adgNode and IOB
// is2Input: whether connected to IB or OB
// bool Mapping::routeDfgEdge(DFGEdge* edge, ADGNode* adgNode, bool is2Input){
//     std::set<int> dstPortRange; // the input port index range of the dstNode
//     if(!is2Input){ // route to OB
//         return routeDfgEdgeFromSrc(edge, adgNode, nullptr, dstPortRange);       
//     } 
//     dstPortRange = availDstPorts(edge, adgNode); // the input port index range of the dstNode
//     if(dstPortRange.empty()){ // no available input port in the dstNode
//         return false;
//     }
//     return routeDfgEdgeFromDst(edge, nullptr, adgNode, dstPortRange);
// }


// // unroute DFG edge and unmap the to-be-free input/output nodes if any
// unroute DFG edge without unmapping the to-be-free input/output nodes if any
void Mapping::unrouteDfgEdge(DFGEdge* edge){
    int eid = edge->id();
    int bitWidth = edge->bitWidth();
    if(!_dfgEdgeAttr.count(eid)) return;
    auto& edgeAttr = _dfgEdgeAttr[eid]; 
    spdlog::debug("Unroute DFG edge {0} ({1} -> {2}) ", eid, _dfg->node(edge->srcId())->name(), _dfg->node(edge->dstId())->name());   
    // std::cout << "unroute DFG edge " << eid << " (" << _dfg->edge(eid)->srcId() << ", " << _dfg->edge(eid)->dstId() << ")\n";
    // remove this edge from all the routed ADG links
    for(auto& edgeLink : edgeAttr.edgeLinks){
        auto node = edgeLink.adgNode;
        if(!_adgNodeAttr.count(node->id())) continue; 
        auto& nodeAttr = _adgNodeAttr[node->id()];                   
        auto& nodeEdges = nodeAttr.dfgEdgePass;
        auto iter = std::remove_if(nodeEdges.begin(), nodeEdges.end(), [&](DfgEdgePassAttr& x){ return (x.edge == edge); });
        nodeEdges.erase(iter, nodeEdges.end());
        bool setInPortUnused = (edgeLink.srcPort != -1);
        bool setOutPortUnused = (edgeLink.dstPort != -1);
        for(auto& nodeEdge : nodeEdges){ // the srcPort/dstPort may be used by other edges
            if(setInPortUnused && nodeEdge.edge->bitWidth() == bitWidth &&
               (nodeEdge.srcPort == edgeLink.srcPort)){
                setInPortUnused = false; // do not set unused
            }
            if(setOutPortUnused && nodeEdge.edge->bitWidth() == bitWidth &&
               (nodeEdge.dstPort == edgeLink.dstPort)){
                setOutPortUnused = false;
            }
            if(!setInPortUnused && !setOutPortUnused){
                break;
            }
        }
        if(setInPortUnused){
            nodeAttr.inPortUsedByWidth[bitWidth][edgeLink.srcPort] = false;
            bool usedAtAnyWidth = false;
            for(auto& widthPorts : nodeAttr.inPortUsedByWidth){
                auto it = widthPorts.second.find(edgeLink.srcPort);
                usedAtAnyWidth |= it != widthPorts.second.end() && it->second;
            }
            nodeAttr.inPortUsed[edgeLink.srcPort] = usedAtAnyWidth;
            // if(node->type() == "GPE"){
            //     std::cout << "Set ADG node " << node->id() << " inport " << edgeLink.srcPort << " unused\n";
            // }
        }
        if(setOutPortUnused){
            nodeAttr.outPortUsedByWidth[bitWidth][edgeLink.dstPort] = false;
            bool usedAtAnyWidth = false;
            for(auto& widthPorts : nodeAttr.outPortUsedByWidth){
                auto it = widthPorts.second.find(edgeLink.dstPort);
                usedAtAnyWidth |= it != widthPorts.second.end() && it->second;
            }
            nodeAttr.outPortUsed[edgeLink.dstPort] = usedAtAnyWidth;
        }            
    }
    _dfgEdgeAttr.erase(eid);
    // // unmap DFG IO Node
    // std::vector<int> ioNodeIds;
    // ioNodeIds.push_back(edge->dstId());     
    // ioNodeIds.push_back(edge->srcId());       
    // for(int ioNodeId : ioNodeIds){       
    //     if(_dfg->isIONode(ioNodeId)){ // connected to DFG IO node
    //         DFGNode *ioNode = _dfg->node(ioNodeId);
    //         bool mapped = false;
    //         std::vector<int> eids;
    //         for(auto& elem : ioNode->inputEdges()){
    //             eids.push_back(elem.second);            
    //         }
    //         for(auto& elem : ioNode->outputEdges()){
    //             for(auto& out : elem.second){
    //                 eids.push_back(out);
    //             }
    //         }
    //         for(auto eid : eids){
    //             if(isRouted(eid)){ // there are still routed edges connected to the dfg io node
    //                 mapped = true;
    //                 break;
    //             }
    //         }
    //         if(!mapped){
    //             unmapDfgNodeNoUnroute(ioNode);
    //         }
    //     }
    // }
}


// if succeed to map all DFG nodes
bool Mapping::success(){
    return _dfg->nodes().size() == _numNodeMapped;
}


// total/max edge length (link number)
void Mapping::getEdgeLen(int& totalLen, int& maxLen){
    int total = 0;
    int maxVal = 0;
    for(auto& elem : _dfgEdgeAttr){
        int num = elem.second.edgeLinks.size();
        total += num;
        maxVal = std::max(maxVal, num);
    }
    totalLen = total;
    maxLen = maxVal;
}


// number of total mapped ADG nodes
int Mapping::getMappedAdgNodeNum(){
    return _adgNodeAttr.size();
}



// ==== tming schedule >>>>>>>>>>>>>
// // reset the timing bounds of each DFG node
// void Mapping::resetBound(){
//     // int INF = 0x3fffffff;
//     for(auto& elem : _dfg->nodes()){
//         auto& attr = _dfgNodeAttr[elem.second->id()];
//         attr.minLat = 0;
//         attr.maxLat = 0;
//     }
// }

// get currently available delay cycles in the FU node according to the mapped DFG node
int Mapping::getAvailDelay(FUNode* fuNode, DFGNode* dfgNode, int bitWidth){
    if(bitWidth == 0) bitWidth = dfgNode->operation() == "LUT" ? 1 : fuNode->bitWidth();
    if(fuNode->numOperands(bitWidth) < 2){
        return 0; // no delay unit
    }
    int maxDelay = fuNode->maxDelay(bitWidth);
    int numOp = dfgNode->inputs(bitWidth).size();

    // int numOp = 0;
    // for(auto elem : dfgNode->inputEdges()){
    //     if(!_dfg->edge(elem.second)->isMemBackEdge()){
    //         numOp++;
    //     }
    // }
    if(numOp == 0) return 0;
    int availDelay;

    /// already used this fu as delay unit
    auto delayKey = std::make_pair(fuNode->id(), bitWidth);
    if(_fuDelayAttr.count(delayKey)){
        auto& attr = _fuDelayAttr[delayKey];
        int leftOps = numOp - attr.delayUsed.size();
        availDelay = (maxDelay - attr.totalDelayUsed) / std::max(leftOps, 1); // left delay cycles divided by the left operands
    }else{
        availDelay = maxDelay / numOp;
    }
    return availDelay;
}


// @jhlou: pre assign delay pipe for MERGE node
void Mapping::preAssignRdu(){
    for(auto elem : _adgNodeAttr){
        int adgnodeId = elem.first;     

        if(isMapped(_adg->node(adgnodeId))){
            DFGNode* dfgnode = elem.second.dfgNode;
            if(dfgnode->operation() == "MERGE4" || dfgnode->operation() == "INTLV4"){
                // _fuDelayAttr[adgnodeId].delayUsed[1] = 1;
                // _fuDelayAttr[adgnodeId].delayUsed[2] = 2;
                // _fuDelayAttr[adgnodeId].delayUsed[2] = 3;
                _fuDelayAttr[std::make_pair(adgnodeId, _adg->node(adgnodeId)->bitWidth())].totalDelayUsed = 6;
            }
            else if(dfgnode->operation() == "MERGE3" || dfgnode->operation() == "INTLV3"){
                // _fuDelayAttr[adgnodeId].delayUsed[1] = 1;
                // _fuDelayAttr[adgnodeId].delayUsed[2] = 2;
                _fuDelayAttr[std::make_pair(adgnodeId, _adg->node(adgnodeId)->bitWidth())].totalDelayUsed = 3;
            }        
            else if(dfgnode->operation() == "MERGE2" || dfgnode->operation() == "INTLV2"){
                // _fuDelayAttr[adgnodeId].delayUsed[1] = 1;
                _fuDelayAttr[std::make_pair(adgnodeId, _adg->node(adgnodeId)->bitWidth())].totalDelayUsed = 1;
            }
        }    
    }
}

// @jhlou: pre assign additional start delay pipe for nodes following merge/intlv node
// NOTE(optional optimization):
// This function currently uses a fixpoint iteration (while !finished) to pre-assign
// additionalStartDelay for nodes after MERGE/INTLV and to align other input branches.
// The iteration is robust: it keeps propagating until convergence.
//
// A faster alternative is a *two-pass propagation*:
//   (1) Forward topo pass (inputs -> outputs): propagate delay to "following" nodes.
//   (2) Reverse topo pass (outputs -> inputs): back-propagate to fill/align other
//       non-merge input branches.
// This often works for simple/monotonic DFGs, but is NOT always equivalent.
//
// WARNING:
// For complex graphs with multiple reconvergences / multi-level merge chains / bypass
// paths that re-join later, the two-pass method may miss updates that require more
// than one forward-backward sweep. In such cases, the fixpoint iteration is required
// to guarantee convergence.
//
// If needed, implement it as an option, e.g. `--preassign-delay-two-pass`, and keep
// this fixpoint version as the default for correctness.
void Mapping::preAssignAdditionalStartDelay(){
    DFG* dfg = getDFG();
    bool finished = false;
    while(!finished){
        finished = true;
        for(auto id : dfg->topoNodes()){
            DFGNode* dfgNode = dfg->node(id);
            std::vector<DFGNode*> nodes_to_assign;
            int additional_delay = 0;
            for(auto elem : dfgNode->inputEdges()){
                int eid = elem.second;
                DFGEdge* edge = _dfg->edge(eid);
                if(edge->isBackEdge()){
                    continue;
                }  
                DFGNode* inNode = dfg->node(edge->srcId());
                if(inNode->operation() == "MERGE4" || inNode->operation() == "INTLV4"){
                    assert(additional_delay == 0 || additional_delay == 3);
                    additional_delay = 3;
                }
                else if(inNode->operation() == "MERGE3" || inNode->operation() == "INTLV3"){
                    assert(additional_delay == 0 || additional_delay == 2);
                    additional_delay = 2;
                }
                else if(inNode->operation() == "MERGE2" || inNode->operation() == "INTLV2"){
                    assert(additional_delay == 0 || additional_delay == 1);
                    additional_delay = 1;
                } 
                else if(inNode->additionalStartDelay() != 0){
                    assert(additional_delay == 0 || additional_delay == inNode->additionalStartDelay());
                    additional_delay = inNode->additionalStartDelay();
                }   
                else{
                    nodes_to_assign.push_back(inNode);
                }
            }
            if(additional_delay != 0){
                nodes_to_assign.push_back(dfgNode);
                for(auto node : nodes_to_assign){
                    if(node->additionalStartDelay() == 0){
                        node->setAdditionalStartDelay(additional_delay);
                        finished = false;
                    }
                    else{
                        assert(node->additionalStartDelay() == additional_delay);
                    }
                }
            }
        }
    }
}

// calculate the routing latency of each edge, not inlcuding the RDU
void Mapping::calEdgeRouteLat(){
    for(auto& elem : _dfgEdgeAttr){
        auto& attr = elem.second;
        int lat = 0;
        for(auto& linkAttr : attr.edgeLinks){
            if(linkAttr.adgNode->type() == "GIB"){ // edge only pass-through GIB nodes
                bool reged = dynamic_cast<GIBNode*>(linkAttr.adgNode)->outReged(linkAttr.dstPort); // output port reged
                lat += reged;
            }
        }
        attr.lat = lat;
        // attr.latNoDelay = lat;
    }
}


// calculate the DFG node latency bounds not considering the Delay components, including 
// min latency of the output ports
// ID of the DFG node with the max latency
void Mapping::latencyBound(){
    int maxLatDfg = 0; // max latency of DFG
    int maxLatNodeId;  // ID of the DFG node with the max latency
    // calculate the LOWER bounds in topological order
    // int inputLat = _adg->loadLatency(); // IOB input latency
    // auto dfgInNodes = _dfg->getInNodes(); // In nodes: INPUT node, LOAD node without input
    // available latency%II for multiport IO nodes, 0~(II-1)
    std::map<std::string, std::set<int>> multiportAvailLatModII;
    std::set<int> nums;
    for(int i = 0; i < _II; i++){
        nums.insert(i);
    }
    // for(auto &elem : _dfg->multiportIOs()){
    //     multiportAvailLatModII[elem.first] = nums;
    // }
    for(auto nodeId : _dfg->topoNodes()){
        DFGNode *node = _dfg->node(nodeId);
        // node->printDfgNode();
        int maxLat = 0; // max latency of the input ports
        int nodeOpLat = _dfg->node(nodeId)->opLatency(); // operation latency
        for(auto& elem : node->inputEdges()){
            int eid = elem.second;
            DFGEdge *edge = _dfg->edge(eid);
            int routeLat = _dfgEdgeAttr[eid].lat; // latNoDelay;
            int srcNodeId = edge->srcId();    

            if(edge->isBackEdge()){ // skip back edge
                maxLat = std::max(maxLat, std::max(_dfgNodeAttr[srcNodeId].maxLat + routeLat - _II * edge->iterDist(), 0));
                continue;
            }

            int srcNodeLat = _dfgNodeAttr[srcNodeId].lat;
            int inPortLat = srcNodeLat + routeLat;
            maxLat = std::max(maxLat, inPortLat);
        }
        int lat = maxLat + nodeOpLat;
        int multi_lat = lat%_II;
        // adjust the latency of the multiport IO node to avoid spad access conflict
        // if(_dfg->isMultiportIoNode(nodeId)){
        //     if(_dfg->getInNodes().count(nodeId))
        //     {
        //         multi_lat = (maxLat+1)%_II;
        //     }
        //     std::string name = dynamic_cast<DFGIONode*>(node)->memRefName();
        //     while(!multiportAvailLatModII[name].count(multi_lat)){
        //         lat++;
        //         if(_dfg->getInNodes().count(nodeId))
        //         {
        //             multi_lat = (lat - nodeOpLat + 1)%_II;
        //         }else{
        //             multi_lat = lat %_II;
        //         }
        //     }
        //     multiportAvailLatModII[name].erase(multi_lat);
        // }
        _dfgNodeAttr[nodeId].lat = lat;
        if(lat >= maxLatDfg){
            maxLatDfg = lat;
            maxLatNodeId = nodeId;
        }       
    }
    _maxLat = maxLatDfg;
    _maxLatNodeId = maxLatNodeId;
}


// schedule the latency of each DFG node based on the mapping status
// DFG node latency: output port latency
// DFG edge latency: latency from the output port of src node to the ALU Input port of dst Node, including RDU 
void Mapping::latencySchedule(){
    _fuDelayAttr.clear();
    std::set<int> scheduledNodeIds;  

    // std::vector<int> unscheduledNodes = _dfg->topoNodes();
    // std::reverse(unscheduledNodes.begin(), unscheduledNodes.end()); // in reversed topological order
    auto& topoNodes = _dfg->topoNodes();
    std::list<int> unscheduledNodes(topoNodes.rbegin(), topoNodes.rend());        

    // @jhlou set pre-assigned RDU for some node: Interleave(INTLV/MERGE)
    preAssignRdu();
    preAssignAdditionalStartDelay();

    // calculate the routing latency of each edge, not inlcuding the RDU
    calEdgeRouteLat();
    // calculate the DFG node latency bounds, finding the max-latency path 
    latencyBound();
    // schedule the DFG nodes in the max-latency path
    DFGNode* dfgNode = _dfg->node(_maxLatNodeId);
    scheduledNodeIds.emplace(dfgNode->id());
    // auto iterEnd = std::remove(unscheduledNodes.begin(), unscheduledNodes.end(), dfgNode->id());
    // unscheduledNodes.erase(std::remove(unscheduledNodes.begin(), unscheduledNodes.end(), dfgNode), unscheduledNodes.end());
    // int inputLat = _adg->loadLatency(); // IOB input latency 
    while(dfgNode){ // until getting to the input port
        int nodeId = dfgNode->id();     
        // std::cout << "critical path node name: "<< dfgNode->name()<<" id: " << nodeId  << std::endl;
        // std::cout << "id: " << nodeId << std::endl;   
        // use the latency lower bound as the target latency
        int inPortLat = _dfgNodeAttr[nodeId].lat - _dfg->node(nodeId)->opLatency(); // input port latency
        FUNode* fuNode = dynamic_cast<FUNode*>(_dfgNodeAttr[nodeId].adgNode); // mapped FU node (GPE/IOB)
        DFGNode* srcNode = nullptr;
        // dfgNode->printDfgNode();
        int inPort;
        int inBitWidth = dfgNode->operation() == "LUT" ? 1 : fuNode->bitWidth();
        int delayRequired = 0;
        int maxArriveLat = -1;
        bool postrm = true;
        for(auto& elem : dfgNode->inputEdges()){
            int eid = elem.second;

            DFGEdge* edge = _dfg->edge(eid);// @yuan: don't touch the back-edge
            if(edge->isBackEdge()){            
                continue;
            }

            inPort = elem.first;
            int routeLat = _dfgEdgeAttr[eid].lat; // latNoDelay;
            int srcNodeId = _dfg->edge(eid)->srcId();
            int srcNodeLat = _dfgNodeAttr[srcNodeId].lat;    
            int arriveLat = srcNodeLat + routeLat;                   
            // if(inPortLat == srcNodeLat + routeLat){ 
            //     scheduledNodeIds.emplace(srcNodeId); // latency fixed
            //     srcNode = _dfg->node(srcNodeId);
            //     // iterEnd = std::remove(unscheduledNodes.begin(), iterEnd, srcNodeId);
            //     // unscheduledNodes.erase(std::remove(unscheduledNodes.begin(), unscheduledNodes.end(), srcNode), unscheduledNodes.end());
            //     break; // only find one path
            // } 
            if(inPortLat == arriveLat){ 
                delayRequired = 0;
                scheduledNodeIds.emplace(srcNodeId); // latency fixed
                srcNode = _dfg->node(srcNodeId); 
                unscheduledNodes.remove(srcNodeId);
                // iterEnd = std::remove(unscheduledNodes.begin(), iterEnd, srcNodeId);
                // unscheduledNodes.erase(std::remove(unscheduledNodes.begin(), unscheduledNodes.end(), srcNode), unscheduledNodes.end());
                inPort = elem.first;
                inBitWidth = edge->bitWidth();
                postrm = false;
                break; // only find one path
            }else if(arriveLat > maxArriveLat){
                maxArriveLat = arriveLat;
                delayRequired = inPortLat - arriveLat;
                srcNode = _dfg->node(srcNodeId);
                inPort = elem.first;
                inBitWidth = edge->bitWidth();
            }
        }
        if(maxArriveLat < 0 && srcNode == nullptr){
            _dfgNodeAttr[nodeId].maxLat = inPortLat;
            _dfgNodeAttr[nodeId].minLat = inPortLat;
            break;
        }
        auto delayKey = std::make_pair(fuNode->id(), inBitWidth);
        _fuDelayAttr[delayKey].delayUsed[inPort] =
            std::min(delayRequired, getAvailDelay(fuNode, dfgNode, inBitWidth));
        _dfgNodeAttr[nodeId].maxLat = inPortLat; // input port max latency 
        _dfgNodeAttr[nodeId].minLat =
            std::max(inPortLat - getAvailDelay(fuNode, dfgNode, inBitWidth), 0);
        dfgNode = srcNode;
        if(postrm && srcNode != nullptr){
            scheduledNodeIds.emplace(srcNode->id()); // latency fixed
            unscheduledNodes.remove(srcNode->id());
            // iterEnd = std::remove(unscheduledNodes.begin(), iterEnd, srcNode->id());
        }
    }
    // unscheduledNodes.erase(iterEnd, unscheduledNodes.end());
    
    // std::cout << " schedule critical path done!!  unscheduledNodes size: " << unscheduledNodes.size() << " scheduledNodeIds size: " <<scheduledNodeIds.size()<< std::endl;
    // schedule the DFG nodes not in the max-latency path
    while(!unscheduledNodes.empty()){
        for(auto iter = unscheduledNodes.begin(); iter != unscheduledNodes.end();){           
            int nodeId = *iter;
            dfgNode = _dfg->node(nodeId);
            DebugMode(dfgNode->printDfgNode();)
            DebugMode(std::cout << "unschedule node: " << dfgNode->name() << " id: " << nodeId << " unscheduledNodes size: " << unscheduledNodes.size() << std::endl;)
            
            DebugMode(std::cout << "id: " << nodeId << std::endl;)
            int maxLat = 0x3fffffff;
            int minLat = 0;
            int targetLat = _dfgNodeAttr[nodeId].lat; 
            bool fail = false; // fail to schedule this DFG node
            bool updated = false; // maxLat/minLat UPDATED
            for(auto& outsPerPort : dfgNode->outputEdges()){
                for(auto& eid : outsPerPort.second){
                    DFGEdge* edge = _dfg->edge(eid); 
                    int routeLat = _dfgEdgeAttr[eid].lat; // latNoDelay;
                    int dstNodeId = _dfg->edge(eid)->dstId();
                    // if(dstNodeId == _dfg->id()){ // connected to DFG output port
                    //     continue;
                    // }
                    DebugMode(std::cout << "output node: " << _dfg->node(dstNodeId)->name() << std::endl;)
                    if(scheduledNodeIds.count(dstNodeId)){ // already scheduled  
                        if(edge->isBackEdge()){     
                            routeLat -= _II * edge->iterDist(); // latency due to iteration distance
                            //std::cout<<"edge"<<edge->id()<<std::endl;
                            if(_dfgNodeAttr[dstNodeId].maxLat - routeLat < targetLat){ // need to increase II
                                continue;
                            }
                        }                           
                        maxLat = std::min(maxLat, _dfgNodeAttr[dstNodeId].maxLat - routeLat);
                        minLat = std::max(minLat, _dfgNodeAttr[dstNodeId].minLat - routeLat);
                        updated = true;
                    }else{
                        if(edge->isBackEdge()) {
                            updated = false;
                            continue;
                        }
                        fail = true;
                        break;
                    }
                }
                if(fail) break;
            }
            if(fail){
                iter++;
                continue;
            } 
            // if all its output ports are connected to DFG output ports, keep original latency
            // otherwise, update the latency
            // int targetLat = _dfgNodeAttr[nodeId].lat; 
            if(updated){
                // targetLat = std::max(targetLat, std::min(maxLat, minLat)); /// for backedge jhlou
                if(_dfg->backEdgesheadNodeId().count(nodeId)){
                    targetLat = std::max(targetLat, std::max(maxLat, minLat));  
                }else{
                    targetLat = std::max(targetLat, std::min(maxLat, minLat));
                }
            }           
            _dfgNodeAttr[nodeId].lat = targetLat; 
            // update             
            int maxInportLat = targetLat - _dfg->node(nodeId)->opLatency(); // input port max latency   
            _dfgNodeAttr[nodeId].maxLat = maxInportLat;   
            scheduledNodeIds.emplace(nodeId); // latency fixed
            iter = unscheduledNodes.erase(iter);
            // std::cout << "unscheduledNodes size: " << unscheduledNodes.size() << std::endl;
            // update the delay status and minLat of the this node if its srcNode already scheduled
            FUNode* fuNode = dynamic_cast<FUNode*>(_dfgNodeAttr[nodeId].adgNode); // mapped FU node (GPE/IOB)
            for(auto& elem : dfgNode->inputEdges()){
                int eid = elem.second;

                DFGEdge* edge = _dfg->edge(eid);
                int routeLat = _dfgEdgeAttr[eid].lat; // latNoDelay + logicLat    
                if(edge->isBackEdge()){
                    if(edge->isMemEdge()){ // not need to make sync for memory-dependent back-edge
                        continue;
                    } 
                    routeLat -= _II * edge->iterDist(); // latency due to iteration distance
                    //std::cout<<"routeLat"<<routeLat<<std::endl;
                }  

                int inPort = elem.first;               
                int srcNodeId = _dfg->edge(eid)->srcId();
                if(scheduledNodeIds.count(srcNodeId)){ // already scheduled 
                    // int routeLat = _dfgEdgeAttr[eid].lat; // latNoDelay;
                    int srcNodeLat = _dfgNodeAttr[srcNodeId].lat;     
                    // int delayRequired = maxInportLat - routeLat - srcNodeLat;   
                    int delayRequired = std::max(maxInportLat - routeLat - srcNodeLat, 0); 
                    int bitWidth = edge->bitWidth();
                    int delayUsed = std::min(delayRequired, getAvailDelay(fuNode, dfgNode, bitWidth));
                    auto delayKey = std::make_pair(fuNode->id(), bitWidth);
                    _fuDelayAttr[delayKey].delayUsed[inPort] = delayUsed;
                    _fuDelayAttr[delayKey].totalDelayUsed += delayUsed;
                }
            }
            _dfgNodeAttr[nodeId].minLat = std::max(maxInportLat - getAvailDelay(fuNode, dfgNode), 0); // input port min latency         
            // update the delay status and minLat of the dstNodes
            for(auto& outsPerPort : dfgNode->outputEdges()){
                for(auto& eid : outsPerPort.second){
                    int routeLat = _dfgEdgeAttr[eid].lat; // latNoDelay;
                    int dstNodeId = _dfg->edge(eid)->dstId();
                    // if(dstNodeId == _dfg->id()){ // connected to DFG output port
                    //     continue;
                    // }
                    DFGEdge* edge = _dfg->edge(eid);
                    if(edge->isBackEdge()){
                        // if(edge->isMemEdge()){ // not need to make sync for memory-dependent back-edge
                        //     continue;
                        // } 
                        routeLat -= _II * edge->iterDist(); // latency due to iteration distance
                    }
                    DFGNode* dstNode = _dfg->node(dstNodeId);
                    int inPort = _dfg->edge(eid)->dstPortIdx();
                    // int delayRequired = _dfgNodeAttr[dstNodeId].maxLat - routeLat - targetLat;   
                    int delayRequired = std::max(_dfgNodeAttr[dstNodeId].maxLat - routeLat - targetLat, 0);                                   
                    FUNode* dstFuNode = dynamic_cast<FUNode*>(_dfgNodeAttr[dstNodeId].adgNode); // mapped FU node
                    int bitWidth = edge->bitWidth();
                    int delayUsed = std::min(delayRequired, getAvailDelay(dstFuNode, dstNode, bitWidth));
                    auto delayKey = std::make_pair(dstFuNode->id(), bitWidth);
                    _fuDelayAttr[delayKey].delayUsed[inPort] = delayUsed;
                    _fuDelayAttr[delayKey].totalDelayUsed += delayUsed;
                    _dfgNodeAttr[dstNodeId].minLat = std::max(
                        _dfgNodeAttr[dstNodeId].maxLat -
                        getAvailDelay(dstFuNode, dstNode, bitWidth), 0);
                }
            }
            // std::cout << "//========= after scheduling node " << nodeId << " =========// \n";
            // for(auto node : _dfgNodeAttr){
            //     std::cout << "node ID:" << node.first;
            //     std::cout << ", lat:" << node.second.lat;
            //     std::cout << ", minLat:" << node.second.minLat;
            //     std::cout << ", maxLat:" << node.second.maxLat << "\n" ;
            // }
        }
    }

    // calculate the latency of DFG IO
    // calIOLat();
    // calculate the latency violation of each edge
    calEdgeLatVio();
}


// // calculate the latency of DFG IO
// void Mapping::calIOLat(){
//     // DFG input port latency
//     // int inputLat = _adg->loadLatency(); // IOB input latency
//     for(auto& insPerPort : _dfg->inputEdges()){
//         int minLat = 0;
//         int maxLat = 0x3fffffff;
//         for(auto& eid : insPerPort.second){
//             int routeLat = _dfgEdgeAttr[eid].lat; // latNoDelay;
//             int dstNodeId = _dfg->edge(eid)->dstId();
//             // if(dstNodeId == _dfg->id()){ // connected to DFG output port
//             //     continue;
//             // }
//             minLat = std::max(minLat, _dfgNodeAttr[dstNodeId].minLat - routeLat);
//             maxLat = std::min(maxLat, _dfgNodeAttr[dstNodeId].maxLat - routeLat);
//         }
//         int targetLat = std::min(maxLat, minLat);
//         _dfgInputAttr[insPerPort.first].lat = targetLat;
//     }
//     int maxLat = 0;
//     // DFG output port latency
//     for(auto& elem : _dfg->outputEdges()){
//         int eid = elem.second;
//         int routeLat = _dfgEdgeAttr[eid].lat; // latNoDelay;
//         int srcNodeId = _dfg->edge(eid)->srcId();
//         int srcNodeLat;
//         if(srcNodeId == _dfg->id()){ // connected to DFG input port
//             srcNodeLat = _dfgInputAttr[_dfg->edge(eid)->srcPortIdx()].lat;               
//         } else{
//             srcNodeLat = _dfgNodeAttr[srcNodeId].lat;
//         }
//         int targetLat = srcNodeLat + routeLat;
//         _dfgOutputAttr[elem.first].lat = targetLat;
//         maxLat = std::max(maxLat, targetLat);
//     }
//     _maxLat = maxLat;
// }


// calculate the latency violation of each edge
void Mapping::calEdgeLatVio(){
    int dfgSumVio = 0; // total edge latency violation
    int dfgMaxVio = 0; // max edge latency violation
    _vioDfgEdges.clear(); // DFG edges with latency violation
    for(auto nodeId : _dfg->topoNodes()){
        DFGNode *node = _dfg->node(nodeId);
        // node->printDfgNode();
        // int minLat = _dfgNodeAttr[nodeId].minLat; // min latency of the input ports
        int maxLat = _dfgNodeAttr[nodeId].maxLat; // max latency of the input ports =  latency - operation_latency
        int Nodelat = _dfgNodeAttr[nodeId].lat;
        FUNode* fuNode = dynamic_cast<FUNode*>(_dfgNodeAttr[nodeId].adgNode);
        int fuNodeId = fuNode->id();
        for(auto& elem : node->inputEdges()){
            int eid = elem.second;
            DFGEdge* edge = _dfg->edge(eid);     
            int bitWidth = edge->bitWidth();
            auto& fuDelayAttr = _fuDelayAttr[std::make_pair(fuNodeId, bitWidth)];
            auto& delayUsed = fuDelayAttr.delayUsed;
            int routeLat = _dfgEdgeAttr[eid].lat; // latNoDelay;
            if(edge->isBackEdge()){
                // if(edge->isMemEdge()){ // not need to make sync for memory-dependent back-edge
                //     _dfgEdgeAttr[eid].vio = 0;
                //     _dfgEdgeAttr[eid].delay = 0;
                //     continue;
                // } 
                routeLat -= _II * edge->iterDist(); // latency due to iteration distance
            }

            int srcNodeId = _dfg->edge(eid)->srcId();
            // int srcNodeLat;
            // if(srcNodeId != _dfg->id()){ // not connected to DFG input port
            int srcNodeLat = _dfgNodeAttr[srcNodeId].lat;                
            // } else{ // connected to DFG input port
            //     srcNodeLat = _dfgInputAttr[_dfg->edge(eid)->srcPortIdx()].lat;  
            // }
            // _dfgEdgeAttr[eid].lat = maxLat - srcNodeLat; // including RDU latency   
            if(srcNodeLat == Nodelat){//@yuan: fix the bug that srclatency = dstlatency
                // std::cout << node->name() << " src lat: " << srcNodeLat << " self lat: "<< Nodelat<<std::endl;
                _dfgEdgeAttr[eid].vio = 0;
                _vioDfgEdges.push_back(eid);
                dfgSumVio += 1;
                _totalViolation = dfgSumVio;
                return;
            }      
            int requiredDelay = maxLat - srcNodeLat - routeLat; // RDU latency  
            if(requiredDelay < 0){
                // std::cout<<"eid_vio: "<<srcNodeId<<std::endl;
                // std::cout<<"requiredDelay"<<requiredDelay<<std::endl;
                // std::cout<<"maxLat"<<maxLat<<std::endl;
                // std::cout<<"srcNodeLat"<<srcNodeLat<<std::endl;
                //  std::cout<<"nodeId: "<<node->name()<<std::endl;
                // std::cout<<"routeLat"<<routeLat<<std::endl;
                _dfgEdgeAttr[eid].vio = requiredDelay;
                _vioDfgEdges.push_back(eid);
                dfgSumVio += std::abs(requiredDelay);
                _backViolation = edge->isBackEdge();
                continue; // need to increase II or re-calculate the output node's target latency
            }

            _dfgEdgeAttr[eid].delay = requiredDelay;    
            int assignedDelay; 
            if(!delayUsed.count(elem.first)){ // not assign delay for this port
                assignedDelay = std::min(getAvailDelay(fuNode, node, bitWidth), requiredDelay);
                delayUsed[elem.first] = assignedDelay;
                fuDelayAttr.totalDelayUsed += assignedDelay;
                _dfgNodeAttr[nodeId].minLat = std::max(
                    _dfgNodeAttr[nodeId].maxLat -
                    getAvailDelay(fuNode, node, bitWidth), 0);
            }else{
                assignedDelay = delayUsed[elem.first];
            }         
            assert(assignedDelay <= requiredDelay && assignedDelay >= 0); 
            if(assignedDelay < requiredDelay){ // need to add pass node to compensate the latency gap
                int vio = requiredDelay - assignedDelay;
                _dfgEdgeAttr[eid].vio = vio;
                _vioDfgEdges.push_back(eid);
                dfgSumVio += vio;
                dfgMaxVio = std::max(dfgMaxVio, vio);
            } else {
                _dfgEdgeAttr[eid].vio = 0;
            }
        }    
    }
    _totalViolation = dfgSumVio;
    _maxViolation = dfgMaxVio;
}


// insert pass-through DFG nodes into a copy of current DFG
void Mapping::insertPassDfgNodes(DFG* newDfg){
    *newDfg = *_dfg;
    int maxNodeId = newDfg->nodes().rbegin()->first; // std::map auto sort the key
    int maxEdgeId = newDfg->edges().rbegin()->first; 
    // int maxDelay = 1;
    // for(auto& elem : _dfgNodeAttr){
    //     auto adgNode = elem.second.adgNode;
    //     if(adgNode){
    //         maxDelay = dynamic_cast<GPENode*>(adgNode)->maxDelay() + 1;
    //         break;
    //     }
    // }
    // int maxInsertNodesPerEdge = 2;
    for(int eid : _vioDfgEdges){ // DFG edges with latency violation
        int vio = _dfgEdgeAttr[eid].vio; // maybe add multiple nodes according to vio   
        int num = 1; // std::min(maxInsertNodesPerEdge, std::max(1, vio/maxDelay));  
        DFGEdge* e = newDfg->edge(eid);
        int srcId = e->srcId();
        int dstId = e->dstId();
        int srcPortIdx = e->srcPortIdx();
        int dstPortIdx = e->dstPortIdx();
        bool isBackedge = e->isBackEdge();
        int iterDist = e->iterDist();
        newDfg->delEdge(eid);
        int lastId = srcId;
        int lastPort = srcPortIdx;
        for(int i = 0; i < num; i++){
            DFGNode* newNode = new DFGNode();
            newNode->setId(++maxNodeId);
            newNode->setName("pass"+std::to_string(maxNodeId));
            newNode->setOperation("PASS");
            newDfg->addNode(newNode);
            DFGEdge* e1 = new DFGEdge(lastId, maxNodeId);
            e1->setId(++maxEdgeId); 
            e1->setSrcPortIdx(lastPort);
            e1->setDstPortIdx(0);
            newDfg->addEdge(e1);
            lastId = maxNodeId;
            lastPort = 0;
        }           
        DFGEdge* e2 = new DFGEdge(maxNodeId, dstId);
        e2->setId(++maxEdgeId); 
        e2->setSrcPortIdx(0);
        e2->setDstPortIdx(dstPortIdx);     
        if(isBackedge){
            e2->setBackEdge(true);
            e2->setIterDist(iterDist);
        }          
        newDfg->addEdge(e2);
    }
}

int Mapping::evaluateII(){
    int II = 1;
    for(auto& elem: _dfg->backEdgeLoops()){
        DFGEdge *edge = _dfg->edge(elem.first);
        int srcId = edge->srcId();
        int dstId = edge->dstId();
        int dstInportLat = dfgNodeAttr(dstId).lat - _dfg->node(dstId)->opLatency();
        int srclat = dfgNodeAttr(srcId).lat;  //
        int routeLat = _dfgEdgeAttr[elem.first].lat ; // latNoDelay
        int iterDist = std::max(edge->iterDist(), 1); // >= 1
        int newII = (srclat + routeLat - dstInportLat + iterDist -1) / iterDist; // round up
        II = std::max(newII, II);
    }
    auto& _backEdgeLoops = _dfg->backEdgeLoops();
    for(auto& elem: _dfg->backEdges()){
        if(_backEdgeLoops.count(elem)){
            continue;
        }
        DFGEdge *edge = _dfg->edge(elem);
        int srcId = edge->srcId();
        int dstId = edge->dstId();
        int dstInportLat = dfgNodeAttr(dstId).lat - _dfg->node(dstId)->opLatency();
        int srclat = dfgNodeAttr(srcId).lat;
        int routeLat = _dfgEdgeAttr[elem].lat ; // latNoDelay
        int iterDist = std::max(edge->iterDist(), 1); // >= 1
        int newII = (srclat + routeLat - dstInportLat + iterDist -1) / iterDist; // round up
        II = std::max(newII, II);

    }
    return std::max(II, _II);
}
