
#include "mapper/visualize.h"


Graphviz::Graphviz(Mapping* mapping, std::string dirname) : _mapping(mapping), _dirname(dirname) {std::filesystem::create_directory(_dirname);}


// // create name for DFG node
// std::string Graphviz::getDfgNodeName(int id, int idx, bool isDfgInput){
//     std::string name;
//     DFG* dfg = _mapping->getDFG();
//     if(id != dfg->id()){
//         name = dfg->node(id)->name();
//     }else if(isDfgInput){
//         name = dfg->inputName(idx);
//     }else{
//         name = dfg->outputName(idx);
//     }
//     return name;
// }


void Graphviz::drawDFG(){
    std::string filename = _dirname + "/mapped_dfg.dot";
    std::ofstream ofs(filename);
    DFG* dfg = _mapping->getDFG();
    int dfgId = dfg->id();
    ofs << "Digraph G {\n";
    // the input/output latency is the latency of valid data, 
    // not when start to load or have stored the data in SRAM
    // for(auto& elem : dfg->inputs()){
    //     std::string name = getDfgNodeName(dfgId, elem.first);
    //     std::string quoteName = "\"" + name + "\"";
    //     int lat = _mapping->dfgInputAttr(elem.first).lat;
    //     ofs << quoteName << "[label = \"\\N\\nlat=" << lat << "\"];\n";
    // }
    // for(auto& elem : dfg->outputs()){
    //     std::string name = getDfgNodeName(dfgId, elem.first, false);
    //     std::string quoteName = "\"" + name + "\"";
    //     int srcNodeId = elem.second.first;
    //     std::string srcName = getDfgNodeName(srcNodeId, elem.second.second);
    //     std::string quoteSrcName = "\"" + srcName + "\"";
    //     int lat = _mapping->dfgOutputAttr(elem.first).lat;
    //     ofs << quoteName << "[label = \"\\N\\nlat=" << lat << "\"];\n";
    //     ofs << quoteSrcName << "->" << quoteName << std::endl;
    // }
    for(auto& elem : dfg->nodes()){
        auto node = elem.second;
        auto& attr = _mapping->dfgNodeAttr(node->id());
        auto name = node->name();
        std::string quoteName = "\"" + name + "\"";
        ofs << quoteName << "[label = \"\\N\\nlat=" << attr.lat;
        if(node->hasImm())
            ofs << "\\nimm=" << node->imm() << "\\nimmIdx=" << node->immIdx();
        if(node->isLoopIndexAcc())
            ofs << "\\nloop_index_acc=1";
        ofs << "\"];\n";
        // for(auto& input : node->inputs()){
        //     int srcNodeId = input.second.first;
        //     std::string srcName = dfg->node(srcNodeId)->name();
        //     std::string quoteSrcName = "\"" + srcName + "\"";
        //     ofs << quoteSrcName << "->" << quoteName << ";\n";
        // }
        for(auto& elem : node->inputEdges()){
            int eid = elem.second;
            DFGEdge* edge = dfg->edge(eid);
            int srcNodeId = edge->srcId();
            int operandIdx = edge->dstPortIdx();
            auto& edgeAttr = _mapping->dfgEdgeAttr(eid);
            int edgeLat = edgeAttr.lat; // edge latency not including delay unit

            /// @jhlou: add rdu delay information to dfgdot
            int portDelay = edgeAttr.delay; // delay cycles used by this port

            std::string srcName = dfg->node(srcNodeId)->name();
            std::string quoteSrcName = "\"" + srcName + "\"";
            ofs << quoteSrcName << "->" << quoteName << "[label = \"lat=" << edgeLat << 
                                                "\\nop=" << operandIdx <<
                                                "\\nrdudelay=" << portDelay << "\"];\n";
        }
    }
    ofs << "}\n";
}


void Graphviz::originaldrawDFG(){
    std::string filename = _dirname + "/mapped_dfg.dot";
    std::ofstream ofs(filename);
    DFG* dfg = _mapping->getDFG();
    int dfgId = dfg->id();
    ofs << "Digraph G {\n";
    // the input/output latency is the latency of valid data, 
    // not when start to load or have stored the data in SRAM
    // for(auto& elem : dfg->inputs()){
    //     std::string name = getDfgNodeName(dfgId, elem.first);
    //     std::string quoteName = "\"" + name + "\"";
    //     int lat = _mapping->dfgInputAttr(elem.first).lat;
    //     ofs << quoteName << "[label = \"\\N\\nlat=" << lat << "\"];\n";
    // }
    // for(auto& elem : dfg->outputs()){
    //     std::string name = getDfgNodeName(dfgId, elem.first, false);
    //     std::string quoteName = "\"" + name + "\"";
    //     int srcNodeId = elem.second.first;
    //     std::string srcName = getDfgNodeName(srcNodeId, elem.second.second);
    //     std::string quoteSrcName = "\"" + srcName + "\"";
    //     int lat = _mapping->dfgOutputAttr(elem.first).lat;
    //     ofs << quoteName << "[label = \"\\N\\nlat=" << lat << "\"];\n";
    //     ofs << quoteSrcName << "->" << quoteName << std::endl;
    // }
    for(auto& elem : dfg->nodes()){
        auto node = elem.second;
        auto& attr = _mapping->dfgNodeAttr(node->id());
        auto name = node->name();
        std::string quoteName = "\"" + name + "\"";
        ofs << quoteName << "[label = \"\\N\\nlat=" << attr.lat << "\"];\n";
        // for(auto& input : node->inputs()){
        //     int srcNodeId = input.second.first;
        //     std::string srcName = dfg->node(srcNodeId)->name();
        //     std::string quoteSrcName = "\"" + srcName + "\"";
        //     ofs << quoteSrcName << "->" << quoteName << ";\n";
        // }
        for(auto& elem : node->inputEdges()){
            int eid = elem.second;
            DFGEdge* edge = dfg->edge(eid);
            int srcNodeId = edge->srcId();
            int operandIdx = edge->dstPortIdx();
            auto& edgeAttr = _mapping->dfgEdgeAttr(eid);
            int edgeLat = edgeAttr.lat; // edge latency not including delay unit
            std::string srcName = dfg->node(srcNodeId)->name();
            std::string quoteSrcName = "\"" + srcName + "\"";
            ofs << quoteSrcName << "->" << quoteName << "[label = \"lat=" << edgeLat << "\\nop=" << operandIdx << "\"];\n";
        }
    }
    ofs << "}\n";
}


// // create name for ADG node
// std::string Graphviz::getAdgNodeName(int id){
//     ADG* adg = _mapping->getADG();
//     assert(id != adg->id());  
//     return adg->node(id)->type() + std::to_string(id);
// }


void Graphviz::drawADG(){
    std::string filename = _dirname + "/mapped_adg.dot";
    std::ofstream ofs(filename);
    ADG* adg = _mapping->getADG();
    int adgId = adg->id();
    DFG* dfg = _mapping->getDFG();
    int dfgId = dfg->id();
    ofs << "Digraph G {\nlayout = sfdp;\noverlap = scale;\n";
    for(auto& elem : adg->nodes()){
        auto node = elem.second;
        auto& attr = _mapping->adgNodeAttr(node->id());
        auto name = node->name();
        auto dfgNode = _mapping->mappedNode(node);
        if(dfgNode){
            ofs << name << "[label = \"" << name << "\\nDFG:" << dfgNode->name() << "\""
                << ", color = red];\n";
        // }else if(node->type() == "IOB" && _mapping->isUsedAsIB(node)){ // IOB used as IB
        //     int dfgInport = _mapping->adgNodeAttr(node->id()).dfgEdgePass.begin()->edge->srcPortIdx(); // all the edge has the same srcPort
        //     ofs << name << "[label = \"" << name << "\\nDFG:" << getDfgNodeName(dfgId, dfgInport) << "\""
        //         << ", color = red];\n";
        // }else if(node->type() == "IOB" && _mapping->isUsedAsOB(node)){ // IOB used as OB
        //     int dfgOutport = _mapping->adgNodeAttr(node->id()).dfgEdgePass.begin()->edge->dstPortIdx(); // all the edge has the same dstPort
        //     ofs << name << "[label = \"" << name << "\\nDFG:" << getDfgNodeName(dfgId, dfgOutport, false) << "\""
        //         << ", color = red];\n";
        }else if(node->type() == "GIB" && dynamic_cast<GIBNode*>(node)->trackReged()){
            ofs << name << "[label = \"" << name << "\", color = blue];\n";
        }else{
            ofs << name << "[label = \"" << name << "\", color = black];\n";
        }
        std::set<int> srcNodeIds;
        for(auto& input : node->inputs()){
            int srcNodeId = input.second.first;
            // only keep one connection
            srcNodeIds.emplace(srcNodeId);        
            // std::string srcName = getAdgNodeName(srcNodeId, input.second.second);
            // ofs << srcName << "->" << name << "[color = gray80];\n";        
        }
        for(auto srcNodeId : srcNodeIds){
            std::string srcName = adg->node(srcNodeId)->name();
            ofs << srcName << "->" << name << "[color = gray80];\n";  
        }
    }
    ofs << "edge [colorscheme=paired12];\n";
    for(auto& elem : dfg->edges()){
        int eid = elem.first;
        auto& attr = _mapping->dfgEdgeAttr(eid);
        auto& edgeLinks = attr.edgeLinks;
        int i = edgeLinks.size();
        for(auto& edgeLink : edgeLinks){
            i--;
            auto adgNode = edgeLink.adgNode;
            auto name = adgNode->name();
            if(i > 0){
                ofs << name << "->";             
            }else{
                ofs << name << "[weight = 4, color = " << ((eid%12)+1) << "];\n";
            }
        }
    }
    ofs << "}\n";
}


// print edge path
void Graphviz::printDFGEdgePath(){
    DFG* dfg = _mapping->getDFG();
    for(auto& elem : dfg->edges()){
        int eid = elem.first;
        auto& attr = _mapping->dfgEdgeAttr(eid);
        auto& edgeLinks = attr.edgeLinks;
        // print edge path
        auto dfgSrcNodeName = dfg->node(elem.second->srcId())->name();
        auto dfgDstNodeName = dfg->node(elem.second->dstId())->name();
        std::cout << dfgSrcNodeName << " => " << dfgDstNodeName << ":\n";
        int i = edgeLinks.size();
        for(auto& edgeLink : edgeLinks){
            auto adgNodeName = edgeLink.adgNode->name();
            int adgNodeSrcPort = edgeLink.srcPort;
            int adgNodeDstPort = edgeLink.dstPort;
            std::cout << "(" << adgNodeName << ", " << adgNodeSrcPort << ", " << adgNodeDstPort;
            i--;
            if(i > 0){
                std::cout << ") -> ";
            }else{
                std::cout << ");\n";
            }
        }
    }
}


void Graphviz::dumpMappedRoutes(){
    std::string filename = _dirname + "/mapped_routes.tsv";
    std::ofstream ofs(filename);
    DFG* dfg = _mapping->getDFG();

    ofs << "edge_id\tsrc_dfg\tdst_dfg\tdst_operation\tlogical_operand"
        << "\tdst_adg\tdst_physical_input\tsrc_latency\troute_latency"
        << "\trdu_delay\tarrival_latency\ttarget_latency\n";
    for(auto& elem : dfg->edges()){
        int edgeId = elem.first;
        DFGEdge* edge = elem.second;
        if(!_mapping->isRouted(edgeId) || edge->isMemEdge() ||
           edge->dstPortIdx() < 0){
            continue;
        }

        const auto& edgeAttr = _mapping->dfgEdgeAttr(edgeId);
        const auto& edgeLinks = edgeAttr.edgeLinks;
        assert(!edgeLinks.empty());
        DFGNode* srcNode = dfg->node(edge->srcId());
        DFGNode* dstNode = dfg->node(edge->dstId());
        const auto& dstAttr = _mapping->dfgNodeAttr(dstNode->id());
        int srcLatency = _mapping->dfgNodeAttr(srcNode->id()).lat;
        int routeLatency = edgeAttr.lat;
        int rduDelay = edgeAttr.delay;
        int arrivalLatency = srcLatency + routeLatency + rduDelay;
        int targetLatency = dstAttr.lat - dstNode->opLatency();

        ofs << edgeId << '\t' << srcNode->id() << '\t' << dstNode->id()
            << '\t' << dstNode->operation() << '\t' << edge->dstPortIdx()
            << '\t' << dstAttr.adgNode->id()
            << '\t' << edgeLinks.back().srcPort
            << '\t' << srcLatency << '\t' << routeLatency
            << '\t' << rduDelay << '\t' << arrivalLatency
            << '\t' << targetLatency << '\n';
    }
}


// dump mapped DFG IO ports with mapped IOB and latency annotated
void Graphviz::dumpDFGIO(){
    std::string filename = _dirname + "/mapped_dfgio.txt";
    std::ofstream ofs(filename);
    ADG* adg = _mapping->getADG();
    DFG* dfg = _mapping->getDFG();
    // int dfgId = dfg->id();
    int ioid = 0;
    ofs << "# format: DFG-IO-Tag, DFG-IO-Name, ADG-IOB-Id, ADG-IOB-Index, DFG-IO-Latency\n";
    auto outNodeIds = dfg->getOutNodes(); // OUTPUT/STORE nodes
    for(auto& id : dfg->ioNodes()){
        auto dfgIONode = dfg->node(id);
        auto& attr =  _mapping->dfgNodeAttr(id);
        int iobId = attr.adgNode->id();
        int iobIdx = dynamic_cast<IOBNode*>(adg->node(iobId))->index();
        std::string tag = dfgIONode->operation() + "_" + std::to_string(ioid++);    
        int lat = attr.lat;
        if(!outNodeIds.count(id)){ // INPUT/LOAD node latency is the input port latency
            lat = attr.lat - dfgIONode->opLatency();
        }    
        ofs << tag << ", " << dfgIONode->name() << ", " << iobId << ", " << iobIdx << ", " << lat << std::endl;
    }
    // latency in the dfgInputAttr.lat is the latency of the loaded data
    // the latency used for IOB config should be the latency of starting to load data
    // int loadLat = adg->loadLatency(); // input load latency
    // for(auto& elem : dfg->inputs()){
    //     std::string dfgIOname = getDfgNodeName(dfgId, elem.first);
    //     auto& attr =  _mapping->dfgInputAttr(elem.first);
    //     int iobId = attr.iobId;
    //     int iobIdx = dynamic_cast<IOBNode*>(adg->node(iobId))->index();
    //     // auto adgIOname = getAdgNodeName(adgId, adgIOPort);
    //     std::string tag = "input_" + std::to_string(ioid++);
    //     ofs << tag << ", " << dfgIOname << ", " << iobId << ", " << iobIdx << ", " << (attr.lat - loadLat) << std::endl;
    // }
    // for(auto& elem : dfg->outputs()){
    //     std::string dfgIOname = getDfgNodeName(dfgId, elem.first, false);
    //     auto& attr =  _mapping->dfgOutputAttr(elem.first);
    //     int iobId = attr.iobId;
    //     int iobIdx = dynamic_cast<IOBNode*>(adg->node(iobId))->index();
    //     // auto adgIOname = getAdgNodeName(adgId, adgIOPort, false);
    //     std::string tag = "output_" + std::to_string(ioid++);
    //     ofs << tag << ", " << dfgIOname << ", " << iobId << ", " << iobIdx << ", " << attr.lat << std::endl;
    // }
}
