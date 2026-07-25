#ifndef __ADG_IR_H__
#define __ADG_IR_H__

#include <iostream>
#include <fstream>
#include <map>
#include <algorithm>
#include <tuple>
#include "nlohmann/json.hpp"
#include "adg/adg.h"

using json = nlohmann::json; 


class ADGIR
{
private:
    ADG* _adg;
    int _maxLUTInput = 0;
    std::map<int, std::string> _iobModeNames;
    // parse ADG json object
    ADG* parseADG(json& adgJson);
    // parse ADGNode from sub-modules json object 
    ADGNode* parseADGNode(json& nodeJson);
    // parse ADGNode from instances json object, 
    // modules<moduleId, <ADGNode*, used>>,  
    ADGNode* parseADGNode(json& nodeJson, std::map<int, std::pair<ADGNode*, bool>>& modules);
    // parse ADGEdge json object
    void parseADGEdges(ADG* adg, json& edgeJson);
    // Normalize AuFORA's per-tile fine_grained_networks into ordinary ADG
    // nodes and width-qualified edges.
    void parseFineGrainedNetworks(ADG* adg, json& networksJson);
    // analyze the connections among the internal sub-modules for GPENode, fill _operandInputs 
    void analyzeIntraConnect(GPENode* node);
    // analyze the connections among the internal sub-modules for IOBNode, fill _operandInputs 
    void analyzeIntraConnect(IOBNode* node);
    // analyze the connections among the internal sub-modules for GIBNode
    // fill _out2ins, _in2outs 
    void analyzeIntraConnect(GIBNode* node);
    // analyze if there are registers in the output ports of GIB
    // fill _outReged
    void analyzeOutReg(ADG* adg, GIBNode* node);
    // post-process the ADG nodes
    void postProcess(ADG* adg);
public:
    ADGIR(std::string filename);
    ~ADGIR();
    ADG* getADG(){ return _adg; }
};





#endif
