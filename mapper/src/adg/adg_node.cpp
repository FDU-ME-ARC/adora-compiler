
#include "adg/adg_node.h"


// ===================================================
//   ADGNode functions
// ===================================================

// get config info for sub-module
const CfgDataLoc& ADGNode::configInfo(int id){
    assert(_configInfo.count(id));
    return _configInfo[id];
}


// add config info for sub-module
void ADGNode::addConfigInfo(int id, CfgDataLoc subModuleCfg){
    _configInfo[id] = subModuleCfg;
}


void ADGNode::printADGNode(){
    // printGraphNode();
    // std::cout << "cfgBlkIdx: " << _cfgBlkIdx << std::endl;
}


void ADGNode::print(){
    // printADGNode();
}



// ===================================================
//   FUNode functions
// ===================================================

// add supported operation
void FUNode::addOperation(std::string op){
    _operations.emplace(op);
}

// delete supported operation
void FUNode::delOperation(std::string op){
    _operations.erase(op);
}

// check if the operation is supported
bool FUNode::opCapable(std::string op){
    return _operations.count(op);
}

// set numOperands and resize _operandInputs 
void FUNode::setNumOperands(int numOperands){ 
    _numOperands = numOperands; 
    _operandInputs.resize(numOperands);
}

int FUNode::numOperands(int bitWidth){
    auto it = _numOperandsByWidth.find(bitWidth);
    return it == _numOperandsByWidth.end() ? 0 : it->second;
}

void FUNode::setNumOperands(int bitWidth, int numOperands){
    _numOperandsByWidth[bitWidth] = numOperands;
    _operandInputsByWidth[bitWidth].resize(numOperands);
    addBitWidth(bitWidth);
}

int FUNode::maxDelay(int bitWidth){
    auto it = _maxDelayByWidth.find(bitWidth);
    return it == _maxDelayByWidth.end() ? 0 : it->second;
}

void FUNode::setMaxDelay(int bitWidth, int maxDelay){
    _maxDelayByWidth[bitWidth] = maxDelay;
    addBitWidth(bitWidth);
}

// get input ports connected to this operand
const std::set<int>& FUNode::operandInputs(int opeIdx){
    assert(_operandInputs.size() > opeIdx);
    return _operandInputs[opeIdx]; 
}

// add input port connected to this operand
void FUNode::addOperandInputs(int opeIdx, int inputIdx){
    assert(_operandInputs.size() > opeIdx);
    if(inputs().count(inputIdx)){
        _operandInputs[opeIdx].emplace(inputIdx);
    }
}


// delete input port connected to this operand
void FUNode::delOperandInputs(int opeIdx, int inputIdx){
    assert(_operandInputs.size() > opeIdx);
    // if(inputs().count(inputIdx)){
    _operandInputs[opeIdx].erase(inputIdx);
    // }
}


// add input ports connected to this operand
void FUNode::addOperandInputs(int opeIdx, std::set<int> inputIdxs){
    assert(_operandInputs.size() > opeIdx);
    _operandInputs[opeIdx] = inputIdxs;
}


// get which operand this input is connected
int FUNode::getOperandIdx(int inputIdx){
    for(int i = 0; i < _numOperands; i++){
        if(_operandInputs[i].count(inputIdx)){
            return i;
        }
    }
    return -1;
}

const std::set<int>& FUNode::operandInputs(int bitWidth, int operandIdx){
    static const std::set<int> empty;
    auto it = _operandInputsByWidth.find(bitWidth);
    if(it == _operandInputsByWidth.end() || operandIdx < 0 ||
       operandIdx >= static_cast<int>(it->second.size())) return empty;
    return it->second[operandIdx];
}

void FUNode::addOperandInput(int bitWidth, int operandIdx, int inputIdx){
    auto it = _operandInputsByWidth.find(bitWidth);
    assert(it != _operandInputsByWidth.end());
    assert(operandIdx >= 0 && operandIdx < static_cast<int>(it->second.size()));
    it->second[operandIdx].emplace(inputIdx);
}

void FUNode::setOperandInputs(int bitWidth, int operandIdx, const std::set<int>& inputIdxs){
    auto it = _operandInputsByWidth.find(bitWidth);
    assert(it != _operandInputsByWidth.end());
    assert(operandIdx >= 0 && operandIdx < static_cast<int>(it->second.size()));
    it->second[operandIdx] = inputIdxs;
}

int FUNode::getOperandIdx(int bitWidth, int inputIdx){
    auto it = _operandInputsByWidth.find(bitWidth);
    if(it == _operandInputsByWidth.end()) return -1;
    for(int i = 0; i < static_cast<int>(it->second.size()); ++i){
        if(it->second[i].count(inputIdx)) return i;
    }
    return -1;
}


void FUNode::printFU(){
    // printADGNode();
    // std::cout << "operations: ";
    // for(auto& elem : _operations){
    //     std::cout << elem << ", ";        
    // }
    // std::cout << "\nmaxDelay: " << _maxDelay << std::endl;
    // std::cout << "numOperands: " << _numOperands << std::endl;
    // std::cout << "operandInputs: " << std::endl;
    // int i = 0;
    // for(auto& elem : _operandInputs){
    //     std::cout << i++ << ": ";
    //     for(auto in : elem){
    //         std::cout << in << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << "cfgIdMap: ";
    // for(auto& elem : cfgIdMap){
    //     std::cout << elem.first << ", " << elem.second << "; ";      
    // }
    // std::cout << std::endl;
}

void FUNode::print(){
    printFU();
}



// ===================================================
//   GPENode functions
// ===================================================

void GPENode::print(){
    // printFU();
    // std::cout << "numRfReg: " << _numRfReg << std::endl;
}

int GPENode::getOperandIdxLUT(int lutOperandIdx){
    int fineOperands = numOperands(1);
    return fineOperands > _numInputLUT
        ? lutOperandIdx + fineOperands - _numInputLUT
        : lutOperandIdx;
}


// ===================================================
//   IOBNode functions
// ===================================================

void IOBNode::print(){
    // printFU();
    // std::cout << "index: " << _index << std::endl;
    // std::cout << "iocCfgIdMap: ";
    // for(auto& elem : cfgIdMap){
    //     std::cout << elem.first << ", " << elem.second << "; ";      
    // }
    // std::cout << std::endl;
}


// ===================================================
//   GIBNode functions
// ===================================================

// if there are registers in the output port 
bool GIBNode::outReged(int idx){
    return _outReged[idx];
}

// if there are registers in the output port 
void GIBNode::setOutReged(int idx, bool reged){
    _outReged[idx] = reged;
}

// inputs connected to each output
std::set<int> GIBNode::out2ins(int outPort){
    if(_out2ins.count(outPort)){
        return _out2ins[outPort];
    } else {
        return {};
    }
}


// outputs connected to each input
std::set<int> GIBNode::in2outs(int inPort){
    if(_in2outs.count(inPort)){
        return _in2outs[inPort];
    } else {
        return {};
    }
}

// add input connected to the output
void GIBNode::addOut2ins(int outPort, int inPort){
    if(_inputs.count(inPort) && _outputs.count(outPort)){
        _out2ins[outPort].emplace(inPort);
    }
}

// add output connected to the input
void GIBNode::addIn2outs(int inPort, int outPort){
    if(_inputs.count(inPort) && _outputs.count(outPort)){
        _in2outs[inPort].emplace(outPort);
    }
}


// check if the input and output port are connected
bool GIBNode::isInOutConnected(int inPort, int outPort){
    if(_out2ins.count(outPort)){
        for(auto in : _out2ins[outPort]){
            if(in == inPort){
                return true;
            }
        }        
    }
    return false;
}


void GIBNode::print(){
    // printADGNode();
    // std::cout << "trackReged: " << _trackReged << std::endl;
    // std::cout << "outReged: " << std::endl;
    // for(auto& elem : _outReged){
    //     std::cout << elem.first << ": " << elem.second << std::endl;
    // }
    // std::cout << "out2ins: " << std::endl;
    // for(auto& elem : _out2ins){
    //     for(auto sec : elem.second)
    //         std::cout << elem.first << ", " << sec << std::endl;
    // }
    // std::cout << "in2outs: " << std::endl;
    // for(auto& elem : _in2outs){
    //     for(auto sec : elem.second)
    //         std::cout << elem.first << ", " << sec << std::endl;
    // }
}

