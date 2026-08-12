#include "../inc/llvm_cdfg_node.h"




// LLVMCDFGNode::LLVMCDFGNode(Instruction *ins, LLVMCDFG *parent)
// {
//     _parent = parent;
//     setInstruction(ins);
// }

// void LLVMCDFGNode::setInstruction(Instruction *ins)
// { 
//     _instruction = ins; 
//     if(dyn_cast<PHINode>(ins)){
//         return;
//     }
//     // may have multiple constant operand, create CONST node separately
//     // for(int i = 0; i < ins->getNumOperands(); i++){ 
//     //     if(ConstantInt *CI = dyn_cast<ConstantInt>(ins->getOperand(i))){
//     //         setConstVal(CI->getSExtValue());
//     //         break; // only one constant
//     //     }
//     // }
//     if(dyn_cast<LoadInst>(ins)){
//         setDataBits(ins->getType()->getPrimitiveSizeInBits());
//         llvm::Value *address_of_load;
//         if(auto GEP = dyn_cast<GetElementPtrInst>(ins->getOperand(0))){
//             address_of_load = GEP->getOperand(0);
//         }else if(auto GEP = dyn_cast<ConstantExpr>(ins->getOperand(0))){
//             address_of_load = GEP->getOperand(0);                            
//         }else{
//             address_of_load = ins->getOperand(0);
//         }
//         setLSaddress(address_of_load);
//     }else if(StoreInst *SI = dyn_cast<StoreInst>(ins)){
//         setDataBits(SI->getValueOperand()->getType()->getPrimitiveSizeInBits());
//         llvm::Value *address_of_store;
//         if(auto GEP = dyn_cast<GetElementPtrInst>(ins->getOperand(1))){
//             address_of_store = GEP->getOperand(0);
//         }else if(auto GEP = dyn_cast<ConstantExpr>(ins->getOperand(1))){
//             address_of_store = GEP->getOperand(0);                            
//         }else{
//             address_of_store = ins->getOperand(1);
//         }
//         setLSaddress(address_of_store);
//     }else if(dyn_cast<GetElementPtrInst>(ins)){
//         setDataBits(32);
//     }else if(dyn_cast<AllocaInst>(ins)){
//         setDataBits(32);
//     }else if(dyn_cast<PointerType>(ins->getType())){
//         setDataBits(32);
//     }
// }


std::string LLVMCDFGNode::getName()
{
    std::string name;
    // if(!_finalInstruction.empty()){
    //     name = _finalInstruction;
    // }else if(_instruction && _customInstruction.empty()){
    //     name = _instruction->getOpcodeName();
    // }else{
    //     name = _customInstruction;
    // }
    // name += std::to_string(_id);
    name = _TypeName + std::to_string(_id);
    return name;
}

// inputs : predecessor instructions
void LLVMCDFGNode::addInputNode(LLVMCDFGNode *node, int idx, bool isBackEdge, CondVal cond)
{
    _inputPortMap[idx] = node;
    NodeInfo info;
    info.idx = idx;
    info.isBackEdge = isBackEdge;
    info.cond = cond;

    if(std::find(_inputNodes.begin(), _inputNodes.end(), node) != _inputNodes.end()){
        auto &infos = _inputInfoMap[node];
        if(std::any_of(infos.begin(), infos.end(),
                       [idx](const NodeInfo &existing) {
                           return existing.idx == idx;
                       })){
            errs()<<"%%%%%"<<this->getName()<<"'s inputlist already has "<<node->getName()<<"\n";
            return;
        }
        infos.push_back(info);
        return;
    }
    _inputNodes.push_back(node);
    // if(node->instruction() != NULL && dyn_cast<PHINode>(node->instruction())){
    //     _inputInfoMap[node].isPHI = true;
    // }else{
    //     _inputInfoMap[node].isPHI = false;
    // }
    _inputInfoMap[node].push_back(info);
}


// outputs : successor instructions
void LLVMCDFGNode::addOutputNode(LLVMCDFGNode *node, bool isBackEdge, CondVal cond)
{
    if(std::find(_outputNodes.begin(), _outputNodes.end(), node) != _outputNodes.end()){
        errs()<<"%%%%%"<<this->getName()<<"'s outputlist already has "<<node->getName()<<"\n";
        return;
    }
    _outputNodes.push_back(node);
    // if(node->instruction() != NULL && dyn_cast<PHINode>(node->instruction())){
    //     _outputInfoMap[node].isPHI = true;
    // }else{
    //     _outputInfoMap[node].isPHI = false;
    // }
    _outputInfoMap[node].isBackEdge = isBackEdge;
    _outputInfoMap[node].cond = cond;
}


// input index
void LLVMCDFGNode::setInputIdx(LLVMCDFGNode *node, int idx)
{
    // _inputInfoMap[node].idx = idx;

    //@jhlou
    if(_inputInfoMap[node].size() == 1){
        _inputInfoMap[node][0].idx = idx;
    }
    else if(_inputInfoMap[node].size() == 0){
        NodeInfo info;
        info.idx = idx;
        _inputInfoMap[node].push_back(info);
    }
    else{
        /// set the last one;
        _inputInfoMap[node].at(_inputInfoMap[node].size()-1).idx = idx;
    }

} 


// input -> this node is back-edge
void LLVMCDFGNode::setInputBackEdge(LLVMCDFGNode *node, bool isBackEdge)
{
    // _inputInfoMap[node].isBackEdge = isBackEdge;
    //@jhlou
    if(_inputInfoMap[node].size() == 1){
        _inputInfoMap[node][0].isBackEdge = isBackEdge;
    }
    else if(_inputInfoMap[node].size() == 0){
        NodeInfo info;
        info.isBackEdge = isBackEdge;
        _inputInfoMap[node].push_back(info);
    }
    else{
        /// set the last one;
        _inputInfoMap[node].at(_inputInfoMap[node].size()-1).isBackEdge = isBackEdge;
    }
} 

// conditional dependence between inputs and this node
void LLVMCDFGNode::setInputCondVal(LLVMCDFGNode *node, CondVal cond)
{
    // _inputInfoMap[node].cond = cond;

    //@jhlou
    if(_inputInfoMap[node].size() == 1){
        _inputInfoMap[node][0].cond = cond;
    }
    else if(_inputInfoMap[node].size() == 0){
        NodeInfo info;
        info.cond = cond;
        _inputInfoMap[node].push_back(info);
    }
    else{
        _inputInfoMap[node].at(_inputInfoMap[node].size()-1).cond = cond;
    }
}   

// this node -> output is back-edge
void LLVMCDFGNode::setOutputBackEdge(LLVMCDFGNode *node, bool isBackEdge)
{
    _outputInfoMap[node].isBackEdge = isBackEdge;
}  

// conditional dependence between this node and output
void LLVMCDFGNode::setOutputCondVal(LLVMCDFGNode *node, CondVal cond)
{
    _outputInfoMap[node].cond = cond;
}   

// input index
int LLVMCDFGNode::getInputIdx(LLVMCDFGNode *node)
{
    if(_inputInfoMap.count(node)){
        return _inputInfoMap[node][0].idx;
    }
    return -1;
}

// @jhlou input index
std::vector<int> LLVMCDFGNode::getInputIndices(LLVMCDFGNode *node)
{  
    std::vector<int> indices;
    for(auto elem : _inputPortMap){
        int idx = elem.first;
        LLVMCDFGNode * inputnode = elem.second;
        if(inputnode == node){
            indices.push_back(idx);
        }
    }
    return indices;
}

LLVMCDFGNode* LLVMCDFGNode::getInputPort(int idx)
{
    return _inputPortMap[idx];
} 

void LLVMCDFGNode::setInputPort(LLVMCDFGNode *node, int idx)
{
    _inputPortMap[idx] = node;
} 

void LLVMCDFGNode::swapInputPorts(int firstIdx, int secondIdx)
{
    if (firstIdx == secondIdx)
        return;

    auto firstIt = _inputPortMap.find(firstIdx);
    auto secondIt = _inputPortMap.find(secondIdx);
    bool hasFirst = firstIt != _inputPortMap.end();
    bool hasSecond = secondIt != _inputPortMap.end();
    LLVMCDFGNode *firstNode = hasFirst ? firstIt->second : nullptr;
    LLVMCDFGNode *secondNode = hasSecond ? secondIt->second : nullptr;

    if (hasFirst)
        _inputPortMap.erase(firstIdx);
    if (hasSecond)
        _inputPortMap.erase(secondIdx);
    if (hasFirst)
        _inputPortMap[secondIdx] = firstNode;
    if (hasSecond)
        _inputPortMap[firstIdx] = secondNode;

    for (auto &entry : _inputInfoMap) {
        for (NodeInfo &info : entry.second) {
            if (info.idx == firstIdx)
                info.idx = secondIdx;
            else if (info.idx == secondIdx)
                info.idx = firstIdx;
        }
    }
}

// input -> this node is back-edge
bool LLVMCDFGNode::isInputBackEdge(LLVMCDFGNode *node)
{
    if(_inputInfoMap.count(node)){
        return _inputInfoMap[node][0].isBackEdge;
    }
    return false;
}   

// conditional dependence between inputs and this node
CondVal LLVMCDFGNode::getInputCondVal(LLVMCDFGNode *node)
{
    if(_inputInfoMap.count(node)){
        return _inputInfoMap[node][0].cond;
    }
    return UNCOND;
}

// this node -> output is back-edge
bool LLVMCDFGNode::isOutputBackEdge(LLVMCDFGNode *node)  
{
    if(_outputInfoMap.count(node)){
        return _outputInfoMap[node].isBackEdge;
    }
    return false;
}

// conditional dependence between this node and output
CondVal LLVMCDFGNode::getOutputCondVal(LLVMCDFGNode *node)
{
    if(_outputInfoMap.count(node)){
        return _outputInfoMap[node].cond;
    }
    return UNCOND;
} 


// inputs : predecessor instructions
int LLVMCDFGNode::delInputNode(LLVMCDFGNode *node)
{
    _inputNodes.erase(std::remove(_inputNodes.begin(), _inputNodes.end(), node), _inputNodes.end());
    // int idx = _inputInfoMap[node].idx; //@jhlou
    int idx = _inputInfoMap[node][0].idx;    
    _inputInfoMap.erase(node);
    return idx;
} 


// outputs : successor instructions
void LLVMCDFGNode::delOutputNode(LLVMCDFGNode *node)
{
    _outputNodes.erase(std::remove(_outputNodes.begin(), _outputNodes.end(), node), _outputNodes.end());
    _outputInfoMap.erase(node);
}


// loop-carried dependence source instructions
DependInfo LLVMCDFGNode::getSrcDep(LLVMCDFGNode *node)
{
    assert(_srcDepInfoMap.count(node));
    return _srcDepInfoMap[node];
}

// loop-carried dependence destination instructions
DependInfo LLVMCDFGNode::getDstDep(LLVMCDFGNode *node)
{
    assert(_dstDepInfoMap.count(node));
    return _dstDepInfoMap[node];
}

// loop-carried dependence source instructions
void LLVMCDFGNode::addSrcDep(LLVMCDFGNode *node, DependInfo info)
{
    _srcDepInfoMap[node] = info;
}

// loop-carried dependence destination instructions
void LLVMCDFGNode::addDstDep(LLVMCDFGNode *node, DependInfo info)
{
    _dstDepInfoMap[node] = info;
}


void LLVMCDFGNode::delSrcDep(LLVMCDFGNode *node)
{
    _srcDepInfoMap.erase(node);
}


void LLVMCDFGNode::delDstDep(LLVMCDFGNode *node)
{
    _dstDepInfoMap.erase(node);
}


// bool LLVMCDFGNode::isConditional() {
// 	if(_instruction != NULL){
// 		if(_instruction->getOpcode() == Instruction::Br){
// 			return true;
// 		}
// 	}
// 	return false;
// }
