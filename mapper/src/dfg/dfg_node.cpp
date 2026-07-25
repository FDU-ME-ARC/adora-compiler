
#include "dfg/dfg_node.h"

// ===================================================
//   DFGNode functions
// ===================================================

// set operation, latency, commutative according to operation name
void DFGNode::setOperation(std::string operation){ 
    bool fineIoOp = operation == "CINPUT" || operation == "COUTPUT";
    if(!Operations::opCapable(operation) && operation != "LUT" && !fineIoOp){
        std::cout << operation << " is not supported!" << std::endl;
        exit(1);
    }
    _operation = operation; 
    if(operation == "LUT" || fineIoOp){
        setOpLatency(1);
        setCommutative(false);
        setAccumulative(false);
    }else{
        setOpLatency(Operations::latency(operation));
        setCommutative(Operations::isCommutative(operation));
        setAccumulative(Operations::isAccumulative(operation));
    }

    // if(operation == "ISEL" || operation == "CISEL"){
    //     setInitSelection(true);
    // }

}


int DFGNode::numInputs(){ 
    return (hasImm())? (_inputs.size()+1) : _inputs.size(); 
}


int DFGNode::numInputs(int bits) {
    return GraphNode::numInputs(bits)
        + ((hasImm() && bitWidth() == bits) ? 1 : 0);
}

void DFGNode::printDfgNode(){
    printGraphNode();
    std::cout << "operation: " << _operation << ", opLatency: " << _opLatency << std::endl;
    std::cout << "commutative: " << _commutative << std::endl;
    std::cout << "accumulative: " << _accumulative << std::endl;
    std::cout << "imm: " << _imm << ", immIdx: " << _immIdx << std::endl;    
}

void DFGNode::print(){
    printDfgNode();
}



// ===================================================
//   DFGIONode functions
// ===================================================

void DFGIONode::print(){
    printDfgNode();
    std::cout << "memRefName: " << _memRefName << std::endl;
    std::cout << "pattern: ";
    for(auto &elem : _pattern){
        std::cout << elem.first << " " << elem.second << " ";
    }
    std::cout << std::endl;    
}


// ===================================================
//   VariableConfig functions
// ===================================================

bool VariableConfig::IsPatternVariable(){
    for(auto pair : pattern){
        if((pair.first != "__const__" && pair.first != "-" && !pair.first.empty())
         ||(pair.second != "__const__" && pair.second != "-" && !pair.second.empty())){
            return true;
        }
    }
    return false;
}

bool VariableConfig::IsAccVariable(){
    return (initVal != "__const__" && initVal != "-" && !initVal.empty() )
        || (cycles  != "__const__" && cycles  != "-" && !cycles.empty() )
        || (interval!= "__const__" && interval!= "-" && !interval.empty())
        || (repeats != "__const__" && repeats != "-" && !repeats.empty());
}

bool VariableConfig::IsVariable(){
    return IsPatternVariable() || IsAccVariable() 
        || (memOffset != "__const__" && memOffset != "-" && !memOffset.empty())
        || (reducedmemOffset != "__const__" && reducedmemOffset != "-" && !reducedmemOffset.empty());
}


void VariableConfig::print(){
  
   switch (type)
   {
   case NodeT::IONode:
        std::cout << "Variable Config ";
        std::cout << "IO Node type:" ;
        std::cout << ", memOffset: "<< memOffset << "\n";
        std::cout << ", reducedmemOffset: "<< reducedmemOffset << "\n";
        if(pattern.size() == 0)
            std::cout << "pattern is not set yet." ;
        else {
            std::cout << "pattern: " ;
            for(auto& elem : pattern){
                std::cout << "(" << elem.first 
                         << "," << elem.second << ") ";
            }
            std::cout << "\n" ;
        }
        break;
    case NodeT::ACCNode:
        std::cout << "Variable Config ";
        std::cout << "ACC Node type:" << "\n";
        std::cout << "initVal: "<< initVal 
                    << ", cycles: " << cycles 
                    << ", interval: "<< interval 
                    << ", repeats: "<< repeats << "\n";
        break;
   
   default:
        std::cout << "Non-Variable Config. " << "\n";
    break;
   }
}
