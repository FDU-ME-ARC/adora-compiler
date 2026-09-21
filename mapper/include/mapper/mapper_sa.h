#ifndef __MAPPER_SA_H__
#define __MAPPER_SA_H__

#include "mapper/mapper.h"
// #include "emit/EmitCGRACall.h"
#include <cmath>
#include <random>


#include "mlir/IR/BuiltinOps.h"
#include "ADORA/Dialect/ADORA/IR/ADORA.h"



// mapper using simulated annealing algorithm
class MapperSA : public Mapper
{
private:
    // max iteration number of mapping 
    int _maxIters;
    // if optimize mapping objective
    bool _objOpt;
    const float MAX_TEMP = 100.0; // max temperature
    const float MIN_TEMP = 0.01;  // min temperature
    // cache various cost
    int _dfgLat;
    int _mappedAdgNodeNum;
    long _ioDeps; // IO dependence on previous tasks
    std::mt19937 _randomEngine;
    std::uniform_real_distribution<float> _unitDistribution{0.0f, 1.0f};

    
public:
    MapperSA(ADG* adg, int timeout_ms = 600000, int maxIter = 30,
             bool objOpt = true, unsigned randomSeed = 0);
    // MapperSA(ADG* adg, DFG* dfg);
    MapperSA(ADG* adg, DFG* dfg, int timeout_ms = 600000, int maxIter = 30,
             bool objOpt = true, unsigned randomSeed = 0);
    ~MapperSA();
    // generate random 0~1 float value
    float randfloat(){ return _unitDistribution(_randomEngine); }
    void setMaxIters(int num){ _maxIters = num; }
    void setObjOpt(bool objOpt){ _objOpt = objOpt; }
    // PnR and Data Synchronization on the same DFG, i.e. without modifying DFG
    bool pnrSyncSameDfg(float T0, int maxItersPerTemp, int maxItersNoImprv);
    // PnR and Data Synchronization
    // return -1 : preMapCheck failed; 0 : fail; 1 : success
    int pnrSync(float T0, int maxItersPerTemp = 30, int maxItersNoImprv = 200, bool modifyDfg = true);
    // PnR, Data Synchronization, and objective optimization
    bool pnrSyncOpt();
    // map the DFG to the ADG, mapper API
    virtual bool mapper();
    
protected:
    // // PnR with SA temperature(max = 100)
    // int pnr(Mapping* mapping, int temp);
    // unmap some DFG nodes with SA temperature(max = 100)
    void unmapSA(Mapping* mapping, float temp);
    // incremental PnR, try to map all the left DFG nodes based on current mapping status
    bool incrPnR(Mapping* mapping);
    // try to map one DFG node to one of its candidates
    // return selected candidate index
    int tryCandidates(Mapping* mapping, DFGNode* dfgNode, const std::vector<ADGNode*>& candidates);
    // find candidates for one DFG node based on current mapping status
    std::vector<ADGNode*> findCandidates(Mapping* mapping, DFGNode* dfgNode, int range, int maxCandidates);
    // get the shortest distance between ADG node and the available IOB
    int getAdgNode2IODist(Mapping* mapping, int id);
    // // get the shortest distance between ADG node and the available ADG input
    // int getAdgNode2InputDist(Mapping* mapping, int id);
    // // get the shortest distance between ADG node and the available ADG output
    // int getAdgNode2OutputDist(Mapping* mapping, int id);
    // sort candidates according to their distances with the mapped src and dst ADG nodes of this DFG node 
    // return sorted index of candidates
    std::vector<int> sortCandidates(Mapping* mapping, DFGNode* dfgNode, const std::vector<ADGNode*>& candidates);
    
    // objective funtion
    float objFunc(Mapping* mapping, bool init);
    // SA: the probablity of accepting new solution
    bool metropolis(float diff, float temp);
    // annealing funtion
    float annealFunc(float temp);
};




#endif
