#ifndef __CONFIGURATION_H__
#define __CONFIGURATION_H__

#include "mapper/mapping.h"
#include <unordered_map>
#include <utility>

#define MapHasKey(_map, _key) (_map.find(_key)!=_map.end())

// configuration data
struct CfgData{
    int len; // config data length
    std::vector<uint32_t> data; // config data
    CfgData(){}
    CfgData(int len_) : len(len_){}
    CfgData(int len_, uint32_t data_){
        len = len_;
        data.push_back(data_);
    }
    CfgData(int len_, std::vector<uint32_t> data_){
        len = len_;
        data = data_;
    }
    CfgData& operator=(const CfgData& that){
        if(this == &that) return *this;
        len = that.len;
        data = that.data;
        isRuntimeCfg = that.isRuntimeCfg;
        return *this;
    }

    //// @jhlou: for pingpong
    bool isRuntimeCfg = false;
};


// configuration data packet
struct CfgDataPacket{
    unsigned addr; // config address
    std::vector<uint32_t> data; // config data
    CfgDataPacket(unsigned addr_) : addr(addr_){}
    CfgDataPacket(unsigned addr_, uint32_t data_){
        addr = addr_;
        data.push_back(data_);
    }
    void print(){
        std::cout << std::hex << "[CDP] addr 0x"<< addr << ", data :";
        for(uint32_t data_ : data){
            std::cout << std::hex  << " 0x" << data_;
        }
        std::cout << std::dec << std::endl;
    }
};


class ConfigReplaceInfo32bit{
public:
    uint32_t addr = 0;
    uint32_t Idx0, Idx1;
    uint32_t lshift = 0, rshift = 0;        /// Var needs to be left shift with lshift bits

    uint32_t getMask(){/// mask of Var
        assert(lshift == 0 || rshift == 0);
        if(lshift != 0)
            return ~((uint32_t)0xFFFF << lshift);
        else
            return ~(uint32_t)0xFFFF >> rshift;
    }
    void print(){
        std::cout << "ConfigReplaceInfo32bit: "  << "addr " << addr
                << ", Idx0 " << Idx0 << ", Idx1 " << Idx1
                << ", lshift " << lshift << ", rshift " << rshift
                << ", mask "<<  getMask()
                << std::endl;
    }
};

class ConfigReplaceInfo16bit{
public:
    uint16_t addr = 0;
    uint16_t Idx0, Idx1;
    uint16_t lshift = 0, rshift = 0;        /// Var needs to be left shift with lshift bits

    uint16_t getMask(){/// mask of Var
        assert(lshift == 0 || rshift == 0);
        if(lshift != 0)
            return ~((uint16_t)0xFFFF << lshift);
        else
            return ~(uint16_t)0xFFFF >> rshift;
    }
    void print(){
        std::cout << "ConfigReplaceInfo16bit: "  << "addr " << addr
                << ", Idx0 " << Idx0 << ", Idx1 " << Idx1
                << ", lshift " << lshift << ", rshift " << rshift
                << ", mask "<<  getMask()
                << std::endl;
    }
};
typedef ConfigReplaceInfo16bit ConfigReplaceInfo;

#define IsConstStrExpr(_str) (_str == "__const__" || _str == "-" || _str.empty())

// CGRA Configuration
class Configuration
{
private:
    Mapping* _mapping;
    std::map<int, int> _dfgIoSpadAddrs; // base address in spad for each DFG IO, <id, addr>
    static void addIobDelayCfg(std::map<int, CfgData> &cfg, IOBNode *node,
                               ADGNode *delayNode,
                               const std::map<int, int> &delayUsed);
    
    // std::string getVarConfigFromAddr(std::map<int, std::vector<int>> lsb2addr,int  addr);
public:
    Configuration() : _mapping(nullptr) {}
    Configuration(Mapping* mapping) : _mapping(mapping) {}
    Configuration operator=(const Configuration& other){
        _mapping = other._mapping;
        _dfgIoSpadAddrs = other._dfgIoSpadAddrs;
        return *this;
    }
    ~Configuration(){}
    const std::map<int, int>& dfgIoSpadAddrs(){ return _dfgIoSpadAddrs; }
    void setDfgIoSpadAddr(int id, int addr){ _dfgIoSpadAddrs[id] = addr; }
    // get config data for GPE, return<LSB-location, CfgData>
    std::map<int, CfgData> getGpeCfgData(GPENode* node);
    // get config data for GIB, return<LSB-location, CfgData>
    std::map<int, CfgData> getGibCfgData(GIBNode* node);
    // get config data for IOB, return<LSB-location, CfgData>
    std::map<int, CfgData> getIobCfgData(IOBNode* node);
    // get config data for ADG node
    void getNodeCfgData(ADGNode* node, std::vector<CfgDataPacket>& cfg);
    // get config data for ADG
    void getCfgData(std::vector<CfgDataPacket>& cfg);
    // dump config data
    void dumpCfgData(std::ostream& os);

    // @jhlou
    std::map<int, int> addAdditionalDelayForMERGEOp(DFGNode* dfgNode,std::map<int, int>& delayUsed);
    int addAdditionalLatencyForMERGEOp(DFGNode* dfgNode, int latency);
    // int addAdditionalLatencyForOpsFolloingMERGE(DFGNode* dfgNode, int latency);

    /// For variable configurations
    VariableConfig* getConfigVariable(ADGNode* node);
    bool IsADGNodeConfigVariable(ADGNode* node);

    std::map<std::string, std::vector<ConfigReplaceInfo>> VarReplaceInfo;
    void printVarReplaceInfo();

    /// pingpong support
    void getNodePingpongCfgData(ADGNode* node, std::vector<CfgDataPacket>& cfg, int pingpong_phase);
    void getPingpongCfgData(std::vector<CfgDataPacket>& cfg_ping, std::vector<CfgDataPacket>& cfg_pong);
    std::map<int, CfgData> getIobPingpongCfgData(IOBNode* node, bool en_pingpong = 0, int pingpong_phase = 0);
    
    /// get the related tiles of the configuration
    std::set<int> getConfiguredTiles();


    //// static tool functions
    static void addCfgData(std::map<int, CfgData> &cfg, const CfgDataLoc &loc, uint32_t data);
    static void addCfgData(std::map<int, CfgData> &cfg, const CfgDataLoc &loc, uint64_t data);
    static void addCfgData(std::map<int, CfgData> &cfg, const CfgDataLoc &loc, const std::vector<uint32_t> &data);
  
    int findMask32(uint32_t value, int begin, int l);
    int findMask16(uint16_t value, short begin, short l);
    // void Valid(){
    //     _mapping->getDFG()
    // }
};






#endif
