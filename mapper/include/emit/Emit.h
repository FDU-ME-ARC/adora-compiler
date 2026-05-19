#ifndef ADORA_EMIT_BASE_H
#define ADORA_EMIT_BASE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Analysis/CallGraph.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/raw_ostream.h"

#include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "ADORA/Dialect/ADORA/Utility/Utility.h"

#include "mapper/mapper.h"
#include "mapper/mapper_sa.h"
#include "mapper/io_scheduler.h"

using namespace mlir;
using namespace mlir::ADORA;


//////////////////////////
// Tool functions in EmitUtility.cpp
//////////////////////////
std::vector<std::string> split_str_by_char(const std::string &s, const char delimiter);


/// @brief Retrieves the C++ type string corresponding to a given MLIR type.
/// @param valType An MLIR type for which the corresponding C++ type string is to be determined.
/// @return A C++ type representation as a string, such as "float", "double", "int16_t", etc. 
///         Returns an error string if the type is unsupported.
std::string getEmitType(const mlir::Type valType);

/// @brief Retrieves the C++ type string for a given MLIR value by determining its type.
/// @param v An MLIR value whose type is used to determine the C++ type string.
/// @return A string representing the C++ type corresponding to the value's MLIR type. 
///         Emits an error and aborts if the type is unsupported.
std::string getEmitType(const mlir::Value v);

/// @brief A function to retrieve the IOB IDs that can access a specified SPAD bank.
/// @param adg A pointer to an instance of the ADG class, which contains the mapping of IOBs to their connected SPAD banks.
/// @param bankId The identifier for the SPAD bank to check for accessibility.
/// @return A vector of IOB IDs that are connected to the specified SPAD bank.
std::vector<int> spadBankToIobs(ADG* adg, int bankId);

/// @brief A utility function to check if a block access operation can be simplified
/// @param op The candidate operation, either a datablockload or datablockstore
/// @return True if the operation is simplified, otherwise false
int64_t getByteSizeFromMemref(mlir::MemRefType memref);
int64_t getByteSizeFromMemref(mlir::TypedValue<mlir::MemRefType> memref);

namespace mlir{
namespace ADORA{
/// @brief A function to simplify affine map of datablockload or datablockstore op
/// @param op 
void SimplifyBlockAccessOp(mlir::Region& region);
void SimplifyBlockAccessOp(mlir::ModuleOp& m);

//////
/// Check whether a DataBlockStoreOp is the last of one kernel in one Block.
bool IsLastBlockStoreOp(ADORA::DataBlockStoreOp op);
}}


class Op_Name_C{
public:
  int id;
  std::string type;
  std::string name(){return type + "_" + std::to_string(id);}
  Op_Name_C(std::string type, int id): type(type) , id(id) {}
  Op_Name_C(int id, std::string type): type(type) , id(id) {}
  Op_Name_C(){}
};


/**
 * @class BYTES_LIST is a class designed to manage and manipulate an array of bytes.
 * It allows dynamic setting and retrieval of bit values within the byte array.
 * Additionally, it provides a method to output the current contents of the byte list for easy debugging and analysis.
 * 
 */ 
class BYTES_LIST{
public:
  std::vector<std::string> bytes; /// TODO: change std::vector<std::string> to std::vector<char>

  BYTES_LIST(){}
  BYTES_LIST(int n){ bytes.resize(n ,"0x00");}
  BYTES_LIST(int n, const std::string init){bytes.resize(n, init);}

  void setBitTo(const int bit_idx, const bool bit);
  int getBit(int bit_idx);

  std::string getByte(int byte_idx){return bytes[byte_idx];}
  std::string operator[](int byte_idx){return getByte(byte_idx);}

  int size(){return bytes.size();}

  std::vector<std::string> As32b();

  void dump() const;

private:
  void ensureSize(int bit_idx);
};



/**
 * @class BaseEmitter
 * @brief A base class for implementing various emission strategies for kernels and data operations.
 * 
 * The BaseEmitter class serves as an abstract base class for emitting code from various 
 * intermediate representations of kernels. It defines a set of pure virtual functions 
 * that must be implemented by derived classes to provide specific emission logic 
 * for functions, blocks, and configurations. 
 * 
 */
class BaseEmitter{
public:
  /////////////////////////////////
  // The following functions must be implemented in your derived class.
  // These functions are defined as pure virtual functions in the base class
  // to ensure each subclass provides implementations according to its specific needs.
  /////////////////////////////////
  virtual void emitFunctionHead(func::FuncOp &funcop, llvm::raw_ostream &os) = 0;
  virtual void emitBlock(mlir::Block &block, llvm::raw_ostream &os) = 0;
  virtual std::string GenerateCGRAConfig(ADORA::KernelOp& kernel, MapperSA* mapper) = 0;
  virtual std::string GenerateCGRAConfig(ADORA::KernelOp& kernel, Configuration configuration, ADG* adg) = 0;
  virtual void GenerateCGRACFGAndEXE(ADORA::KernelOp& kernel, MapperSA* mapper) = 0;
  virtual void GenerateCGRACFGAndEXE(ADORA::KernelOp& kernel, Configuration configuration, ADG* adg) = 0;
  virtual bool emitCGRACallFunction(llvm::raw_ostream &os) = 0;
  /////////////////////////////////

  BaseEmitter(mlir::ModuleOp& m): _moduleop(m){};
  ~BaseEmitter(){};

  unsigned getIndent(){return _currentIndent;}

  llvm::SmallVector<dfgIoInfo> getDfgIoInfosFromBlockLoad(ADORA::DataBlockLoadOp op) {return _LoadToDfgIoInfos[op];}
  dfgIoInfo getDfgIoInfosFromBlockStore(ADORA::DataBlockStoreOp op) {return _StoreToDfgIoInfo[op];}

  llvm::SmallVector<std::pair<int, dfgIoInfo>> getSPMInfosFromBlockLoad(ADORA::DataBlockLoadOp op) {return _LoadToSPMInfos[op];}
  std::pair<int, dfgIoInfo> getSPMInfosFromLocalAlloc(ADORA::LocalMemAllocOp op) {return _LocalAllocToSPMInfo[op];}
  

  // @brief Establishes placement constraints for local memory allocations within a given kernel operation.
  /// @param kernel A reference to an ADORA::KernelOp object representing the kernel operation.
  /// @param mapper A pointer to a MapperSA object used for accessing the data flow graph (DFG) and architecture description graph (ADG).
  void preestablishPlacementConstraints(ADORA::KernelOp& kernel, MapperSA* mapper);

  llvm::SmallDenseMap<ADORA::KernelOp, Configuration> KnToConfiguration;
  void setMapResult(ADORA::KernelOp k, MapperSA* mapper);
  void setTileEnsForKernel(ADORA::KernelOp k);
  void setTileEnsForEachKernel();
  
  /// @brief Get SPAD information (which bank to transfer data, data size...) for every data block load.
  ///        This information is store in _LoadToDfgIoInfos/_StoreToDfgIoInfo
  /// @param kernel Kernel which has been translated to CDFG
  /// @param mapper The SA mapper which has completed mapping 
  void DataBlockOperationsToSPADInfo(mlir::ADORA::KernelOp& kernel, MapperSA* mapper);
//   llvm::SmallDenseMap<ADORA::KernelOp, std::pair<std::string, int>> KnToCfgArrayInfo;   // map : kernelop -> (cfg array's name, cfgnum)
//   llvm::SmallDenseMap<ADORA::KernelOp, std::string> KnToCfgData; // map : kernelop -> total cfg array declaration
//   llvm::SmallDenseMap<ADORA::KernelOp, std::string> KnToCfgExe;

  void setADG(ADG* _) {_adg = _;}
  ADG* getADG(){return _adg;}

  void setTotalTileNum(int n){_totalTileNum = n;};
  int getTotalTileNum(){return _totalTileNum;};
  BYTES_LIST getIobEns(KernelOp& kernel){return _kernel_to_iob_ens[kernel];};
  void setTileEns(KernelOp& kernel, BYTES_LIST ens){_kernel_to_tile_ens[kernel] = ens;};
  BYTES_LIST getTileEns(KernelOp& kernel){return _kernel_to_tile_ens[kernel];};
  llvm::SmallDenseMap<ADORA::DataBlockLoadOp, llvm::SmallVector<dfgIoInfo>> 
    getLoadToDfgIoInfosMap(){return _LoadToDfgIoInfos;};
  llvm::SmallDenseMap<ADORA::DataBlockStoreOp, dfgIoInfo>
    getStoreToDfgIoInfoMap(){return _StoreToDfgIoInfo;};  
  llvm::SmallDenseMap<ADORA::DataBlockLoadOp, llvm::SmallVector< std::pair<int, dfgIoInfo> > >
    getLoadToSPMInfosMap(){return _LoadToSPMInfos;};
  llvm::SmallDenseMap<ADORA::LocalMemAllocOp, std::pair<int, dfgIoInfo>>
    getLocalAllocToSPMMap(){return _LocalAllocToSPMInfo;};
protected:
  mlir::ModuleOp _moduleop;
  std::stringstream _CFGandEXE;
  std::map<KernelOp, std::map<int, dfgIoInfo>> _kernel_to_dfg_io_infos;
  // std::map<int, int> _dfgIoSpadAddrs;
  ADG* _adg = nullptr;

  int _totalTileNum = 1;
  std::map<KernelOp, BYTES_LIST> _kernel_to_iob_ens;
  std::map<KernelOp, BYTES_LIST> _kernel_to_tile_ens;
  // uint64_t _iob_ens = 0;
  llvm::SmallDenseMap<ADORA::DataBlockLoadOp, llvm::SmallVector<dfgIoInfo>> _LoadToDfgIoInfos;
  llvm::SmallDenseMap<ADORA::DataBlockStoreOp, dfgIoInfo> _StoreToDfgIoInfo;

  llvm::SmallDenseMap<ADORA::DataBlockLoadOp, llvm::SmallVector< std::pair<int, dfgIoInfo> > > _LoadToSPMInfos; // int : spm bank idx, dfgioinfo
  llvm::SmallDenseMap<ADORA::LocalMemAllocOp, std::pair<int, dfgIoInfo>> _LocalAllocToSPMInfo; /// localmallocOp->(bank id, dfgio)

  /// Variables for emitting
  unsigned _currentIndent = 0;
  virtual void addIndent(){_currentIndent += 2;}
  virtual void reduceIndent(){_currentIndent = _currentIndent >= 2 ? _currentIndent - 2 : 0;}
};


#endif // ADORA_EMIT_BASE_H