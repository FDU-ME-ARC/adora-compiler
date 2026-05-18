#include "emit/Emit.h"
#include "mlir/Dialect/Affine/Utils.h"
//===----------------------------------------------------------------------===//
// Some tool functions
//===----------------------------------------------------------------------===//
std::vector<std::string> split_str_by_char(const std::string &s, const char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

std::string getEmitType(const mlir::Type valType){
   // Handle float types.
  if (valType.isa<Float32Type>())
    return std::string("float");
  else if (valType.isa<Float64Type>())
    return std::string("double");
  else if (valType.isa<BFloat16Type>())
    return std::string("bfloat");
  else if (valType.isa<Float16Type>())
    return std::string("float16");

  // Handle integer types.
  else if (valType.isa<IndexType>())
    return std::string("int");
  else if (auto intType = valType.dyn_cast<mlir::IntegerType>()) {
    if(intType.isInteger(16)){    
      return std::string("int16_t");
    }
    else if(intType.isInteger(32)){
      return std::string("int32_t");
    }
    else if(intType.isInteger(64)){ 
      return std::string("int64_t");
    }
    else if(intType.isUnsignedInteger(16)){    
      return std::string("uint16_t");
    }
    else if(intType.isUnsignedInteger(32)){
      return std::string("uint32_t");
    }
    else if(intType.isUnsignedInteger(64)){ 
      return std::string("uint64_t");
    }
  }

  return "-----------------error-----------------";
}

std::string getEmitType(const mlir::Value v){
  auto valType = v.getType();
  std::string type_str = getEmitType(valType);

  if(type_str == "-----------------error-----------------"){
    v.getDefiningOp()->emitError("Unsupported data type.");
    abort();
  }

  return type_str;
}

static void CloneUnitAttr(Operation* from, Operation* to){
 for (auto &namedAttr : from->getAttrs()) {
    if (isa<UnitAttr>(namedAttr.getValue())) {
      to->setAttr(namedAttr.getName(), namedAttr.getValue());
    }
  }
}

/// @brief A tool function to check whether a block access op is simplified
/// @param op can be datablockload or datablockstore op
/// @return
template <typename opT> bool IsSimplified(opT op){
  for(AffineExpr expr : op.getAffineMap().getResults()){
    switch (expr.getKind())
    {
    case AffineExprKind::DimId :
      continue;
      break;
    
    case AffineExprKind::Constant :
      continue;
      break;
  
    default:
      return false;
      break;
    }
  }
  return true;
}

/// @brief A utility function to check if a block access operation can be simplified
/// @param op The candidate operation, either a datablockload or datablockstore
/// @return True if the operation is simplified, otherwise false
int64_t getByteSizeFromMemref(mlir::MemRefType memref) {
    mlir::Type elementType = memref.getElementType();

    if (!elementType.isIntOrFloat()) {
        llvm::errs() << "Unsupported element type in MemRef: " << elementType << "\n";
        return -1;
    }
    unsigned elemBitWidth = elementType.getIntOrFloatBitWidth();

    int64_t numElements = 1;
    for (int64_t dim : memref.getShape()) {
        if (dim == mlir::ShapedType::kDynamic) {
            llvm::errs() << "Dynamic dimension found in MemRef: " << memref << "\n";
            assert(false);
        }
        numElements *= dim;
    }

    int64_t totalBytes = (elemBitWidth / 8) * numElements;
    return totalBytes;
}

int64_t getByteSizeFromMemref(mlir::TypedValue<mlir::MemRefType> memref) {
  return getByteSizeFromMemref(memref.getType());
}

/// @brief A function to retrieve the IOB IDs that can access a specified SPAD bank.
/// @param adg A pointer to an instance of the ADG class, which contains the mapping of IOBs to their connected SPAD banks.
/// @param bankId The identifier for the SPAD bank to check for accessibility.
/// @return A vector of IOB IDs that are connected to the specified SPAD bank.
std::vector<int> spadBankToIobs(ADG* adg, int bankId) {
    std::vector<int> accessibleIobs; // Vector to store IOBs that can access the specified bankId
    const auto& iobToBanksMap = adg->iobToSpadBanks(); // Get the mapping of all IOBs to their SPAD banks

    // Iterate over all IOBs
    for (const auto& entry : iobToBanksMap) {
      int iobId = entry.first; // Current IOB ID
      const std::vector<int>& connectedBanks = entry.second; // SPAD banks connected to the current IOB

      // Check if the current IOB is connected to the specified bankId
      if (std::find(connectedBanks.begin(), connectedBanks.end(), bankId) != connectedBanks.end()) {
        accessibleIobs.push_back(iobId); // If connected to bankId, add it to the result
      }
    }

    return accessibleIobs; // Return all IOB IDs that can access the specified bankId
}

/// @brief A function to simplify affine map of datablockload or datablockstore op
/// @param op 
void mlir::ADORA::SimplifyBlockAccessOp(mlir::Region& region){
  region.walk([&](ADORA::DataBlockLoadOp blockload) {
    /// If block load op is simple, such as:
    /// %4 = ADORA.BlockLoad %arg2 [%arg3, %arg4, %arg5, %arg6]
    /// Then there is no need to simplify.
    if(IsSimplified(blockload))
      return WalkResult::advance(); ;

    mlir::OpBuilder b(blockload); 
    auto resultOperands =
        ::mlir::affine::expandAffineMap(b, blockload.getLoc(), blockload.getAffineMap(), blockload.getMapOperands());

    SmallVector<AffineExpr, 4> Exprs;
    for(int operandidx = 0; operandidx < (*resultOperands).size(); operandidx++){
      Value operand = (*resultOperands)[operandidx];
      // blockload.getOperation()->setOperand(operandidx + 1, operand);
      // operand.dump();
      AffineExpr Expr = b.getAffineDimExpr(operandidx);
      Exprs.push_back(Expr);
    }
    
    AffineMap map = AffineMap::get((*resultOperands).size(), /*symbolCount=*/0, Exprs, b.getContext());   /// stores corresponding AffineMap of above memIVs
    // map.dump();

    ADORA::DataBlockLoadOp newBlockLoad = b.create<ADORA::DataBlockLoadOp>\
                (blockload.getLoc(), blockload.getOriginalMemref(), map, *resultOperands, blockload.getResultType());

    // newBlockLoad.getOperation()->setAttrs(blockload.getOperation()->getAttrDictionary());
    newBlockLoad.setKernelName(blockload.getKernelName().str());
    newBlockLoad.setId(blockload.getId().str());
    if(blockload.hasStrides())
      newBlockLoad.setStrides(blockload.getStrides());
    CloneUnitAttr(blockload.getOperation(), newBlockLoad.getOperation());

    blockload.getOperation()->replaceAllUsesWith(newBlockLoad);
    blockload.erase();
  });

  //// DataBlockStoreOp
  region.walk([&](ADORA::DataBlockStoreOp blockstore) {
    /// If block load op is simple, such as:
    /// %4 = ADORA.BlockLoad %arg2 [%arg3, %arg4, %arg5, %arg6]
    /// Then there is no need to simplify.
    if(IsSimplified(blockstore))
      return WalkResult::advance(); ;

    mlir::OpBuilder b(blockstore); 
    auto resultOperands =
        ::mlir::affine::expandAffineMap(b, blockstore.getLoc(), blockstore.getAffineMap(), blockstore.getMapOperands());

    SmallVector<AffineExpr, 4> Exprs;
    for(int operandidx = 0; operandidx < (*resultOperands).size(); operandidx++){
      Value operand = (*resultOperands)[operandidx];
      // blockload.getOperation()->setOperand(operandidx + 1, operand);
      // operand.dump();
      AffineExpr Expr = b.getAffineDimExpr(operandidx);
      Exprs.push_back(Expr);
    }
    
    AffineMap map = AffineMap::get((*resultOperands).size(), /*symbolCount=*/0, Exprs, b.getContext());   /// stores corresponding AffineMap of above memIVs
    // map.dump();

    ADORA::DataBlockStoreOp newBlockStore = b.create<ADORA::DataBlockStoreOp>\
                (blockstore.getLoc(), blockstore.getSourceMemref(), blockstore.getTargetMemref(), map, *resultOperands);

    newBlockStore.setKernelName(blockstore.getKernelName().str());
    newBlockStore.setId(blockstore.getId().str());
    if(blockstore.hasStrides())
      newBlockStore.setStrides(blockstore.getStrides());
    CloneUnitAttr(blockstore.getOperation(), newBlockStore.getOperation());

    blockstore.getOperation()->replaceAllUsesWith(newBlockStore);
    blockstore.erase();

    return WalkResult::advance();
  });
}

void mlir::ADORA::SimplifyBlockAccessOp(mlir::ModuleOp& m){
  SimplifyBlockAccessOp(m.getRegion());
}

//////
/// Check whether a DataBlockStoreOp is the last of one kernel in one Block.
///
bool mlir::ADORA::IsLastBlockStoreOp(ADORA::DataBlockStoreOp op){
  Block* parentBlock = op.getOperation()->getBlock();
  bool behindThisOp = false;
  for(auto it = parentBlock->begin(); it != parentBlock->end(); it++){
    if(behindThisOp && isa<ADORA::DataBlockStoreOp>(it)){
      ADORA::DataBlockStoreOp toCheck = dyn_cast<ADORA::DataBlockStoreOp>(it);
      
      if(toCheck.getKernelName() == op.getKernelName()){
        //// this is not the last datablockop of one kernel in this block
        return false;
      }
    }
    if(&*it == op.getOperation()){
      behindThisOp = true;
    }
  }
  
  return true;
}

/////////////////////////////////////////////
//// BYTES_LIST class
/////////////////////////////////////////////
void BYTES_LIST::ensureSize(int bit_idx) {
  int byte_idx = bit_idx / 8;
  if (byte_idx >= bytes.size()) {
    bytes.resize(byte_idx + 1, "0x00");
  }
}

void BYTES_LIST::setBitTo(int bit_idx, bool bit) {
  ensureSize(bit_idx);

  int byte_idx = bit_idx / 8; 
  int bit_position = bit_idx % 8;

  std::stringstream ss;
  ss << std::hex << bytes[byte_idx].substr(2);
  unsigned int byte_val;
  ss >> byte_val;

  if (bit) {
    byte_val |= (1 << bit_position); 
  } else {
    byte_val &= ~(1 << bit_position);
  }

  std::stringstream ss_out;
  ss_out << "0x" << std::setw(2) << std::setfill('0') << std::hex << byte_val;
  bytes[byte_idx] = ss_out.str();
}

// Converts a hex string to an integer
static unsigned int hexStringToInt(const std::string &hexStr) {
  std::stringstream ss;
  ss << std::hex << hexStr.substr(2); // Skip "0x"
  unsigned int result;
  ss >> result;
  return result;
}


int BYTES_LIST::getBit(int bit_idx) {
  int byte_idx = bit_idx / 8;
  int bit_position = bit_idx % 8;

  // Check if the byte index is within the current size of bytes
  if (byte_idx >= bytes.size()) {
    return 0; // Assuming out-of-bounds bits are 0
  }

  // Get the byte value from the vector
  unsigned int byte_val = hexStringToInt(bytes[byte_idx]);

  // Extract the bit value
  int bit_val = (byte_val >> bit_position) & 1;

  return bit_val;
}

static std::string intToHexString(unsigned int value) {
  std::stringstream ss;
  ss << "0x" << std::setfill('0') << std::setw(8) << std::hex << value;
  return ss.str();
}

std::vector<std::string> BYTES_LIST::As32b() {
  std::vector<std::string> result;
  int total_bits = bytes.size() * 8;

  // Iterate over 32-bit blocks starting from bit_idx
  for (int i = 0; i < total_bits; i += 32) {
    unsigned int value = 0;
        
    // Collect up to 32 bits into value, ensuring we don't exceed total_bits
    for (int j = 31; j >= 0; --j) {
      if (i + j < total_bits) {  // Check to prevent out-of-bounds access
        value <<= 1;
        value |= getBit(i + j);
      } 
    }

    // Convert value to hex and add to result
    result.push_back(intToHexString(value));
  }

  return result;
}

void BYTES_LIST::dump() const {
    for (size_t i = 0; i < bytes.size(); ++i) {
      std::cout << bytes[i];
      if(i != bytes.size()-1)
        std::cout << ",";
    }
}

/////////////////////////////////////////////
//// Base class of emit
/////////////////////////////////////////////
void BaseEmitter::setMapResult(ADORA::KernelOp k, MapperSA* mapper){
  _adg = mapper->getADG();
  KnToConfiguration[k] = Configuration(mapper->_mapping);
}

// @brief Establishes placement constraints for local memory allocations within a given kernel operation.
/// @param kernel A reference to an ADORA::KernelOp object representing the kernel operation.
/// @param mapper A pointer to a MapperSA object used for accessing the data flow graph (DFG) and architecture description graph (ADG).
/// 
/// This function walks through module operations to find local memory allocations that match the specified kernel.
/// For each matching allocation, it identifies the associated I/O nodes and determines which I/O Blocks (IOBs) can access the allocated memory.
/// It then updates the mapper with these placement constraints for proper memory allocation and access during kernel execution.
void BaseEmitter::preestablishPlacementConstraints(ADORA::KernelOp& kernel, MapperSA* mapper){
  DFG* dfg = mapper->getDFG();
  ADG* adg = mapper->getADG();
  int sizeofBank = adg->iobSpadBankSize();
  int dataByte = adg->bitWidth() / 8;

  _moduleop.walk([&](ADORA::LocalMemAllocOp alloc) {  
    if(findElement(alloc.getKernelNameAsStrVector(), kernel.getKernelName()) != -1){
      std::string AllocName = kernel.getKernelName() + ":" + alloc.getId().str();

      if(_LocalAllocToSPMInfo.count(alloc) != 0){
        std::set<int> IONodes = dfg->ioNodes();
        for(auto& id : IONodes){
          DFGIONode* IoNode = dynamic_cast<DFGIONode*>(dfg->node(id));
          if(IoNode->memRefName() == AllocName){
            //// the memory for the ionode has already been allocated
            int bankid = getSPMInfosFromLocalAlloc(alloc).first;
            dfgIoInfo info = getSPMInfosFromLocalAlloc(alloc).second;
            std::vector<int> accessibleIobs = spadBankToIobs(adg, bankid);
            std::vector<ADGNode*> accessibleadgnodes;

            for(auto& elem : adg->nodes()){
              ADGNode* adgNode = elem.second;
              if(adgNode->type() == "IOB"){  
                IOBNode* IobNode = dynamic_cast<IOBNode*>(adgNode);
                if(std::find(accessibleIobs.begin(), accessibleIobs.end(), IobNode->index()) != accessibleIobs.end()){
                  accessibleadgnodes.push_back(adgNode);
                }
              }
            }

            mapper->preestablishPlacementConstraints(dfg->node(id), accessibleadgnodes);
          }
        }  
      }
    }
  });
}

void BaseEmitter::setTileEnsForKernel(ADORA::KernelOp kn){
  Configuration config = KnToConfiguration[kn];
  std::set<int> tiles = config.getConfiguredTiles();

  BYTES_LIST bytes((getADG()->tileNum() + 7)/8);
  for (int tile : tiles) {
    if (tile < 0) {
      assert(tiles.size() == 1);
    }
    else{
      bytes.setBitTo(tile, true);
    }
  }

  setTileEns(kn, bytes);
}

void BaseEmitter::setTileEnsForEachKernel(){
  for(auto elem : this->KnToConfiguration){
    ADORA::KernelOp kn = elem.first;
    Configuration config = elem.second;
    setTileEnsForKernel(kn);
  }
}

/// @brief Get SPAD information (which bank to transfer data, data size...) for every data block load.
///        This information is store in _LoadToDfgIoInfos/_StoreToDfgIoInfo
/// @param kernel Kernel which has been translated to CDFG
/// @param mapper The SA mapper which has completed mapping
void BaseEmitter::DataBlockOperationsToSPADInfo(ADORA::KernelOp& kernel, MapperSA* mapper){
  // mapper._sched->ioSchedule(mapper._mapping);
  DFG* dfg = mapper->getDFG();
  ADG* adg = mapper->getADG();
  int sizeofBank = adg->iobSpadBankSize();
  int dataByte = adg->bitWidth() / 8;
  
  std::vector<spadBankStatus> bank_status;
  // DenseMap<int, dfgIoInfo> dfg_io_infos;
  bank_status.assign(adg->numIobNodes(), {0, 0, 0, 0}); /*{iob, used, start, end}*/

  std::map<int, dfgIoInfo> dfg_io_infos;
  BYTES_LIST iob_ens(adg->numIobNodes()/8);

  /// set occupied banks
  //// TODO: what about _LoadToSPMInfos ??
  for(auto elem : _LocalAllocToSPMInfo){
    ADORA::LocalMemAllocOp alloc = elem.getFirst();
    if(findElement(alloc.getKernelNameAsStrVector(), kernel.getKernelName()) != -1){
      std::string AllocName = kernel.getKernelName() + ":" + alloc.getId().str();
      int selBank = elem.getSecond().first;
      dfgIoInfo ioInfo = elem.getSecond().second;

      std::set<int> IONodes = dfg->ioNodes();
      for(auto& id : IONodes){
        DFGIONode* IoNode = dynamic_cast<DFGIONode*>(dfg->node(id));
        if(IoNode->memRefName() == AllocName){
          int memSize = dynamic_cast<DFGIONode*>(dfg->node(id))->memSize();
          int spadDataByte = adg->cfgSpadDataWidth() / 8; // dual ports of cfg-spad have the same width 
          memSize = (memSize + spadDataByte - 1) / spadDataByte * spadDataByte; // align to spadDataByte
          auto& attr =  mapper->_mapping->dfgNodeAttr(id);
          int iobId = attr.adgNode->id();
          int iobIdx = dynamic_cast<IOBNode*>(adg->node(iobId))->index();
          iob_ens.setBitTo(iobIdx, 1);    // iob_ens |= 1 << iobIdx; 

          // int selStart = bankStatus[0].second;
          // int selBank = availBanks[0];
          // dfgIoInfo ioInfo = = getSPMInfosFromLocalAlloc(alloc).second;   

          // bank_status[selBank].used = ioInfo.isStore ? 2 : 1; /// Load : 1, Store : 2
          bank_status[selBank].used = 1; /// Load : 1, Store : 2
          bank_status[selBank].iob = iobIdx;
          bank_status[selBank].start = 0;
          bank_status[selBank].end = memSize;  

          dfg_io_infos[id] = ioInfo;
        }
      }
    }
  }


  /// Get io information of every block load ops, including Addr(Spad), iobAddr, LorS
  _moduleop.walk([&](ADORA::DataBlockLoadOp blockload) {  
    if(findElement(blockload.getKernelNameAsStrVector(), kernel.getKernelName()) != -1){
      //// this block load belongs to this kernels 
      std::string BlockLoadName = kernel.getKernelName() + ":" + blockload.getId().str();
      blockload.dump();
      if(_LoadToSPMInfos.count(blockload) != 0){
        //// TODO: what to do?
      }
      else {
        std::set<int> IONodes = dfg->ioNodes();
        for(auto& id : IONodes){
          DFGIONode* IoNode = dynamic_cast<DFGIONode*>(dfg->node(id));
          if(IoNode->memRefName() == BlockLoadName){
            int memSize = dynamic_cast<DFGIONode*>(dfg->node(id))->memSize();
            int spadDataByte = adg->cfgSpadDataWidth() / 8; // dual ports of cfg-spad have the same width 
            memSize = (memSize + spadDataByte - 1) / spadDataByte * spadDataByte; // align to spadDataByte
            auto& attr =  mapper->_mapping->dfgNodeAttr(id);
            int iobId = attr.adgNode->id();
            int iobIdx = dynamic_cast<IOBNode*>(adg->node(iobId))->index();
            iob_ens.setBitTo(iobIdx, 1);    // iob_ens |= 1 << iobIdx; 
            std::vector<int> banks = adg->iobToSpadBanks(iobIdx); // spad banks connected to this IOB
            int minBank = *(std::min_element(banks.begin(), banks.end()));
            std::vector<int> availBanks;
            for(int bank : banks){ // two IOs of the same DFG cannot access the same bank
              if(bank_status[bank].used == 0){
                availBanks.push_back(bank);
              }
            }        

            /// dfg io information 
            int selBank = availBanks[0];
            // int selStart = bankStatus[0].second;

            dfgIoInfo ioInfo;   
            ioInfo.isStore = dfg->getOutNodes().count(id);
            ioInfo.addr = selBank * sizeofBank; /*selBank * sizeofBank + selStart;*/
            ioInfo.iobAddr = ((selBank - minBank) * sizeofBank) / dataByte; /*((selBank - minBank) * sizeofBank + selStart) / dataByte*/;        

            bank_status[selBank].used = ioInfo.isStore ? 2 : 1; /// Load : 1, Store : 2
            bank_status[selBank].iob = iobIdx;
            bank_status[selBank].start = 0;
            bank_status[selBank].end = memSize;  

            dfg_io_infos[id] = ioInfo;

            _LoadToDfgIoInfos[blockload].push_back(ioInfo); 

            _LoadToSPMInfos[blockload].push_back(std::pair(selBank, ioInfo));
          }
        }
      }
    }
  });

  /// Get io information of every block store ops, including Addr(Spad), iobAddr, LorS
  _moduleop.walk([&](ADORA::DataBlockStoreOp blockstore) {  
    if(blockstore.getKernelName() == kernel.getKernelName()){
      std::string BlockStoreName = blockstore.getKernelName().str() + ":" + blockstore.getId().str();

      //// set allocation's info
      Operation* srcOfblockstore = GetTheSourceOperationOfBlockStore(blockstore);

      std::set<int> IONodes = dfg->getOutNodes();
      for(auto& id : IONodes){
        DFGIONode* IoNode = dynamic_cast<DFGIONode*>(dfg->node(id));
        if(IoNode->memRefName() == BlockStoreName){
          int memSize = dynamic_cast<DFGIONode*>(dfg->node(id))->memSize();
          int spadDataByte = adg->cfgSpadDataWidth() / 8; // dual ports of cfg-spad have the same width 
          memSize = (memSize + spadDataByte - 1) / spadDataByte * spadDataByte; // align to spadDataByte
          auto& attr =  mapper->_mapping->dfgNodeAttr(id);
          int iobId = attr.adgNode->id();
          int iobIdx = dynamic_cast<IOBNode*>(adg->node(iobId))->index();
          iob_ens.setBitTo(iobIdx, 1);    // iob_ens |= 1 << iobIdx; 
          std::vector<int> banks = adg->iobToSpadBanks(iobIdx); // spad banks connected to this IOB
          int minBank = *(std::min_element(banks.begin(), banks.end()));
          std::vector<int> availBanks;
          for(int bank : banks){ // two IOs of the same DFG cannot access the same bank
            if(bank_status[bank].used == 0){
              availBanks.push_back(bank);
            }
          }        

          /// dfg io information 
          int selBank = availBanks[0];
          // int selStart = bankStatus[0].second;

          dfgIoInfo ioInfo;   
          ioInfo.isStore = dfg->getOutNodes().count(id);
          ioInfo.addr = selBank * sizeofBank; /*selBank * sizeofBank + selStart;*/
          ioInfo.iobAddr = ((selBank - minBank) * sizeofBank) / dataByte; /*((selBank - minBank) * sizeofBank + selStart) / dataByte*/;        

          bank_status[selBank].used = ioInfo.isStore ? 2 : 1; /// Load : 1, Store : 2
          bank_status[selBank].iob = iobIdx;
          bank_status[selBank].start = 0;
          bank_status[selBank].end = memSize;  

          dfg_io_infos[id] = ioInfo;

          assert(_StoreToDfgIoInfo.count(blockstore) == 0 && "One output operation should only be count once.");
          _StoreToDfgIoInfo[blockstore] = ioInfo; 

          if(isa<ADORA::LocalMemAllocOp>(srcOfblockstore) && 
            _LocalAllocToSPMInfo.count(dyn_cast<ADORA::LocalMemAllocOp>(srcOfblockstore)) == 0){
            ADORA::LocalMemAllocOp localAlloc = dyn_cast<ADORA::LocalMemAllocOp>(srcOfblockstore);
            _LocalAllocToSPMInfo[localAlloc] = std::pair(selBank, ioInfo);
          }
        }
      }
    }
  });
  
  for(auto elem: _LoadToDfgIoInfos){
    llvm::errs() << elem.first << "\n";
    for(auto e : elem.second){
      llvm::errs() << e.addr << ", " << e.iobAddr << "\n";        
    } 
  }  
  // /// Get io information of every local allocation ops, including Addr(Spad), iobAddr, LorS
  // _moduleop.walk([&](ADORA::LocalMemAllocOp alloc) {  
  //   if(findElement(alloc.getKernelNameAsStrVector(), kernel.getKernelName()) != -1){
  //     std::string AllocName = kernel.getKernelName() + ":" + alloc.getId().str();

  //     if(_LocalAllocToSPMInfo.count(alloc) != 0){
  //       std::set<int> IONodes = dfg->ioNodes();
  //       for(auto& id : IONodes){
  //         DFGIONode* IoNode = dynamic_cast<DFGIONode*>(dfg->node(id));
  //         if(IoNode->memRefName() == AllocName){
  //           int memSize = dynamic_cast<DFGIONode*>(dfg->node(id))->memSize();
  //           int spadDataByte = adg->cfgSpadDataWidth() / 8; // dual ports of cfg-spad have the same width 
  //           memSize = (memSize + spadDataByte - 1) / spadDataByte * spadDataByte; // align to spadDataByte
  //           auto& attr =  mapper->_mapping->dfgNodeAttr(id);
  //           int iobId = attr.adgNode->id();
  //           int iobIdx = dynamic_cast<IOBNode*>(adg->node(iobId))->index();
  //           iob_ens |= 1 << iobIdx; 

  //           /// dfg io information 
  //           int selBank = getSPMInfosFromLocalAlloc(alloc).first;
  //           // int selStart = bankStatus[0].second;

  //           dfgIoInfo ioInfo = = getSPMInfosFromLocalAlloc(alloc).second;   

  //           bank_status[selBank].used = ioInfo.isStore ? 2 : 1; /// Load : 1, Store : 2
  //           bank_status[selBank].iob = iobIdx;
  //           bank_status[selBank].start = 0;
  //           bank_status[selBank].end = memSize;  

  //           dfg_io_infos[id] = ioInfo;

  //           assert(_StoreToDfgIoInfo.count(blockstore) == 0 && "One output operation should only be count once.");
  //           _StoreToDfgIoInfo[blockstore] = ioInfo; 
  //         }
  //       }
  //     }
  //   }
  // });


  _kernel_to_dfg_io_infos[kernel] = dfg_io_infos;
  _kernel_to_iob_ens[kernel] = iob_ens;
}
