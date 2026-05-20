//===----------------------------------------------------------------------===//
//
// Copyright 2023-2024 The ADORA Authors.
//
//===----------------------------------------------------------------------===//
#include "emit/Emit.h"
#include "emit/EmitCGRACall.h"
#include "emit/OpVisitor.h"
#include "ADORA/Dialect/ADORA/Utility/Utility.h"
#include "mlir/Dialect/Affine/Utils.h"

using namespace mlir;
using namespace mlir::ADORA;


//===----------------------------------------------------------------------===//
// Some tool functions
//===----------------------------------------------------------------------===//
/// @brief Get a new id for new op which will be emit to C 
/// @param value_name_list
static int NewValueNameId(const llvm::SmallDenseMap<mlir::Value, Op_Name_C>& value_name_list){
  int new_id = 0;
  for(auto& elem: value_name_list){
      new_id = new_id < elem.second.id ? elem.second.id : new_id;
  }
  return new_id + 1;
}

// void emitAffineLoad(AffineLoadOp op) {
//   indent();
//   emitValue(op.getResult());
//   os << " = ";
//   emitValue(op.getMemRef());
//   auto affineMap = op.getAffineMap();
//   AffineExprEmitter affineEmitter(state, affineMap.getNumDims(),
//                                   op.getMapOperands());
//   for (auto index : affineMap.getResults()) {
//     os << "[";
//     affineEmitter.emitAffineExpr(index);
//     os << "]";
//   }
//   os << ";";
//   emitInfoAndNewLine(op);
// }

namespace {
class CGRVOpEmitter : public MLIROpVisitorBase<CGRVOpEmitter, bool> {
public:
  CGRVOpEmitter(llvm::raw_ostream &os) : _os(os) {}
  CGRVOpEmitter(CGRACallEmitter& emitter, llvm::raw_ostream &os) : 
      _cgracallemitter(&emitter) ,_os(os) {setIndent(emitter.getIndent());}
  using MLIROpVisitorBase::visitOp;

  /// Build op→slot mapping from a block. Call once before emitBlock iteration.
  void buildOpSlots(mlir::Block &block) {
    int slot = 0;
    for (auto &op : block) {
      if (mlir::isa<ADORA::DataBlockLoadOp,
                    ADORA::DataBlockStoreOp,
                    ADORA::KernelOp>(&op))
        opSlot_[&op] = slot++;
    }
  }

  /// Compute the hardware dep_flag string for a load/execute op based on its
  /// asyncDependencies. Returns "0" for root ops (no deps), otherwise picks the
  /// appropriate LD_DEP_* macro by counting how many task slots back the nearest
  /// producer is.  Falls back to LD_DEP_ST_LAST_TASK when offset > 2.
  std::string computeDepFlag(mlir::Operation *op,
                              const std::string &noDepFlag = "0",
                              const std::string &fallback = "LD_DEP_ST_LAST_TASK") {
    auto deps = ADORA::getAsyncDeps(op);
    if (deps.empty())
      return noDepFlag;

    auto myIt = opSlot_.find(op);
    if (myIt == opSlot_.end()) return fallback;
    int mySlot = myIt->second;

    int maxProducerSlot = -1;
    for (mlir::Value tok : deps) {
      mlir::Operation *producer = tok.getDefiningOp();
      if (!producer) continue;
      auto it = opSlot_.find(producer);
      if (it != opSlot_.end())
        maxProducerSlot = std::max(maxProducerSlot, it->second);
    }
    if (maxProducerSlot < 0) return fallback;

    int offset = mySlot - maxProducerSlot;
    if (offset == 1) return "LD_DEP_ST_LAST_TASK";
    if (offset == 2) return "LD_DEP_ST_LAST_SEC_TASK";
    return fallback;  // offset > 2: conservative
  }

  // op* → task slot index (0-based, increments per Load/Store/KernelOp)
  llvm::DenseMap<mlir::Operation*, int> opSlot_;

  /// Tool functions for emitting
  raw_ostream& indent(){return _os.indent(_indent);}
  void setIndent(unsigned newindent){ _indent = newindent;}

  /// @brief emit a new op to C, add this one to op_name_list. 
  /// @param mlirop the corresponding mlir operation
  /// @param type the C type of this operation
  /// @return 
  std::string EmitNewValueAndGetName(mlir::Value v, const std::string type){
    if(_cgracallemitter->getValueNameList().count(v)){
      return _cgracallemitter->lookupName(v);
    }
    Op_Name_C newopinfo;
    newopinfo.id = NewValueNameId(_cgracallemitter->getValueNameList());
    newopinfo.type = type;
    _cgracallemitter->appendValueNameList(v, newopinfo);
    return newopinfo.name();
  }

  template <typename opT>
    bool EmitBinary(opT op ,std::string op_symbol){
      std::string type = getEmitType(op.getResult());
      std::string Lhs = _cgracallemitter->lookupName(op.getLhs());
      if(Lhs == "")
        Lhs = ConstOpToValueStr[op.getLhs()];
      std::string Rhs = _cgracallemitter->lookupName(op.getRhs());
      if(Rhs == "")
        Rhs = ConstOpToValueStr[op.getRhs()];
      assert(Lhs != "" && Rhs != "");

      indent() << type << " " << EmitNewValueAndGetName(op.getResult(), type) 
          << " = " << Lhs << " " << op_symbol << " " << Rhs << ";\n"; 
      return true;
    }

  /// ADORA dialect operations.
  bool visitOp(ADORA::DataBlockLoadOp op) {
    /// DataBlockLoadOp can be seen as a memref subview op, 
    /// for example: 
    /// %0 = ADORA.BlockLoad %arg0 [%arg3, 0, %arg5 * 2, %arg6 * 2] : memref<1x3x230x230xf32> -> memref<1x3x7x62xf32>  {Id = "0", KernelName = "forward_kernel_0"}
    /// 6 variables should be maintained:
    /// DMA_Len: length of one dma request.
    /// DRAM_BaseAddr: the base address in DRAM of the source memref.
    /// DRAM_Offset: the offset address related to affine map
    /// DMA_Request_Offset: keep changing. For this one, DMA_Request_Offset = 230 * i + 230 * 230 * j, i = [0, 7), j = [0, 3)
    /// SPAD_BaseAddr: the base address of Scratchpad memory of the destinated transfer
    /// SPAD_Offset: keep increasing by DMA_Len.
    indent() << "{\n";
    indent() << "/// " << op << "\n";

    std::string BLid = op.getId().str();
    ::llvm::ArrayRef<int64_t> SourceShape =  op.getOriginalMemrefType().getShape();
    ::llvm::ArrayRef<int64_t> ResultShape =  op.getResultType().getShape();
    // MemRefType SourceType = op.getOriginalMemref();
    // MemRefType ResultType = op.getResultType();
    if(SourceShape.size() == 0){
      //// memref<f32> -> memref<2xf32>
      std::string Memref_BaseAddr = _cgracallemitter->lookupName(op.getOriginalMemref());
      llvm::SmallVector<dfgIoInfo> DfgIoInfos = _cgracallemitter->getDfgIoInfosFromBlockLoad(op);
      assert(DfgIoInfos.size() == 1);
      uint64_t spadbaddr = DfgIoInfos[0].addr;
      uint64_t DMA_Len = 8; /// 64bit
      int fuse = 0;
      std::stringstream load_data;
      load_data <<"load_data(*" << Memref_BaseAddr  ////address not pointer
              <<", 0x" << std::hex << spadbaddr 
              <<", " << std::dec << DMA_Len 
              <<", " << std::dec << fuse  /*fuse*/
              <<", _task_id" /*Task id*/
              << ", " << computeDepFlag(op.getOperation()) /*Task dep*/
              <<");\n";
      indent() << load_data.str();
      _os << "\n";
      indent() << "}\n";
      return true;
    }

    assert(SourceShape.size() == ResultShape.size());

    /// Get DMA_Len 
    uint64_t DataBytes = op.getOriginalMemrefType().getElementTypeBitWidth()/8;
    uint64_t DMA_Len = DataBytes;

    for(int r = SourceShape.size() - 1; r >= 0; r--){
      // assert(SourceShape[r] >= ResultShape[r]);
      if(op.getOriginalMemrefType().isDynamicDim(r)){
        DMA_Len = DMA_Len * ResultShape[r];
        break;
      }
      else{
        assert(SourceShape[r] >= ResultShape[r]);
        DMA_Len = DMA_Len * ResultShape[r];
        if(SourceShape[r] > ResultShape[r])
          break;
      }
    }
    
    /// Get DRAM_BaseAddr
    std::string Memref_BaseAddr = _cgracallemitter->lookupName(op.getOriginalMemref());

    /// Get DRAM_Offset
    std::string DRAM_Offset = "";
    for(int operandIdx = 0; operandIdx < op.getMapOperands().size(); operandIdx++){
      mlir::Value operand = op.getMapOperands()[operandIdx];
      std::string operandname = _cgracallemitter->lookupName(operand);
      if(operandname == "")      
        operandname = ConstOpToValueStr[operand];
      assert(operandname != "");
      if(operandname == "0")
        continue;
      else{
        SmallVector<int>Dimensions = getOperandDimensionsInMap(/*dim=*/operandIdx, /*map=*/op.getAffineMap());
        // assert(Dimensions.size() == 1);

        for(unsigned d = 0; d < Dimensions.size(); d++){
          int64_t elements_each_step = DataBytes;
          for (unsigned i = Dimensions[d] + 1; i < SourceShape.size(); i++){
            elements_each_step *= SourceShape[i];
          }
          if(DRAM_Offset != "")
            DRAM_Offset = DRAM_Offset + " + ";
          DRAM_Offset = DRAM_Offset + std::to_string(elements_each_step) + " * " + operandname;
        }
      }
    }
    if(DRAM_Offset == "") {
      if(op.getAffineMap().isEmpty()){
        ///// For %2 = ADORA.BlockLoad %arg2 [] : memref<?xi32> -> memref<2xi32>
        DRAM_Offset = "0";
      }
      else{
        ///// For %2 = ADORA.BlockLoad %arg2 [11, 10] : memref<20x506xi32> -> memref<2x506xi32>
        for(int exprIdx = 0; exprIdx < op.getAffineMap().getResults().size(); exprIdx++){
          AffineExpr expr = op.getAffineMap().getResult(exprIdx);
          assert(expr.getKind() == AffineExprKind::Constant);
          std::string cstValue = std::to_string(expr.dyn_cast<AffineConstantExpr>().getValue());
          assert(cstValue != "");
          if(cstValue == "0")
            continue;
          else{
            // SmallVector<int>Dimensions = getOperandDimensionsInMap(/*dim=*/exprIdx, /*map=*/op.getAffineMap());
            int64_t elements_each_step = DataBytes;
            for (unsigned i = exprIdx + 1; i < SourceShape.size(); i++){
              elements_each_step *= SourceShape[i];
            }
            if(DRAM_Offset != "")
              DRAM_Offset = DRAM_Offset + " + ";
            DRAM_Offset = DRAM_Offset + std::to_string(elements_each_step) + " * " + cstValue;
          }
        }
      }
      if(DRAM_Offset == "") {
        DRAM_Offset = "0";
      }
    }

    /// Get DMA_Request_Offset
    std::vector<int64_t> DMA_Request_Offsets;
    bool continuous = true;
    for(int r = SourceShape.size() - 1; r >= 0; r--){
      if(!continuous){
        if(SourceShape[r] == 1)
          DMA_Request_Offsets.insert(DMA_Request_Offsets.begin(), -1); /// -1 means transfer of this rank is overlooked
        else 
          DMA_Request_Offsets.insert(DMA_Request_Offsets.begin(), ResultShape[r]);
      }
      else {/// continuous
        if(SourceShape[r] > ResultShape[r]){
          DMA_Request_Offsets.insert(DMA_Request_Offsets.begin(), -1); /// -1 means transfer of this rank is continuous
          continuous = false;
        }
        else if(SourceShape[r] == ResultShape[r]){
          DMA_Request_Offsets.insert(DMA_Request_Offsets.begin(), -1); /// -1 means transfer of this rank is continuous
        }
      }
    }    

    /// SPAD_BaseAddr
    /// SPAD_Offset: keep increasing by DMA_Len.
    llvm::SmallVector<dfgIoInfo> DfgIoInfos = _cgracallemitter->getDfgIoInfosFromBlockLoad(op);
    llvm::SmallVector<uint64_t> SPAD_BaseAddrs;
    for(dfgIoInfo elem: DfgIoInfos){
      SPAD_BaseAddrs.push_back(elem.addr);
    }

    /// Emit C
    unsigned cur_indent = _indent;
    unsigned idx_num = 0;
    // llvm::SmallDenseMap<int, int64_t> LItoStep;
    indent() << "uint64_t dramoffset_" << BLid << " = " << DRAM_Offset <<";\n";
    std::string DMA_Request_Offsets_str = "uint64_t roffset_" + BLid + " = ";
    indent() << "uint64_t spadoffset_" << BLid << " = 0;\n";
    for(int r = 0; r < DMA_Request_Offsets.size(); r++){
      int64_t Roffset = DMA_Request_Offsets[r];
      std::string idx = "idx_" + std::to_string(r);
      if(Roffset != -1){
        idx_num++;
        int64_t elements_each_step = DataBytes;
        for (unsigned i = r + 1; i < SourceShape.size(); i++){
          elements_each_step *= SourceShape[i];
        }
        indent() << "for(int " << idx << " = " << 0 << "; "
              << idx << " < " <<  Roffset << "; "
              << idx << "++){\n";
        // LItoStep[idx] = elements_each_step;
        DMA_Request_Offsets_str += " " + std::to_string(elements_each_step)
                                + "*" + idx + " +";
        _indent += 2;
        setIndent(_indent);
      }
    }
    if(DMA_Request_Offsets_str.substr(DMA_Request_Offsets_str.size()-2, 2) == "= ")
      DMA_Request_Offsets_str += "0;";
    else
      DMA_Request_Offsets_str.back() = ';';
    indent() << DMA_Request_Offsets_str << "\n";
    // _indent += 2;
    // setIndent(_indent);
    // for(auto spadbaddr : SPAD_BaseAddrs){
    for(int i = 0; i < SPAD_BaseAddrs.size(); i++)
    {
      auto spadbaddr = SPAD_BaseAddrs[i];
      int fuse = (i == SPAD_BaseAddrs.size() - 1)? 0 : 1; /// fuse: 0 - broadcast, 1 - non-broadcast
      std::stringstream load_data;
      load_data <<"load_data(" << Memref_BaseAddr 
              <<" + " << "dramoffset_" << BLid
              <<" + " << "roffset_" << BLid
              <<", 0x" << std::hex << spadbaddr 
              <<" + " << "spadoffset_" << BLid 
              <<", " << std::dec << DMA_Len 
              <<", " << std::dec << fuse  /*fuse*/
              <<", _task_id" /*Task id*/
              << ", " << computeDepFlag(op.getOperation()) /*Task dep*/
              <<");\n";
      indent() << load_data.str();

    }
    indent()<< "spadoffset_" << BLid 
            << " = spadoffset_" << BLid 
            << " + " << DMA_Len <<";\n";



    _indent = cur_indent;
    setIndent(_indent);
    indent();
    for(int _ = 0 ; _ < idx_num ; _++){
      _os << "} ";
    }
    _os << "\n";
    indent() << "}\n";

    return true;
  }

  bool visitOp(ADORA::DataBlockStoreOp op) {
    /// DataBlockStoreOp can be seen as an opposite operation of memref subview op, 
    /// 6 variables should be maintained:
    /// DMA_Len: length of one dma request.
    /// DRAM_BaseAddr: the base address in DRAM of the source memref.
    /// DRAM_Offset: the offset address related to affine map
    /// DMA_Request_Offset: keep changing. For this one, DMA_Request_Offset = 230 * i + 230 * 230 * j, i = [0, 7), j = [0, 3)
    /// SPAD_BaseAddr: the base address of Scratchpad memory of the destinated transfer
    /// SPAD_Offset: keep increasing by DMA_Len.
    indent() << "{\n";
    indent() << "/// " << op << "\n";

    std::string BLid = op.getId().str();
    ::llvm::ArrayRef<int64_t> SourceShape =  op.getSourceMemrefType().getShape();
    ::llvm::ArrayRef<int64_t> TargetShape =  op.getTargetMemrefType().getShape();
    // MemRefType SourceType = op.getOriginalMemref();
    // MemRefType ResultType = op.getResultType();
    if(TargetShape.size() == 0){
      //// memref<2xf32> -> memref<f32> 
      std::string Memref_BaseAddr = _cgracallemitter->lookupName(op.getTargetMemref());
      // llvm::SmallVector<dfgIoInfo> DfgIoInfos = _cgracallemitter->getDfgIoInfosFromBlockLoad(op);
      dfgIoInfo DfgIoInfo = _cgracallemitter->getDfgIoInfosFromBlockStore(op);
      // assert(DfgIoInfos.size() == 1);
      uint64_t spadbaddr = DfgIoInfo.addr;
      uint64_t DMA_Len = 8; /// 64bit
      int fuse = 0;

      std::stringstream store_data;
      store_data <<"store(&" << Memref_BaseAddr  //// address not pointer
              <<", 0x" << std::hex << spadbaddr 
              <<", " << std::dec << DMA_Len 
              <<", _task_id" /*Task id*/ << ", 0" /*Task dep*/
              <<");\n";
      indent() << store_data.str();

      _os << "\n";
      indent() << "}\n";

      if(IsLastBlockStoreOp(op)){
        indent() << "_task_id++;\n";
      }

      return true;
    }

    assert(SourceShape.size() == TargetShape.size());

    /// Get DMA_Len 
    uint64_t DataBytes = op.getSourceMemrefType().getElementTypeBitWidth()/8;
    uint64_t DMA_Len = DataBytes;

    for(int r = SourceShape.size() - 1; r >= 0; r--){
      if(op.getTargetMemrefType().isDynamicDim(r)){
        DMA_Len = DMA_Len * SourceShape[r];
        break;
      }
      else{
        assert(SourceShape[r] <= TargetShape[r]);
        DMA_Len = DMA_Len * SourceShape[r];
        if(SourceShape[r] < TargetShape[r])
          break;
      }
    }
    
    /// Get DRAM_BaseAddr
    std::string Memref_BaseAddr = _cgracallemitter->lookupName(op.getTargetMemref());

    /// Get DRAM_Offset
    std::string DRAM_Offset = "";
    for(int operandIdx = 0; operandIdx < op.getMapOperands().size(); operandIdx++){
      mlir::Value operand = op.getMapOperands()[operandIdx];
      operand.dump();
      std::string operandname = _cgracallemitter->lookupName(operand);
      std::cout << operandname <<"\n";
      if(operandname == "")      
        operandname = ConstOpToValueStr[operand];
      assert(operandname != "");
      if(operandname == "0")
        continue;
      else{
        SmallVector<int>Dimensions = getOperandDimensionsInMap(/*dim=*/operandIdx, /*map=*/op.getAffineMap());
        // assert(Dimensions.size() == 1);

        for(unsigned d = 0; d < Dimensions.size(); d++){
          int64_t elements_each_step = DataBytes;
          for (unsigned i = Dimensions[d] + 1; i < TargetShape.size(); i++){
            elements_each_step *= TargetShape[i];
          }
          if(DRAM_Offset != "")
            DRAM_Offset = DRAM_Offset + " + ";
          DRAM_Offset = DRAM_Offset + std::to_string(elements_each_step) + " * " + operandname;
        }
      }
    }
    if(DRAM_Offset == "") {
      if(op.getAffineMap().isEmpty()){
        ///// ADORA.Blockstore %1, %arg2 [] :  memref<2xi32> -> memref<?xi32> 
        DRAM_Offset = "0";
      }
      else{
        ///// ADORA.Blockstore %1, %arg2 [11, 10] : memref<2x506xi32> -> memref<20x506xi32>
        for(int exprIdx = 0; exprIdx < op.getAffineMap().getResults().size(); exprIdx++){
          AffineExpr expr = op.getAffineMap().getResult(exprIdx);
          assert(expr.getKind() == AffineExprKind::Constant);
          std::string cstValue = std::to_string(expr.dyn_cast<AffineConstantExpr>().getValue());
          assert(cstValue != "");
          if(cstValue == "0")
            continue;
          else{
            // SmallVector<int>Dimensions = getOperandDimensionsInMap(/*dim=*/exprIdx, /*map=*/op.getAffineMap());
            int64_t elements_each_step = DataBytes;
            for (unsigned i = exprIdx + 1; i < TargetShape.size(); i++){
              elements_each_step *= TargetShape[i];
            }
            if(DRAM_Offset != "")
              DRAM_Offset = DRAM_Offset + " + ";
            DRAM_Offset = DRAM_Offset + std::to_string(elements_each_step) + " * " + cstValue;
          }
        }
      }
      if(DRAM_Offset == "") {
        DRAM_Offset = "0";
      }
    }

    /// Get DMA_Request_Offset
    std::vector<int64_t> DMA_Request_Offsets;
    bool continuous = true;
    for(int r = SourceShape.size() - 1; r >= 0; r--){
      if(!continuous){
        if(SourceShape[r] == 1)
          DMA_Request_Offsets.insert(DMA_Request_Offsets.begin(), -1); /// -1 means transfer of this rank is overlooked
        else 
          DMA_Request_Offsets.insert(DMA_Request_Offsets.begin(), SourceShape[r]);
      }
      else {/// continuous
        if(TargetShape[r] > SourceShape[r]){
          DMA_Request_Offsets.insert(DMA_Request_Offsets.begin(), -1); /// -1 means transfer of this rank is continuous
          continuous = false;
        }
        else if(TargetShape[r] == SourceShape[r]){
          DMA_Request_Offsets.insert(DMA_Request_Offsets.begin(), -1); /// -1 means transfer of this rank is continuous
        }
      }
    }    

    /// SPAD_BaseAddr
    /// SPAD_Offset: keep increasing by DMA_Len.
    dfgIoInfo DfgIoInfos = _cgracallemitter->getDfgIoInfosFromBlockStore(op);
    llvm::SmallVector<uint64_t> SPAD_BaseAddrs;
    SPAD_BaseAddrs.push_back(DfgIoInfos.addr);

    /// Emit C
    unsigned cur_indent = _indent;
    unsigned idx_num = 0;
    // llvm::SmallDenseMap<int, int64_t> LItoStep;
    indent() << "uint64_t dramoffset_" << BLid << " = " << DRAM_Offset <<";\n";
    std::string DMA_Request_Offsets_str = "uint64_t roffset_" + BLid + " = ";
    indent() << "uint64_t spadoffset_" << BLid << " = 0;\n";
    for(int r = 0; r < DMA_Request_Offsets.size(); r++){
      int64_t Roffset = DMA_Request_Offsets[r];
      std::string idx = "idx_" + std::to_string(r);
      if(Roffset != -1){
        idx_num++;
        int64_t elements_each_step = DataBytes;
        for (unsigned i = r + 1; i < TargetShape.size(); i++){
          elements_each_step *= TargetShape[i];
        }
        indent() << "for(int " << idx << " = " << 0 << "; "
              << idx << " < " <<  Roffset << "; "
              << idx << "++){\n";
        // LItoStep[idx] = elements_each_step;
        DMA_Request_Offsets_str += " " + std::to_string(elements_each_step)
                                + "*" + idx + " +";
        _indent += 2;
        setIndent(_indent);
      }
    }
    if(DMA_Request_Offsets_str.substr(DMA_Request_Offsets_str.size()-2, 2) == "= ")
      DMA_Request_Offsets_str += "0;";
    else
      DMA_Request_Offsets_str.back() = ';';
    indent() << DMA_Request_Offsets_str << "\n";
    // _indent += 2;
    // setIndent(_indent);
    // for(auto spadbaddr : SPAD_BaseAddrs){
    for(int i = 0; i < SPAD_BaseAddrs.size(); i++)
    {
      auto spadbaddr = SPAD_BaseAddrs[i];
      int fuse = (i == SPAD_BaseAddrs.size() - 1)? 0 : 1; /// fuse: 0 - broadcast, 1 - non-broadcast
      std::stringstream store_data;
      store_data <<"store(" << Memref_BaseAddr 
              <<" + " << "dramoffset_" << BLid
              <<" + " << "roffset_" << BLid
              <<", 0x" << std::hex << spadbaddr 
              <<" + " << "spadoffset_" << BLid 
              <<", " << std::dec << DMA_Len 
              <<", _task_id" /*Task id*/ << ", 0" /*Task dep*/
              <<");\n";
      indent() << store_data.str();

    }
    indent()<< "spadoffset_" << BLid 
            << " = spadoffset_" << BLid 
            << " + " << DMA_Len <<";\n";


    _indent = cur_indent;
    setIndent(_indent);
    indent();
    for(int _ = 0 ; _ < idx_num ; _++){
      _os << "} ";
    }
    _os << "\n";
    indent() << "}\n";

    if(IsLastBlockStoreOp(op)){
      indent() << "_task_id++;\n";
    }

    return true;    
  }

  bool visitOp(ADORA::LocalMemAllocOp op) {
    
    return true;
  }
  


  bool visitOp(ADORA::KernelOp op) {
    indent() << "{\n";
    if(!MapHasKey(_cgracallemitter->KnToCfgExe, op)){
      // Configuration cfg = ;
      ADG* adg = _cgracallemitter->getADG();
      _cgracallemitter->GenerateCGRACFGAndEXE(op, _cgracallemitter->KnToConfiguration[op], adg);
    }
    if(!op.getKernelName().empty()){
      indent() << "/// " << op.getKernelName() << "\n";
    }

    std::vector<std::string> strs = split_str_by_char(_cgracallemitter->KnToCfgExe[op], '\n');
    for(std::string str: strs){
      indent() << str << "\n";
    }
    
    indent() << "}\n";
    return true;
  }
  // bool visitOp(BufferOp op) {
  //   if (op.getDepth() == 1)
  //     return emitter.emitAlloc(op), true;
  //   return op.emitOpError("only support depth of 1"), false;
  // }
  // bool visitOp(ConstBufferOp op) { return emitter.emitConstBuffer(op), true; }
  // bool visitOp(StreamOp op) { return emitter.emitStreamChannel(op), true; }
  // bool visitOp(StreamReadOp op) { return emitter.emitStreamRead(op), true; }
  // bool visitOp(StreamWriteOp op) { return emitter.emitStreamWrite(op), true; }
  // bool visitOp(AxiBundleOp op) { return true; }
  // bool visitOp(AxiPortOp op) { return emitter.emitAxiPort(op), true; }
  // bool visitOp(AxiPackOp op) { return false; }
  // bool visitOp(PrimMulOp op) { return emitter.emitPrimMul(op), true; }
  // bool visitOp(PrimCastOp op) { return emitter.emitAssign(op), true; }
  // bool visitOp(hls::AffineSelectOp op) {
  //   return emitter.emitAffineSelect(op), true;
  // }

  /// Function operations.
  // bool visitOp(func::CallOp op) { return emitter.emitCall(op), true; }
  bool visitOp(memref::AllocaOp op) { 
    mlir::MemRefType mt = op.getType();
    mlir::Type t = mt.getElementType();
    std::string type = getEmitType(t);
    std::string name = EmitNewValueAndGetName(op.getResult(), type);

    auto shape = mt.getShape();
    if (shape.size() == 0) {
      // scalar alloca: e.g. int x;
      indent() << type << " " << name << ";\n";
    } else {
      // array alloca: e.g. int score[8][8];
      indent() << type << " " << name;
      for (auto dim : shape)
        _os << "[" << dim << "]";
      _os << ";\n";
    }
    return true; 
  }
  bool visitOp(func::ReturnOp op) { return true; }

  /// SCF statements.
  // bool visitOp(scf::ForOp op) { return emitter.emitScfFor(op), true; };
  // bool visitOp(scf::IfOp op) { return emitter.emitScfIf(op), true; };
  // bool visitOp(scf::ParallelOp op) { return false; };
  // bool visitOp(scf::ReduceOp op) { return false; };
  // bool visitOp(scf::ReduceReturnOp op) { return false; };
  // bool visitOp(scf::YieldOp op) { return emitter.emitScfYield(op), true; };

  /// CF 
  bool visitOp(cf::BranchOp op) { 
    _cgracallemitter->emitBlock(*(op.getDest()), _os);
    return true;
  }

  /// Affine statements.
  bool visitOp(affine::AffineForOp op) { 
    indent() << "for (";
    auto iterVar = op.getInductionVar();

    // Emit lower bound.
    assert(op.getLowerBoundMap().getResults().size()==1);
    _os << "int " << EmitNewValueAndGetName(iterVar, "int") << " = ";
    _os << op.getLowerBoundMap().getResult(0) << "; ";

    // Emit loop invariant(upper bound)
    assert(op.getUpperBoundMap().getResults().size()==1);
    _os << _cgracallemitter->lookupName(iterVar) << " < " ;
    _os << op.getUpperBoundMap().getResult(0) << "; ";

    // Emit loop step
    _os << _cgracallemitter->lookupName(iterVar) << " = " ;
    _os << _cgracallemitter->lookupName(iterVar) << " + "  << op.getStep().getSExtValue() << "){\n";

    _cgracallemitter->emitBlock(*(op.getBody()), _os);
    // reduce
    indent() << "}\n";
    indent() << "\n";
    indent() << "\n";
    indent() << "\n";
    return true;
  }

  /// SCF statements.
  bool visitOp(scf::ForOp op) { 
    indent() << "for (";
    auto iterVar = op.getInductionVar();

    // Emit lower bound.
    std::string lbStr = _cgracallemitter->lookupName(op.getLowerBound());
    if(lbStr.empty()) {
      // try to read as constant op
      auto cstOp = op.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
      if(cstOp) {
        lbStr = std::to_string(cstOp.value());
      } else {
        llvm::errs() << "Warning: could not find lb value in scf.for\n";
        lbStr = "0";
      }
    }
    _os << "int " << EmitNewValueAndGetName(iterVar, "int") << " = " << lbStr << "; ";

    // Emit upper bound.
    std::string ubStr = _cgracallemitter->lookupName(op.getUpperBound());
    if(ubStr.empty()) {
      auto cstOp = op.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
      if(cstOp) {
        ubStr = std::to_string(cstOp.value());
      } else {
        llvm::errs() << "Warning: could not find ub value in scf.for\n";
        ubStr = "0";
      }
    }
    _os << _cgracallemitter->lookupName(iterVar) << " < " << ubStr << "; ";

    // Emit step.
    std::string stepStr = _cgracallemitter->lookupName(op.getStep());
    if(stepStr.empty()) {
      auto cstOp = op.getStep().getDefiningOp<arith::ConstantIndexOp>();
      if(cstOp) {
        stepStr = std::to_string(cstOp.value());
      } else {
        llvm::errs() << "Warning: could not find step value in scf.for\n";
        stepStr = "1";
      }
    }
    _os << _cgracallemitter->lookupName(iterVar) << " = " << _cgracallemitter->lookupName(iterVar) << " + " << stepStr << "){\n";

    _cgracallemitter->emitBlock(*(op.getBody()), _os);
    // reduce
    indent() << "}\n";
    indent() << "\n";
    indent() << "\n";
    indent() << "\n";
    return true;
  }

  bool visitOp(scf::YieldOp op) { 
    return true; 
  }
  // bool visitOp(AffineIfOp op) { return emitter.emitAffineIf(op), true; }
  // bool visitOp(AffineParallelOp op) {
  //   return emitter.emitAffineParallel(op), true;
  // }
  // bool visitOp(AffineApplyOp op) { return emitter.emitAffineApply(op), true; }
  // bool visitOp(AffineMaxOp op) {
  //   return emitter.emitAffineMaxMin(op, "max"), true;
  // }
  // bool visitOp(AffineMinOp op) {
  //   return emitter.emitAffineMaxMin(op, "min"), true;
  // }
  // bool visitOp(AffineLoadOp op) { 
    // return 
    // return emitter.emitAffineLoad(op), true; 
  // }
  bool visitOp(::mlir::affine::AffineStoreOp op) { 
    // std::string type = getEmitType(op.getResult());
    std::string value;
    if(isa<LLVM::UndefOp>(op.getValue().getDefiningOp())){
      value = "0";
    }
    else{
      value = _cgracallemitter->lookupName(op.getValue());
      if (value == "")
        value = ConstOpToValueStr[op.getValue()];
    }

    std::string memref = _cgracallemitter->lookupName(op.getMemref());
    if (op.getMemref().getType().cast<MemRefType>().getShape().size() == 0) {
      indent() << memref << " = " << value << ";\n";
    } else {
      std::stringstream ss;
      for (auto idx : op.getIndices())
        ss << "[" << _cgracallemitter->lookupName(idx) << "]";
      indent() << memref << ss.str() << " = " << value << ";\n";
    }
    return true;
    // return emitter.emitAffineStore(op), true; 
  }
  // bool visitOp(AffineVectorLoadOp op) { return false; }
  // bool visitOp(AffineVectorStoreOp op) { return false; }
  bool visitOp(affine::AffineYieldOp op) { return true; }

  /// Vector statements.
  // bool visitOp(vector::TransferReadOp op) {
  //   return emitter.emitTransferRead(op), true;
  // };
  // bool visitOp(vector::TransferWriteOp op) {
  //   return emitter.emitTransferWrite(op), true;
  // };
  // bool visitOp(vector::BroadcastOp op) {
  //   return emitter.emitBroadcast(op), true;
  // };

  // /// Memref statements.
  // bool visitOp(memref::AllocOp op) { return emitter.emitAlloc(op), true; }
  // bool visitOp(memref::AllocaOp op) { return emitter.emitAlloc(op), true; }
  // bool visitOp(memref::LoadOp op) { return emitter.emitLoad(op), true; }
  // bool visitOp(memref::StoreOp op) { return emitter.emitStore(op), true; }
  // bool visitOp(memref::DeallocOp op) { return true; }
  // bool visitOp(memref::CopyOp op) { return emitter.emitMemCpy(op), true; }
  // bool visitOp(memref::ReshapeOp op) { return emitter.emitReshape(op), true;
  // } bool visitOp(memref::CollapseShapeOp op) {
  //   return emitter.emitReshape(op), true;
  // }
  // bool visitOp(memref::ExpandShapeOp op) {
  //   return emitter.emitReshape(op), true;
  // }
  // bool visitOp(memref::ReinterpretCastOp op) {
  //   return emitter.emitReshape(op), true;
  // }

  /// Arithmetic dialect
  bool visitOp(arith::ConstantOp op) {
    // This indicates the constant type is scalar (float, integer, or bool).
    // if (isDeclared(op.getResult()))
    //   return;
    // arith::ConstantOp constin = dyn_cast<arith::ConstantOp>(in);
    // indent();
    mlir::Attribute constattr = op.getOperation()->getAttr(op.getValueAttrName());

    if(isa<FloatAttr>(constattr)){
      FloatAttr floatattr = dyn_cast<FloatAttr>(constattr);
      if(floatattr.getType().isF64()){
        double value = floatattr.getValueAsDouble();
        ConstOpToValueStr[op.getResult()] = std::to_string(value);  
        // std::string name_c = EmitNewValueAndGetName(op.getResult(), "double");
        // _os << "double " << name_c << " = " << std::to_string(value) << ";\n";
      } 
      else if(floatattr.getType().isF32()){
        double value = floatattr.getValueAsDouble();
        ConstOpToValueStr[op.getResult()] = std::to_string(value);  
        // std::string name_c = EmitNewValueAndGetName(op.getResult(), "float");
        // _os << "float " << name_c << " = " << std::to_string(value) << ";\n";
      }
    } 
    else if(isa<IntegerAttr>(constattr))
    {
      IntegerAttr intattr = dyn_cast<IntegerAttr>(constattr);
      if(intattr.getType().isInteger(16)){    
        int value = intattr.getInt();  
        ConstOpToValueStr[op.getResult()] = std::to_string(value);   
        // std::string name_c = EmitNewValueAndGetName(op.getResult(), "int16_t");
        // _os << "int16_t " << name_c << " = " << std::to_string(value) << ";\n";
      }
      else if(intattr.getType().isInteger(32)){
        int value = intattr.getInt();   
        ConstOpToValueStr[op.getResult()] = std::to_string(value);  
        // std::string name_c = EmitNewValueAndGetName(op.getResult(), "int32_t");        
        // _os << "int32_t " << name_c << " = " << std::to_string(value) << ";\n";
      }
      else if(intattr.getType().isInteger(64)){ 
        int value = intattr.getInt();   
        ConstOpToValueStr[op.getResult()] = std::to_string(value);  
        // std::string name_c = EmitNewValueAndGetName(op.getResult(), "int64_t");
        // _os << "int64_t " << name_c << " = " << std::to_string(value) << ";\n";
      }
      else if(intattr.getType().isUnsignedInteger(16)){    
        int value = intattr.getInt();   
        ConstOpToValueStr[op.getResult()] = std::to_string(value);  
        // std::string name_c = EmitNewValueAndGetName(op.getResult(), "uint16_t");
        // _os << "uint16_t " << name_c << " = " << std::to_string(value) << ";\n";
      }
      else if(intattr.getType().isUnsignedInteger(32)){
        int value = intattr.getInt();   
        ConstOpToValueStr[op.getResult()] = std::to_string(value);  
        // std::string name_c = EmitNewValueAndGetName(op.getResult(), "uint32_t");
        // _os << "uint32_t " << name_c << " = " << std::to_string(value) << ";\n";
      }
      else if(intattr.getType().isUnsignedInteger(64)){ 
        int value = intattr.getInt(); 
        ConstOpToValueStr[op.getResult()] = std::to_string(value);  
        // std::string name_c = EmitNewValueAndGetName(op.getResult(), "uint64_t");
        // _os << "uint64_t " << name_c << " = " << std::to_string(value) << ";\n";
      }
      else if(intattr.getType().isIndex()){ 
        int value = intattr.getInt();   
        ConstOpToValueStr[op.getResult()] = std::to_string(value);
        // std::string name_c = EmitNewValueAndGetName(op.getResult(), "int");
        // _os << "int " << name_c << " = " << std::to_string(value) << ";\n";
      }
    }
    else if(isa<BoolAttr>(constattr))
    {

      BoolAttr boolattr = dyn_cast<BoolAttr>(constattr);
      bool value = boolattr.getValue(); 
      ConstOpToValueStr[op.getResult()] = std::to_string(value);
      // std::string name_c = EmitNewValueAndGetName(op.getResult(), "bool");
      // _os << "bool " << name_c << " = " << std::to_string(value) << ";\n";
    }
    else if (auto denseAttr = op.getValue().dyn_cast<DenseElementsAttr>()) {
      // indent();
      denseAttr.dump();
      op.emitError("has unsupported constant denseAttr type.");
      abort();
      // emitArrayDecl(op.getResult());
      // os << " = {";
      // auto type =
      //   op.getResult().getType().template cast<ShapedType>().getElementType();

      // unsigned elementIdx = 0;
      // for (auto element : denseAttr.template getValues<Attribute>()) {
      //   auto string = getConstantString(type, element);
      //   if (string.empty())
      //     op.emitOpError("constant has invalid value");
      //   os << string;
      //   if (elementIdx++ != denseAttr.getNumElements() - 1)
      //     os << ", ";
      // }
      // os << "};";
      // emitInfoAndNewLine(op);
    } else
      op.emitError("has unsupported constant type.");
    
    // ConstOpToValueStr_print();
    return true;
  }

  bool visitOp(arith::AddIOp op) {
      std::string type = getEmitType(op.getResult());
      std::string Lhs = _cgracallemitter->lookupName(op.getLhs());
      if(Lhs == "")
        Lhs = ConstOpToValueStr[op.getLhs()];
      std::string Rhs = _cgracallemitter->lookupName(op.getRhs());
      if(Rhs == "")
        Rhs = ConstOpToValueStr[op.getRhs()];
      assert(Lhs != "" && Rhs != "");

      if( Lhs == "0" && Rhs == "0")
        indent() << type << " " << EmitNewValueAndGetName(op.getResult(), type) 
          << " = " << "0" << ";\n";
      else if( Lhs == "0" && Rhs != "0")
        indent() << type << " " << EmitNewValueAndGetName(op.getResult(), type) 
          << " = " << Rhs << ";\n";
      else if( Lhs != "0" && Rhs == "0")
        indent() << type << " " << EmitNewValueAndGetName(op.getResult(), type) 
          << " = " << Lhs << ";\n";
      else
        indent() << type << " " << EmitNewValueAndGetName(op.getResult(), type) 
          << " = " << Lhs << " " << "+" << " " << Rhs << ";\n"; 
      
      return true;
    // return EmitBinary(op, "+");
  }

  bool visitOp(arith::SubIOp op) {
      std::string type = getEmitType(op.getResult());
      std::string Lhs = _cgracallemitter->lookupName(op.getLhs());
      if(Lhs == "")
        Lhs = ConstOpToValueStr[op.getLhs()];
      std::string Rhs = _cgracallemitter->lookupName(op.getRhs());
      if(Rhs == "")
        Rhs = ConstOpToValueStr[op.getRhs()];
      assert(Lhs != "" && Rhs != "");

      if( Lhs == "0" && Rhs == "0")
        indent() << type << " " << EmitNewValueAndGetName(op.getResult(), type) 
          << " = " << "0" << ";\n";
      else if( Lhs != "0" && Rhs == "0")
        indent() << type << " " << EmitNewValueAndGetName(op.getResult(), type) 
          << " = " << Lhs << ";\n";
      else
        indent() << type << " " << EmitNewValueAndGetName(op.getResult(), type) 
          << " = " << Lhs << " " << "-" << " " << Rhs << ";\n"; 
      
      return true;

    // return EmitBinary(op, "-");
  }

  bool visitOp(arith::MulIOp op) {
    return EmitBinary(op, "*");
  }

  bool visitOp(arith::DivSIOp op) {
    return EmitBinary(op, "/");
  }

  bool visitOp(arith::ShRSIOp op) {
    return EmitBinary(op, ">>");
  }

  bool visitOp(arith::ShLIOp op) {
    return EmitBinary(op, "<<");
  }

  bool visitOp(LLVM::UndefOp op) {
    // op.getRes()
    // return EmitBinary(op, "/");
    return true; /// skip 
  }

  void ConstOpToValueStr_print() {
    llvm::errs()<< "=======Print ConstOpToValueStr=======\n";
    for(auto elem : ConstOpToValueStr){
      llvm::errs() << elem.first << "  --->  " << elem.second << "\n";
    }
    llvm::errs()<< "=======End Print=======\n";
  }

private:
  CGRACallEmitter* _cgracallemitter;
  llvm::DenseMap<mlir::Value, std::string> ConstOpToValueStr;
  llvm::raw_ostream &_os;
  unsigned _indent = 0;
};
CGRVOpEmitter* opEmitter;
} // namespace


// static InFlightDiagnostic emitError(Operation *op, const Twine &message) {
//   // state.encounteredError = true;
//   return op->emitError(message);
// }
//===----------------------------------------------------------------------===//
// Members of CGRACallEmitter class
//===----------------------------------------------------------------------===//
/// @brief Emit the header of a function including function name, function args...
/// @param os
void CGRACallEmitter::emitFunctionHead(func::FuncOp &funcop, llvm::raw_ostream &os) {
  std::stringstream ostr;
  ostr << "void " << funcop.getSymName().str() << "(";
  // Funtion args
  ArrayRef<mlir::Type> argTypes = funcop.getArgumentTypes();
  for(int argIdx = 0; argIdx < argTypes.size(); argIdx++){
    if(argIdx != 0){
      ostr << ", ";
    }
    mlir::Type argType = argTypes[argIdx];

    // argType.dump();
    Op_Name_C arg_info("arg", argIdx);
    appendValueNameList(funcop.getBody().getArgument(argIdx), arg_info);
    if(argType.isa<MemRefType>()){
      ostr << "void* " << "arg_" << argIdx;
    } 
    else if(argType.isIndex()){
      ostr << "int " << "arg_" << argIdx;
    }
    else if(argType.isInteger(32)){
      ostr << "int32_t " << "arg_" << argIdx;
    }
    else if(argType.isInteger(64)){
      ostr << "int64_t " << "arg_" << argIdx;
    }
    else if(argType.isUnsignedInteger(32)){
      ostr << "uint32_t " << "arg_" << argIdx;
    }
    else if(argType.isUnsignedInteger(64)){
      ostr << "uint64_t " << "arg_" << argIdx;
    }
    
  }
  ostr << "){\n";


  os << ostr.str();
}

/// @brief Emit a block (maybe a loop body, maybe a function body), especially the for loop structure
/// @param os
void CGRACallEmitter::emitBlock(mlir::Block &block, llvm::raw_ostream &os) {
  // std::stringstream ostr;
  addIndent();

  // Build op→slot mapping so visitOp can compute precise dep_flags.
  opEmitter->buildOpSlots(block);

  opEmitter->setIndent(getIndent());
  block.dump();

  for (auto &op : block) {
    op.dump();
    // TypeSwitch<Operation *, bool>(&op)
    //   .template Case<
    //     // Affine statements.
    //     affine::AffineForOp,
    //     // // Special expressions.
    //     arith::ConstantOp
    //   >([&](auto opNode) -> bool {
    //     op.dump();
    //     return true;
    //   })
    //   .Default([&](auto opNode) -> bool {
    //     llvm::errs() << "No support!\n";
    //     return false;
    //   });
    if(opEmitter->dispatchVisitor(&op)){
      continue;
    }
    else{
      op.emitError("can't be correctly emitted.");
    } 
  }
  reduceIndent();
  opEmitter->setIndent(getIndent());
  // os << ostr.str();
}


/// @brief Emit the whole module op to CGRA Call function in C languange
/// @param os
/// @return Successful or not
bool CGRACallEmitter::emitCGRACallFunction(llvm::raw_ostream &os) {
  opEmitter = new CGRVOpEmitter(*this, os);
  // ADORAEmitterState state(os);
  // ModuleEmitter(state).emitModule(module);
  // return failure(state.encounteredError);
  os << R"XXX(
//===----------------------------------------------------------------------===//
//
// Automatically generated file for CGRA call function in ADORA.
//
//===----------------------------------------------------------------------===//

#include "include/ISA.h"

static uint8_t _task_id = 0;

#define LD_DEP_ST_LAST_TASK 1     // this load command depends on the store command of last task
#define LD_DEP_EX_LAST_TASK 2     // this load command depends on the execute command of last task
#define LD_DEP_ST_LAST_SEC_TASK 3 // this load command depends on the store command of last second task
#define EX_DEP_ST_LAST_TASK 1     // this execute command depends on the store command of last task


)XXX";

  //// emit configuration data array
  os << R"XXX(
//===----------------------------------------------------------------------===//
// Configuration Data 
//===----------------------------------------------------------------------===//
)XXX";  
  for(auto elem : KnToCfgData){
    os << elem.second << "\n";
  }

  // _moduleop.walk([&](mlir::Operation* op) {
  //   op->dump();
  // });
  /// Emit module
  for(auto funcop : _moduleop.getOps<func::FuncOp>()){
    funcop.dump();
    /// function head
    emitFunctionHead(funcop, os);

    /// function body
    // addIndent();
    emitBlock(funcop.getBody().front(), os);

    // / function tail
    os << "  fence(1);\n";
    os << "}\n";
  }

  delete opEmitter;
}

// Traverse the whole module to find which operation keeps a "VAR_CONFIG" attr
// equal to arg config, and return the result value of the operation
std::string CGRACallEmitter::lookupVarConfigName(const std::string config){ 
  mlir::Value target_value;
  _moduleop.walk([&](mlir::Operation* op)-> WalkResult {
    op->dump();
    if(op->hasAttr("VAR_CONFIG") 
        && config == dyn_cast<StringAttr>(op->getAttr("VAR_CONFIG")).str()){
      /// find the value crresponding to the config
      assert(op->getResults().size() == 1);
      target_value = op->getResult(0);
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  std::string target_name = lookupName(target_value);
  assert(target_name != "");
  return target_name;
}

/// @brief Get the config data
std::string CGRACallEmitter::GenerateCGRAConfig(
  ADORA::KernelOp& kernel, Configuration cfg, ADG* adg){
  // adg->print();
  std::string CFGarrayName = "cin_" + kernel.getKernelName();
  std::stringstream CFGdata;
  CFGdata << "/// " << kernel.getKernelName() << "\n";

   // cfg.dumpCfgData(std::cout);
  std::map<int, dfgIoInfo> dfg_io_infos = std::move(_kernel_to_dfg_io_infos[kernel]);
  for(auto &elem : dfg_io_infos){
    cfg.setDfgIoSpadAddr(elem.first, elem.second.iobAddr);
  }
  std::vector<CfgDataPacket> cfgData;
  cfg.getCfgData(cfgData);
  // cfg.getCfgData(cfgData); /// debug
  int cfgSpadDataByte = adg->cfgSpadDataWidth() / 8;
  int cfgAddrWidth = adg->cfgAddrWidth();
  int cfgDataWidth = adg->cfgDataWidth();
  int alignWidth = (cfgAddrWidth > 16) ? 32 : 16;
  assert(alignWidth >= cfgAddrWidth && cfgDataWidth >= alignWidth);
  int cfgNum = 0;
  for(auto& cdp : cfgData){
    cfgNum += cdp.data.size() * 32 / cfgDataWidth;
  }

  if(cfgAddrWidth > 16){
    CFGdata << "volatile unsigned int ";
  }else{
    CFGdata << "volatile unsigned short ";
  }
  CFGdata << CFGarrayName << "[" << cfgNum << "][" << (1 + cfgDataWidth / alignWidth) << "] __attribute__((aligned(" << cfgSpadDataByte << "))) = {\n";
  CFGdata << std::hex;
  int alignWidthHex = alignWidth/4;
  for(auto& cdp : cfgData){
    CFGdata << "\t\t{";
    for(auto data : cdp.data){      
      if(alignWidth == 32){
         CFGdata << "0x" << std::setw(alignWidthHex) << std::setfill('0') << data << ", ";
      }else{
        CFGdata << "0x" << std::setw(alignWidthHex) << std::setfill('0') << (data & 0xffff) << ", ";
        CFGdata << "0x" << std::setw(alignWidthHex) << std::setfill('0') << (data >> 16) << ", ";
      }
            
    }
    CFGdata << "0x" << std::setw(alignWidthHex) << std::setfill('0') << (cdp.addr) << "},\n";
  }
  CFGdata << std::dec << "\t};\n\n";

  KnToCfgData[kernel] = CFGdata.str();
  KnToCfgArrayInfo[kernel] = std::pair(CFGarrayName, cfgNum);

  return CFGdata.str();
}

/// @brief Get the config data
std::string CGRACallEmitter::GenerateCGRAConfig(
  ADORA::KernelOp& kernel, MapperSA* mapper){
  ADG* adg = mapper->getADG();
  Configuration cfg(mapper->_mapping);
  KnToConfiguration[kernel] = cfg;
  _adg = adg;
  return GenerateCGRAConfig(kernel, cfg, adg);
}



/// @brief Get the config and execution instructions of CGRA
void CGRACallEmitter::GenerateCGRACFGAndEXE(
  ADORA::KernelOp& kernel, Configuration cfg, ADG* adg){
  // adg->print();
  /////////////////////// CFG is generated in GenerateCGRACFGData()
  std::stringstream CFGandEXE;

  int cfgSpadDataByte = adg->cfgSpadDataWidth() / 8;
  int cfgAddrWidth = adg->cfgAddrWidth();
  int cfgDataWidth = adg->cfgDataWidth();
  int alignWidth = (cfgAddrWidth > 16) ? 32 : 16;
  assert(alignWidth >= cfgAddrWidth && cfgDataWidth >= alignWidth);

  if(!MapHasKey(KnToCfgData, kernel)){
    std::string cfgdata = GenerateCGRAConfig(kernel, cfg, adg);
    CFGandEXE << cfgdata << "\n";
  }
  assert(MapHasKey(KnToCfgData, kernel));
  std::string CFGarrayName = KnToCfgArrayInfo[kernel].first;
  int cfgNum = KnToCfgArrayInfo[kernel].second;

  /// replace the variable part of the config
  CFGandEXE << std::hex;
  for(auto elem : cfg.VarReplaceInfo){
    std::string varcfg_name = lookupVarConfigName(elem.first);
    for(auto& replace: elem.second) {
      CFGandEXE << std::dec << CFGarrayName << "[" << replace.Idx0 << "][" << replace.Idx1 << "] = ";
      assert(replace.lshift == 0 || replace.rshift == 0);
      if(replace.lshift == 0){
        // right shift
        CFGandEXE << std::hex << "(" << varcfg_name << " >> 0x" << replace.rshift << ")"
                  << " | " 
                  << std::dec << "(" << CFGarrayName  <<"[" << replace.Idx0 << "][" << replace.Idx1 << "]" 
                  << std::hex << " & 0x" << replace.getMask() << ");\n" ;
      }
      else{
        // left shift
        CFGandEXE << std::hex << "(" << varcfg_name << " << 0x" << replace.lshift << ")"
                  << " | " 
                  << std::dec << "(" << CFGarrayName <<"[" << replace.Idx0 << "][" << replace.Idx1 << "]" 
                  << std::hex << " & 0x" << replace.getMask() << ");\n" ;
      }
    }
  }
  CFGandEXE << std::dec;
 
    // _cfg_num = cfgNum;
    // _cfg_len = cfgNum * (alignWidth + cfgDataWidth) / 8; // length of config_addr and config_data in bytes
    // int cfgSpadSize = _adg->cfgSpadSize();
    // int cfgBaseAddr;
    // _ld_cfg_dep = 0;
    // if(_cfg_len <= cfgSpadSize - _old_cfg_status.end){
    //     cfgBaseAddr = _old_cfg_status.end;
    // }else if(_cfg_len <= _old_cfg_status.start){
    //     cfgBaseAddr = 0;
    // }else{ // cfg data space overlap last cfg data space
    //     cfgBaseAddr = 0;
    //     _ld_cfg_dep = LD_DEP_EX_LAST_TASK;
    // }
    // _old_cfg_status.start = cfgBaseAddr;
    // _old_cfg_status.end = cfgBaseAddr + (_cfg_len +  cfgSpadDataByte - 1) / cfgSpadDataByte * cfgSpadDataByte;
    // _old_cfg_status.end = std::min(_old_cfg_status.end, cfgSpadSize);
  int cfg_len = cfgNum * (alignWidth + cfgDataWidth) / 8; // length of config_addr and config_data in bytes
  int cfgBaseAddr = 0;
  int banks = adg->numIobNodes();
  int sizeofBank = adg->iobSpadBankSize();
  int cfgBaseAddrSpad = cfgBaseAddr + banks * sizeofBank; // cfg spad on top of iob spad
  int cfgBaseAddrCtrl = cfgBaseAddr / cfgSpadDataByte; // config base address the controller access
  
  BYTES_LIST iob_ens = _kernel_to_iob_ens[kernel];
  assert(iob_ens.As32b().size() == 1);

  CFGandEXE << "load_cfg((void*)" << CFGarrayName << ", 0x" << std::hex << cfgBaseAddrSpad << std::dec << ", " 
       << cfg_len << ", " << /*_task_id=*/"_task_id" << ", " << /*_ld_cfg_dep*/"LD_DEP_EX_LAST_TASK" << ");\n";
  CFGandEXE << "config(0x" << std::hex << cfgBaseAddrCtrl << std::dec << ", " << cfgNum << ", " << /*_task_id*/"_task_id" << ", " << /*_ex_dep*/ 0 << ");\n";
  // Compute execute dep_flag from the KernelOp's async token dependencies.
  // If kernel has no async deps (root task), use 0. Otherwise EX_DEP_ST_LAST_TASK.
  std::string ex_dep = "0";
  {
    auto deps = ADORA::getAsyncDeps(kernel.getOperation());
    if (!deps.empty()) {
      ex_dep = "EX_DEP_ST_LAST_TASK";
    }
  }
  CFGandEXE << "execute(" << iob_ens.As32b()[0] << std::dec << ", " << /*_task_id*/"_task_id" << ", " << ex_dep << ");\n";

  KnToCfgExe[kernel] = CFGandEXE.str();
  
  // std::cout << CFGandEXE.str() << std::endl;
}


/// @brief Get the config and execution instructions of CGRA
/// @param mapper The SA mapper which has completed mapping
void CGRACallEmitter::GenerateCGRACFGAndEXE(ADORA::KernelOp& kernel, MapperSA* mapper){
  /// why MapperSA (without&) will cause bug ?? memory leakage?
  /// Generate CGRA configuration
  ADG* adg = mapper->getADG();
  Configuration cfg(mapper->_mapping);
  KnToConfiguration[kernel] = cfg;
  _adg = adg;
  GenerateCGRACFGAndEXE(kernel, cfg, adg);
}

// CGRACallEmitter::~CGRACallEmitter(){}