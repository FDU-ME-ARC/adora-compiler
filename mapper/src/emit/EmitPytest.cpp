//===----------------------------------------------------------------------===//
//
// Copyright 2023-2024 The ADORA Authors.
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Affine/Utils.h"

#include <ctime>

#include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "ADORA/Dialect/ADORA/Utility/Utility.h"

#include "emit/Emit.h"
#include "emit/EmitPytest.h"
#include "emit/OpVisitor.h"

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::ADORA;
using namespace mlir::ADORA::ADORATensor;

//===----------------------------------------------------------------------===//
// Some tool functions
//===----------------------------------------------------------------------===//
/// @brief Get a new id for new op which will be emit to C
/// @param value_name_list
int NewValueNameId(const llvm::SmallDenseMap<mlir::Value, Op_Name_C> &value_name_list)
{
  int new_id = 0;
  for (auto &elem : value_name_list)
  {
    new_id = new_id < elem.second.id ? elem.second.id : new_id;
  }
  return new_id + 1;
}

void setEmitSkipAttr(mlir::Operation *op)
{
  op->setAttr("EmitSkip", mlir::UnitAttr::get(op->getContext()));
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

namespace
{
  class PyOpEmitter : public MLIROpVisitorBase<PyOpEmitter, bool>
  {
  public:
    PyOpEmitter(llvm::raw_ostream &os) : _os(os) {}
    PyOpEmitter(PytestEmitter &emitter, llvm::raw_ostream &os) : _pytestemitter(&emitter), _os(os) { setIndent(emitter.getIndent()); }
    using MLIROpVisitorBase::visitOp;

    /// Tool functions for emitting
    raw_ostream &indent() { return _os.indent(_indent); }
    void setIndent(unsigned newindent) { _indent = newindent; }

    /// pingpong indicator
    bool _pingpong = false;

    // --- PR4-F: async task tracking ---
    // task index counter, increments each time we emit a create_task
    int _taskIdx = 0;
    // map from BlockStore Operation* → Python task variable name
    llvm::DenseMap<mlir::Operation*, std::string> _storeToTask;

    // Compute stream IDs by graph-colouring the SSA token edges.
    // Called once at the start of emitBlock; results stored in _streamId.
    // Replaces the assign-streams IR-attribute approach: no IR attr needed.
    void computeStreamIds(mlir::Block &block) {
      if (!_pytestemitter) return;
      auto &sid = _pytestemitter->_streamId;
      sid.clear();
      int nextStream = 0;
      for (mlir::Operation &op : block) {
        if (!ADORA::isAsyncCapable(&op)) continue;
        int minPred = -1;
        for (mlir::Value tok : ADORA::getAsyncDeps(&op)) {
          auto *prod = tok.getDefiningOp();
          if (!prod) continue;
          auto it = sid.find(prod);
          if (it != sid.end())
            minPred = (minPred < 0) ? it->second
                                    : std::min(minPred, it->second);
        }
        sid[&op] = (minPred < 0)
                   ? (nextStream++ % PytestEmitter::kMaxStreams)
                   : minPred;
      }
    }

    // Collect Python task variable names that the given BlockStore op
    // depends on via SSA async tokens.
    llvm::SmallVector<std::string> getDepsTaskNames(ADORA::DataBlockStoreOp op) {
      return getDepsTaskNames(op.getOperation());
    }

    // Generic: collect task variable names for all async-token deps of any op.
    llvm::SmallVector<std::string> getDepsTaskNames(mlir::Operation *op) {
      // hw_dep_type = LD_DEP_NONE means this task's LOAD overlaps the previous
      // task's EXECUTE — no asyncio.gather() should be emitted before it.
      // LLMPipelineSchedulePass sets this attr on both the DataBlockLoadOp and
      // the DataBlockStoreOp of the same task for convenience.
      if (auto attr = op->getAttrOfType<mlir::StringAttr>("hw_dep_type"))
        if (attr.getValue() == "LD_DEP_NONE")
          return {};

      llvm::SmallVector<std::string> names;
      for (mlir::Value tok : ADORA::getAsyncDeps(op)) {
        mlir::Operation *producer = tok.getDefiningOp();
        if (!producer) continue;
        auto it = _storeToTask.find(producer);
        if (it != _storeToTask.end())
          names.push_back(it->second);
      }
      return names;
    }

    // True if this store's SSA token result has any downstream users.
    bool storeHasConsumers(ADORA::DataBlockStoreOp op) {
      if (mlir::Value tok = ADORA::getAsyncTokenOrNull(op.getOperation()))
        return !tok.use_empty();
      return false;
    }

    /// @brief emit a new op to python, add this one to op_name_list.
    /// @param mlirop the corresponding mlir operation
    /// @param type the C type of this operation
    /// @return
    std::string EmitNewValueAndGetName(mlir::Value v, const std::string type)
    {
      if (_pytestemitter->getValueNameList().count(v))
      {
        return _pytestemitter->lookupName(v);
      }
      Op_Name_C newopinfo;
      newopinfo.id = NewValueNameId(_pytestemitter->getValueNameList());
      newopinfo.type = type;
      _pytestemitter->appendValueNameList(v, newopinfo);
      return newopinfo.name();
    }

    template <typename opT>
    bool EmitBinary(opT op, std::string op_symbol)
    {
      if (op.getOperation()->hasAttr("EmitSkip"))
      {
        return true;
      }
      std::string type = getEmitType(op.getResult());
      std::string Lhs = _pytestemitter->lookupName(op.getLhs());
      if (Lhs == "")
        Lhs = ConstOpToValueStr[op.getLhs()];
      std::string Rhs = _pytestemitter->lookupName(op.getRhs());
      if (Rhs == "")
        Rhs = ConstOpToValueStr[op.getRhs()];
      assert(Lhs != "" && Rhs != "");

      indent() << EmitNewValueAndGetName(op.getResult(), type)
               << " = " << Lhs << " " << op_symbol << " " << Rhs << "\n";
      return true;
    }
    ///////////////////////////////
    /// ADORA dialect operations.
    ///////////////////////////////

    /// TODO: 1. support np.array 2.strided blockload 3.mutiple-dimmension list. only support one dimmension list rightnow
    bool visitOp(ADORA::DataBlockLoadOp op)
    {
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
      if (op.getOperation()->hasAttr("EmitSkip"))
      {
        return true;
      }
      indent() << "\n";
      indent() << "## " << op << "\n";

      std::string BLid = op.getId().str();
      ::llvm::ArrayRef<int64_t> SourceShape = op.getOriginalMemrefType().getShape();
      ::llvm::ArrayRef<int64_t> ResultShape = op.getResultType().getShape();
      // MemRefType SourceType = op.getOriginalMemref();
      // MemRefType ResultType = op.getResultType();
      if (SourceShape.size() == 0)
      {
        /// TODO:
        //// memref<f32> -> memref<2xf32>
        std::string Memref_BaseAddr = _pytestemitter->lookupName(op.getOriginalMemref());
        llvm::SmallVector<dfgIoInfo> DfgIoInfos = _pytestemitter->getDfgIoInfosFromBlockLoad(op);
        assert(DfgIoInfos.size() == 1);
        uint64_t DataBytes = op.getOriginalMemrefType().getElementTypeBitWidth() / 8;
        uint64_t DMA_Len = DataBytes;
        uint64_t spadbaddr = DfgIoInfos[0].addr;
        // int fuse = 0;
        std::stringstream load_data, spm_ptr;

        load_data << "idata.append(" << Memref_BaseAddr;
        // assert(DRAM_Offset_EachDim.size() == LenEachDim.size());
        // for(int i = 0; i < DRAM_Offset_EachDim.size(); i++){
        //   load_data << "[" << DRAM_Offset_EachDim[i]
        //             << ":" << DRAM_Offset_EachDim[i] << "+" << LenEachDim[i] << "]";
        // }
        load_data << ")";

        if (_pingpong == true)
        {
          spm_ptr << "iptrs.append(DeviceData("
                  << "0x" << std::hex << spadbaddr << "+" << std::dec << DMA_Len
                  << " if pingpong else 0x" << std::hex << spadbaddr
                  << ", " << std::dec << DMA_Len
                  << "))";
        }
        else
        {
          spm_ptr << "iptrs.append(DeviceData("
                  << "0x" << std::hex << spadbaddr
                  << ", " << std::dec << DMA_Len
                  << "))";
        }

        indent() << load_data.str() << "\n";
        indent() << spm_ptr.str() << "\n\n";

        return true;
      }

      // assert(SourceShape.size() == ResultShape.size());

      /// Get DMA_Len
      uint64_t DataBytes = op.getOriginalMemrefType().getElementTypeBitWidth() / 8;
      uint64_t DMA_Len = DataBytes;

      for (int r = ResultShape.size() - 1; r >= 0; r--)
      {
        DMA_Len = DMA_Len * ResultShape[r];
      }

      /// Get DRAM_BaseAddr
      std::string Memref_BaseAddr = _pytestemitter->lookupName(op.getOriginalMemref());

      /// Get DRAM_Offset
      std::vector<std::string> DRAM_Offset_EachDim;
      DRAM_Offset_EachDim.resize(SourceShape.size(), "-1");
      /// initialize DRAM_Offset_EachDim
      // for(auto elem : DRAM_Offset_EachDim){
      //   elem = "-1";
      // }

      assert(SourceShape.size() == op.getAffineMap().getResults().size());
      for (int exprIdx = 0; exprIdx < op.getAffineMap().getResults().size(); exprIdx++)
      {
        AffineExpr expr = op.getAffineMap().getResult(exprIdx);
        if (expr.getKind() == AffineExprKind::Constant)
        {
          std::string cstValue = std::to_string(expr.dyn_cast<AffineConstantExpr>().getValue());
          DRAM_Offset_EachDim[exprIdx] = cstValue;
        }
        else if (expr.getKind() == AffineExprKind::DimId)
        {
          for (int operandIdx = 0; operandIdx < op.getMapOperands().size(); operandIdx++)
          {
            mlir::Value operand = op.getMapOperands()[operandIdx];
            SmallVector<int> Dimensions = getOperandDimensionsInMap(/*dim=*/operandIdx, /*map=*/op.getAffineMap());
            // assert(Dimensions.size() == 1 && "We do not support one index is related to multiple dim of one array.");
            if (Dimensions.size() > 0 && Dimensions[0] == exprIdx)
            {
              std::string operandname = _pytestemitter->lookupName(operand);
              DRAM_Offset_EachDim[exprIdx] = operandname;
            }
          }
        }
        else
        {
          assert(false && "Unsupported Block Access.");
        }
      }

      /// Get DMA_Request_Len from every dim
      std::vector<int64_t> LenEachDim;
      for (int r = SourceShape.size() - 1; r >= 0; r--)
      {
        if (SourceShape.size() == ResultShape.size())
          LenEachDim.insert(LenEachDim.begin(), ResultShape[r]);
        else
          LenEachDim.insert(LenEachDim.begin(), 1); // Use dummy padding to bypass the original assertion when dimensions do not match.
      }

      /// SPAD_BaseAddr
      /// SPAD_Offset: keep increasing by DMA_Len.
      llvm::SmallVector<dfgIoInfo> DfgIoInfos = _pytestemitter->getDfgIoInfosFromBlockLoad(op);

      llvm::SmallVector<uint64_t> SPAD_BaseAddrs;
      for (dfgIoInfo elem : DfgIoInfos)
      {
        SPAD_BaseAddrs.push_back(elem.addr);
      }

      for (int i = 0; i < SPAD_BaseAddrs.size(); i++)
      {
        auto spadbaddr = SPAD_BaseAddrs[i];
        std::stringstream load_data, spm_ptr;

        // Handle dimension mismatch caused by Im2Col (use ravel flattening for slicing).
        if (SourceShape.size() != ResultShape.size())
        {
          std::string flat_offset = "";
          int64_t stride = 1;
          for (int j = SourceShape.size() - 1; j >= 0; j--)
          {
            if (j != SourceShape.size() - 1)
              flat_offset = " + " + flat_offset;
            std::string dim_val = DRAM_Offset_EachDim[j];
            if (dim_val.empty() || dim_val == "-1")
              dim_val = "0";
            flat_offset = "(" + dim_val + ") * " + std::to_string(stride) + flat_offset;
            stride *= SourceShape[j];
          }
          if (flat_offset.empty())
            flat_offset = "0";

          std::string role = "\"input\"";
          if (Memref_BaseAddr.find("arg_1") != std::string::npos) {
              role = "\"weight\"";
          }

          load_data << "idata.append(safe_slice_1d(" << Memref_BaseAddr << ", " << flat_offset << ", (";
          for (size_t j = 0; j < ResultShape.size(); ++j)
          {
            load_data << ResultShape[j] << (j == ResultShape.size() - 1 ? "" : ",");
          }
          load_data << "), role=" << role << "))";
        }
        else
        {
          // original logic
          load_data << "idata.append(" << Memref_BaseAddr;
          for (int j = 0; j < DRAM_Offset_EachDim.size(); j++)
          {
            if (j == 0)
              load_data << "[";
            if (op.hasStrides())
            {
              load_data << DRAM_Offset_EachDim[j] << ":" << DRAM_Offset_EachDim[j] << "+" << LenEachDim[j] * op.getStridesAsArrayRef()[j] << ":" << op.getStridesAsArrayRef()[j];
            }
            else
            {
              load_data << DRAM_Offset_EachDim[j] << ":" << DRAM_Offset_EachDim[j] << "+" << LenEachDim[j];
            }
            if (j == DRAM_Offset_EachDim.size() - 1)
              load_data << "]";
            else
              load_data << ",";
          }
          load_data << ")";
        }

        if (_pingpong == true)
        {
          spm_ptr << "iptrs.append(DeviceData("
                  << "0x" << std::hex << spadbaddr << "+" << std::dec << DMA_Len
                  << " if pingpong else 0x" << std::hex << spadbaddr
                  << ", " << std::dec << DMA_Len
                  << "))";
        }
        else
        {
          spm_ptr << "iptrs.append(DeviceData("
                  << "0x" << std::hex << spadbaddr
                  << ", " << std::dec << DMA_Len
                  << "))";
        }

        indent() << load_data.str() << "\n";
        indent() << spm_ptr.str() << "\n\n";
      }

      return true;
    }

    bool visitOp(ADORA::DataBlockStoreOp op)
    {
      /// DataBlockStoreOp can be seen as an opposite operation of memref subview op,
      /// 6 variables should be maintained:
      /// DMA_Len: length of one dma request.
      /// DRAM_BaseAddr: the base address in DRAM of the source memref.
      /// DRAM_Offset: the offset address related to affine map
      /// DMA_Request_Offset: keep changing. For this one, DMA_Request_Offset = 230 * i + 230 * 230 * j, i = [0, 7), j = [0, 3)
      /// SPAD_BaseAddr: the base address of Scratchpad memory of the destinated transfer
      /// SPAD_Offset: keep increasing by DMA_Len.
      if (op.getOperation()->hasAttr("EmitSkip"))
      {
        return true;
      }
      indent() << "\n";
      indent() << "## " << op << "\n";

      std::string BLid = op.getId().str();
      ::llvm::ArrayRef<int64_t> SourceShape = op.getSourceMemrefType().getShape();
      ::llvm::ArrayRef<int64_t> TargetShape = op.getTargetMemrefType().getShape();

      if (TargetShape.size() == 0)
      {
        //// memref<2xf32> -> memref<f32>
        std::string Memref_BaseAddr = _pytestemitter->lookupName(op.getTargetMemref());
        // llvm::SmallVector<dfgIoInfo> DfgIoInfos = _pytestemitter->getDfgIoInfosFromBlockLoad(op);
        dfgIoInfo DfgIoInfo = _pytestemitter->getDfgIoInfosFromBlockStore(op);
        // assert(DfgIoInfos.size() == 1);
        uint64_t spadbaddr = DfgIoInfo.addr;
        uint64_t DataBytes = op.getSourceMemrefType().getElementTypeBitWidth() / 8;
        uint64_t DMA_Len = DataBytes;
        // int fuse = 0;

        std::stringstream store_data, spm_ptr, olen;
        store_data << "odata.append(" << Memref_BaseAddr;
        store_data << ")";

        if (_pingpong == true)
        {
          spm_ptr << "optrs.append(DeviceData("
                  << "0x" << std::hex << spadbaddr << "+" << std::dec << DMA_Len
                  << " if pingpong else 0x" << std::hex << spadbaddr
                  << ", " << std::dec << DMA_Len
                  << "))";
        }
        else
        {
          spm_ptr << "optrs.append(DeviceData("
                  << "0x" << std::hex << spadbaddr
                  << ", " << std::dec << DMA_Len
                  << "))";
        }
        olen << "olen.append(" << std::dec << DMA_Len << ")";

        indent() << store_data.str() << "\n";
        indent() << spm_ptr.str() << "\n";
        indent() << olen.str() << "\n";

        if (IsLastBlockStoreOp(op))
        {
          int streamId = _pytestemitter ? (_pytestemitter->_streamId.count(op.getOperation()) ? _pytestemitter->_streamId[op.getOperation()] : 0) : 0;
          auto depNames = getDepsTaskNames(op);
          bool hasDeps = !depNames.empty();
          bool hasConsumers = storeHasConsumers(op);

          // Emit gather for upstream deps before running this task.
          if (hasDeps) {
            indent() << "await asyncio.gather(";
            for (size_t i = 0; i < depNames.size(); ++i) {
              if (i) _os << ", ";
              _os << depNames[i];
            }
            _os << ")\n";
          }

          if (_pingpong == true)
          {
            indent() << "await aux_stream_pingpong(stream=stream,\n";
            indent() << "\tconfig_id = 1 if pingpong==False else 2,\n";
            indent() << "\tiptrs=iptrs, idata=idata,\n";
            indent() << "\toptrs=optrs, odata=odata, olen =olen,\n";
            indent() << "\tpingpong=pingpong\n";
            indent() << ")\n\n";
            indent() << "configs.clear()\n";
            indent() << "iptrs.clear(), idata.clear()\n";
            indent() << "optrs.clear(), odata.clear(), olen.clear()\n\n";
            indent() << "pingpong = not pingpong\n";
          }
          else if (hasConsumers)
          {
            // This task feeds downstream consumers — launch concurrently.
            std::string taskVar = "task_" + std::to_string(_taskIdx++);
            _storeToTask[op.getOperation()] = taskVar;
            indent() << taskVar << " = asyncio.create_task(aux_stream(\n";
            indent() << "\tstream=stream_" << streamId << ", config=configs,\n";
            indent() << "\tiptrs=iptrs, idata=idata,\n";
            indent() << "\toptrs=optrs, odata=odata, olen=olen,\n";
            indent() << "))\n\n";
            indent() << "configs=[]; iptrs=[]; idata=[]\n";
            indent() << "optrs=[]; odata=[]; olen=[]\n\n";
          }
          else
          {
            indent() << "await aux_stream(\n";
            indent() << "\tstream=stream, config=configs,\n";
            indent() << "\tiptrs=iptrs, idata=idata,\n";
            indent() << "\toptrs=optrs, odata=odata, olen =olen,\n";
            indent() << ")\n\n";
            indent() << "configs.clear()\n";
            indent() << "iptrs.clear(), idata.clear()\n";
            indent() << "optrs.clear(), odata.clear(), olen.clear()\n\n";
          }
        }

        return true;
      }

      // assert(SourceShape.size() == TargetShape.size());

      /// Get DMA_Len
      uint64_t DataBytes = op.getSourceMemrefType().getElementTypeBitWidth() / 8;
      uint64_t DMA_Len = DataBytes;

      for (int r = SourceShape.size() - 1; r >= 0; r--)
      {
        DMA_Len = DMA_Len * SourceShape[r];
      }

      /// Get DRAM_BaseAddr
      std::string Memref_BaseAddr = _pytestemitter->lookupName(op.getTargetMemref());

      /// Get DRAM_Offset
      std::vector<std::string> DRAM_Offset_EachDim;
      DRAM_Offset_EachDim.resize(TargetShape.size(), "-1");
      /// initialize DRAM_Offset_EachDim
      for (auto elem : DRAM_Offset_EachDim)
      {
        elem = "-1";
      }

      assert(TargetShape.size() == op.getAffineMap().getResults().size());
      for (int exprIdx = 0; exprIdx < op.getAffineMap().getResults().size(); exprIdx++)
      {
        AffineExpr expr = op.getAffineMap().getResult(exprIdx);
        if (expr.getKind() == AffineExprKind::Constant)
        {
          std::string cstValue = std::to_string(expr.dyn_cast<AffineConstantExpr>().getValue());
          DRAM_Offset_EachDim[exprIdx] = cstValue;
        }
        else if (expr.getKind() == AffineExprKind::DimId)
        {
          for (int operandIdx = 0; operandIdx < op.getMapOperands().size(); operandIdx++)
          {
            mlir::Value operand = op.getMapOperands()[operandIdx];
            SmallVector<int> Dimensions = getOperandDimensionsInMap(/*dim=*/operandIdx, /*map=*/op.getAffineMap());
            // assert(Dimensions.size() == 1 && "We do not support one index is related to multiple dim of one array.");
            if (Dimensions.size() > 0 && Dimensions[0] == exprIdx)
            {
              std::string operandname = _pytestemitter->lookupName(operand);
              DRAM_Offset_EachDim[exprIdx] = operandname;
            }
          }
          // SmallVector<int>Dimensions = getOperandDimensionsInMap(/*dim=*/operandIdx, /*map=*/op.getAffineMap());
          // assert(Dimensions.size() == 1);
          // mlir::Value operand = op.getMapOperands()[exprIdx];
        }
        else
        {
          assert(false && "Unsupported Block Access.");
        }
      }

      std::vector<int64_t> LenEachDim;
      bool continuous = true;
      for (int r = SourceShape.size() - 1; r >= 0; r--)
      {
        if (SourceShape.size() == TargetShape.size())
          LenEachDim.insert(LenEachDim.begin(), SourceShape[r]);
        else
          LenEachDim.insert(LenEachDim.begin(), 1); // dummy
      }

      /// SPAD_BaseAddr
      /// SPAD_Offset: keep increasing by DMA_Len.
      dfgIoInfo DfgIoInfos = _pytestemitter->getDfgIoInfosFromBlockStore(op);
      llvm::SmallVector<uint64_t> SPAD_BaseAddrs;
      SPAD_BaseAddrs.push_back(DfgIoInfos.addr);

      /// Emit pytest
      for (int i = 0; i < SPAD_BaseAddrs.size(); i++)
      {
        auto spadbaddr = SPAD_BaseAddrs[i];
        std::stringstream store_data, spm_ptr, olen;

        // Handle dimension mismatch by allocating the write-back region using a one-dimensional flattened layout.
        if (SourceShape.size() != TargetShape.size())
        {
          std::string pad(_indent, ' ');
          store_data << "tmp_out = np.zeros((";
          for (size_t j = 0; j < SourceShape.size(); ++j) store_data << SourceShape[j] << (j == SourceShape.size() - 1 ? "" : ",");
          store_data << "), dtype=" << Memref_BaseAddr << ".dtype)\n";

          store_data << pad << "odata.append(tmp_out)\n";

          std::string starts_str = "(";
          for (int j = 0; j < DRAM_Offset_EachDim.size(); j++) {
              std::string dim_val = DRAM_Offset_EachDim[j];
              if (dim_val.empty() || dim_val == "-1") dim_val = "0";
              starts_str += dim_val;
              if (j != DRAM_Offset_EachDim.size() - 1) starts_str += ", ";
          }
          starts_str += ")";

          store_data << pad << "writeback_tasks.append((" << Memref_BaseAddr << ", " << starts_str << ", tmp_out))";
        }
        else
        {
          store_data << "odata.append(" << Memref_BaseAddr;
          for (int j = 0; j < DRAM_Offset_EachDim.size(); j++)
          {
            if (j == 0)
              store_data << "[";
            if (op.hasStrides())
            {
              store_data << DRAM_Offset_EachDim[j] << ":" << DRAM_Offset_EachDim[j] << "+" << LenEachDim[j] * op.getStridesAsArrayRef()[j] << ":" << op.getStridesAsArrayRef()[j];
            }
            else
            {
              store_data << DRAM_Offset_EachDim[j] << ":" << DRAM_Offset_EachDim[j] << "+" << LenEachDim[j];
            }
            if (j == DRAM_Offset_EachDim.size() - 1)
              store_data << "]";
            else
              store_data << ",";
          }
          store_data << ")";
        }

        if (_pingpong == true)
        {
          spm_ptr << "optrs.append(DeviceData("
                  << "0x" << std::hex << spadbaddr << "+" << std::dec << DMA_Len
                  << " if pingpong else 0x" << std::hex << spadbaddr
                  << ", " << std::dec << DMA_Len
                  << "))";
        }
        else
        {
          spm_ptr << "optrs.append(DeviceData("
                  << "0x" << std::hex << spadbaddr
                  << ", " << std::dec << DMA_Len
                  << "))";
        }

        olen << "olen.append(" << std::dec << DMA_Len << ")";

        indent() << store_data.str() << "\n";
        indent() << spm_ptr.str() << "\n";
        indent() << olen.str() << "\n\n";
      }

      // indent() << "\n";

      if (IsLastBlockStoreOp(op))
      {
        int streamId = _pytestemitter ? (_pytestemitter->_streamId.count(op.getOperation()) ? _pytestemitter->_streamId[op.getOperation()] : 0) : 0;
        auto depNames = getDepsTaskNames(op);
        bool hasDeps = !depNames.empty();
        bool hasConsumers = storeHasConsumers(op);

        if (hasDeps) {
          indent() << "await asyncio.gather(";
          for (size_t i = 0; i < depNames.size(); ++i) {
            if (i) _os << ", ";
            _os << depNames[i];
          }
          _os << ")\n";
        }

        if (_pingpong == true)
        {
          indent() << "await aux_stream_pingpong(stream=stream,\n";
          indent() << "\tconfig_id = 1 if pingpong==False else 2,\n";
          indent() << "\tiptrs=iptrs, idata=idata,\n";
          indent() << "\toptrs=optrs, odata=odata, olen =olen,\n";
          indent() << "\tpingpong=pingpong\n";
          indent() << ")\n\n";
          indent() << "iptrs.clear(), idata.clear()\n";
          indent() << "optrs.clear(), odata.clear(), olen.clear()\n\n";
          indent() << "pingpong = not pingpong\n";
        }
        else if (hasConsumers)
        {
          std::string taskVar = "task_" + std::to_string(_taskIdx++);
          _storeToTask[op.getOperation()] = taskVar;
          indent() << taskVar << " = asyncio.create_task(aux_stream(\n";
          indent() << "\tstream=stream_" << streamId << ", config=configs,\n";
          indent() << "\tiptrs=iptrs, idata=idata,\n";
          indent() << "\toptrs=optrs, odata=odata, olen=olen,\n";
          indent() << "))\n\n";
          indent() << "configs=[]; iptrs=[]; idata=[]\n";
          indent() << "optrs=[]; odata=[]; olen=[]\n\n";
        }
        else
        {
          indent() << "await aux_stream(\n";
          indent() << "\tstream=stream, config=configs,\n";
          indent() << "\tiptrs=iptrs, idata=idata,\n";
          indent() << "\toptrs=optrs, odata=odata, olen =olen,\n";
          indent() << ")\n\n";
          indent() << "configs.clear()\n";
          indent() << "iptrs.clear(), idata.clear()\n";
          indent() << "optrs.clear(), odata.clear(), olen.clear()\n\n";
        }
      }

      return true;
    }

    bool visitOp(ADORA::LocalMemAllocOp op)
    {
      if (op.getOperation()->hasAttr("EmitSkip"))
      {
        return true;
      }
      /// %2 = ADORA.LocalMemAlloc memref<20x20xi32>  {Id = "2", KernelName = "IntVecAdd"}
      indent() << "\n";
      indent() << "## " << op << "\n";

      std::string BLid = op.getId().str();
      ::llvm::ArrayRef<int64_t> ResultShape = op.getResultType().getShape();

      uint64_t DataBytes = op.getResultType().getElementTypeBitWidth() / 8;
      uint64_t Len = DataBytes;
      for (int r = ResultShape.size() - 1; r >= 0; r--)
      {
        Len = Len * ResultShape[r];
      }

      dfgIoInfo DfgIoInfos = _pytestemitter->getSPMInfosFromLocalAlloc(op).second;
      uint64_t SPAD_BaseAddr = DfgIoInfos.addr;

      std::stringstream alloc_ptr;

      if (_pingpong == true)
      {
        alloc_ptr << "data_ptr.append(DeviceData("
                  << "0x" << std::hex << SPAD_BaseAddr << "+" << std::dec << Len
                  << " if pingpong else 0x" << std::hex << SPAD_BaseAddr
                  << ", " << std::dec << Len
                  << "))";
      }
      else
      {
        alloc_ptr << "data_ptr.append(DeviceData("
                  << "0x" << std::hex << SPAD_BaseAddr
                  << ", " << std::dec << Len
                  << "))";
      }

      indent() << alloc_ptr.str() << "\n";

      return true;
    }

    bool visitOp(ADORA::KernelOp op)
    {
      if (op.getOperation()->hasAttr("EmitSkip"))
      {
        return true;
      }
      indent() << "\n";
      if (!MapHasKey(_pytestemitter->KnToCfgExe, op))
      {
        ADG *adg = _pytestemitter->getADG();
        if (adg != nullptr) {
          _pytestemitter->GenerateCGRACFGAndEXE(op, _pytestemitter->KnToConfiguration[op], adg);
        }
      }
      // Await any async DMA tasks this kernel depends on before launching.
      auto depTasks = getDepsTaskNames(op.getOperation());
      if (!depTasks.empty()) {
        indent() << "await asyncio.gather(";
        for (size_t i = 0; i < depTasks.size(); ++i) {
          if (i > 0) _os << ", ";
          _os << depTasks[i];
        }
        _os << ")\n";
      }

      if (!op.getKernelName().empty())
      {
        indent() << "### " << op.getKernelName() << "\n";
      }

      std::vector<std::string> strs = split_str_by_char(_pytestemitter->KnToCfgExe[op], '\n');
      for (std::string str : strs)
      {
        indent() << str << "\n";
      }

      indent() << "\n\n";
      return true;
    }

    ///////////////////////////////
    /// ADORA Tensor dialect operations.
    ///////////////////////////////
    // =====================================================================
    // Emit TensorOp Config Init: Extract hardware stream initialization,
    // and hand over the loop parsing back to the natural AST traversal flow.
    // =====================================================================
    bool visitGemmLikeOp(mlir::Operation *gemmop)
    {
      indent() << "#######################################\n";
      indent() << "### Emit TensorOp Config Init: " << *gemmop << "\n";
      indent() << "#######################################\n";

      // 1. Look ahead to find the bound KernelOp
      ADORA::KernelOp kernel = nullptr;
      mlir::Operation *op = gemmop->getNextNode();
      while (op)
      {
        if (isa<mlir::affine::AffineForOp>(op) && op->hasAttr("ADORAGemm"))
        {
          op->walk([&](ADORA::KernelOp k)
                   { kernel = k; });
          break;
        }
        op = op->getNextNode();
      }

      if (!kernel)
        return true;

      // 2. Generate all Pingpong stream initialization structures
      std::string knName = kernel.getKernelName();
      indent() << "ptrs_ping, ptrs_pong = [], []\n";

      for (auto elem : _pytestemitter->getLoadToSPMInfosMap())
      {
        ADORA::DataBlockLoadOp load = elem.first;
        if (load.getOperation()->hasAttr("Pingpong") && findElement(load.getKernelNameAsStrVector(), kernel.getKernelName()) != -1)
        {
          for (auto pair : elem.second)
          {
            int base_addr = pair.second.addr;
            int len = getByteSizeFromMemref(load.getResultType());
            std::stringstream ss_ping, ss_pong;
            ss_ping << "ptrs_ping.append(DeviceData(0x" << std::hex << base_addr << std::dec << ", " << len << "))";
            ss_pong << "ptrs_pong.append(DeviceData(0x" << std::hex << base_addr << std::dec << "+" << len << ", " << len << "))";
            indent() << ss_ping.str() << "\n";
            indent() << ss_pong.str() << "\n";
          }
        }
      }

      for (auto elem : _pytestemitter->getLocalAllocToSPMMap())
      {
        ADORA::LocalMemAllocOp alloc = elem.first;
        if (alloc.getOperation()->hasAttr("Pingpong") && findElement(alloc.getKernelNameAsStrVector(), kernel.getKernelName()) != -1)
        {
          int base_addr = elem.second.second.addr;
          int len = getByteSizeFromMemref(alloc.getMemrefType());
          std::stringstream ss_ping, ss_pong;
          ss_ping << "ptrs_ping.append(DeviceData(0x" << std::hex << base_addr << std::dec << ", " << len << "))";
          ss_pong << "ptrs_pong.append(DeviceData(0x" << std::hex << base_addr << std::dec << "+" << len << ", " << len << "))";
          indent() << ss_ping.str() << "\n";
          indent() << ss_pong.str() << "\n";
        }
      }

      BYTES_LIST iob_ens = _pytestemitter->getIobEns(kernel);
      BYTES_LIST tile_ens = _pytestemitter->getTileEns(kernel);
      std::stringstream iobens_ss, tileens_ss;
      for (int _ = 0; _ < iob_ens.size(); _++)
      {
        iobens_ss << iob_ens.getByte(_) << (_ != iob_ens.size() - 1 ? "," : "");
      }
      for (int _ = 0; _ < tile_ens.size(); _++)
      {
        tileens_ss << tile_ens.getByte(_) << (_ != tile_ens.size() - 1 ? "," : "");
      }

      indent() << "pingpong = False\n";
      indent() << "stream = runtime.create_stream()\n";
      indent() << "config_" << knName << " = DeviceConfig(config_values=cfgbit_" << knName << ", iob_en=[" << iobens_ss.str() << "], tile_en=[" << tileens_ss.str() << "], data_ptr=data_ptr)\n";
      indent() << "config_" << knName << "_ping = DeviceConfig(config_values=cfgbit_" << knName << "_ping, iob_en=[" << iobens_ss.str() << "], tile_en=[" << tileens_ss.str() << "], data_ptr=ptrs_ping)\n";
      indent() << "config_" << knName << "_pong = DeviceConfig(config_values=cfgbit_" << knName << "_pong, iob_en=[" << iobens_ss.str() << "], tile_en=[" << tileens_ss.str() << "], data_ptr=ptrs_pong)\n";
      indent() << "await aux_stream_pingpong_init(stream, [config_" << knName << ", config_" << knName << "_ping, config_" << knName << "_pong])\n\n";

      // 3. Turn on the global Pingpong switch!
      // This informs the subsequent affine.for traversal to emit pingpong logic.
      _pingpong = true;

      return true;
    }

    bool visitOp(ADORA::ADORATensor::GemmOp op) { return visitGemmLikeOp(op.getOperation()); }
    bool visitOp(ADORA::ADORATensor::ConvOp op) { return visitGemmLikeOp(op.getOperation()); }

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
    bool visitOp(func::CallOp op)
    {
      if (op.getOperation()->hasAttr("EmitSkip"))
        return true;

      // Resolve callee symbol name.
      std::string callee = op.getCallee().str();
      if (callee.empty())
      {
        op.emitError("func.call: failed to resolve callee symbol.");
        return false;
      }

      // Build argument list: runtime is always the first argument.
      std::stringstream callArgs;
      callArgs << "runtime";
      for (mlir::Value operand : op.getOperands())
      {
        // Reuse previously emitted SSA names; fallback to inline constants.
        std::string operandName = _pytestemitter->lookupName(operand);
        if (operandName.empty())
          operandName = ConstOpToValueStr[operand];
        if (operandName.empty())
        {
          op.emitError("func.call: failed to resolve an operand name.");
          return false;
        }
        callArgs << ", " << operandName;
      }

      std::stringstream callExpr;
      callExpr << "await " << callee << "(" << callArgs.str() << ")";

      auto getResultEmitName = [&](mlir::Value v) -> std::string
      {
        mlir::Type ty = v.getType();
        if (ty.isa<MemRefType>())
          return EmitNewValueAndGetName(v, "ndarray");
        return EmitNewValueAndGetName(v, getEmitType(v));
      };

      // No return value: emit a plain awaited call.
      if (op.getNumResults() == 0)
      {
        indent() << callExpr.str() << "\n";
        return true;
      }

      // Single return value: bind to one emitted name.
      if (op.getNumResults() == 1)
      {
        mlir::Value res = op.getResult(0);
        indent() << getResultEmitName(res) << " = "
                 << callExpr.str() << "\n";
        return true;
      }

      // Multiple return values: tuple-unpack in Python.
      std::stringstream lhs;
      for (int i = 0; i < op.getNumResults(); ++i)
      {
        mlir::Value res = op.getResult(i);
        lhs << getResultEmitName(res);
        if (i + 1 != op.getNumResults())
          lhs << ", ";
      }
      indent() << lhs.str() << " = " << callExpr.str() << "\n";
      return true;
    }

    bool visitOp(memref::AllocOp op)
    {
      // Emit a host-side buffer allocation in Python.
      // We use NumPy arrays as a lightweight representation for memref buffers.
      // Notes:
      //  - Only supports static-shaped memrefs for now.
      //  - For rank-0 memref, we emit a scalar placeholder (0).
      //  - Dynamic dims require runtime values, which are not handled here.
      if (op.getOperation()->hasAttr("EmitSkip"))
        return true;

      mlir::MemRefType mt = op.getType();
      ArrayRef<int64_t> shape = mt.getShape();

      // Rank-0 memref: treat as a scalar slot.
      if (shape.size() == 0)
      {
        std::string elemType = getEmitType(mt.getElementType());
        indent() << EmitNewValueAndGetName(op.getResult(), elemType) << " = 0\n";
        return true;
      }

      // Reject dynamic shapes for now to avoid incorrect Python.
      for (int64_t d : shape)
      {
        if (d == mlir::ShapedType::kDynamic)
        {
          op.emitError("memref.alloc with dynamic shape is not supported in PyEmitter yet.");
          return false;
        }
      }

      // Emit: v = np.zeros((d0, d1, ...), dtype=np.<type>)
      // Keep it simple: we only create a buffer, and let subsequent stores fill it.
      std::string name = EmitNewValueAndGetName(op.getResult(), "ndarray");

      // Map element type to a NumPy dtype string.
      // Keep conservative defaults; adjust if you already have a helper elsewhere.
      auto et = mt.getElementType();
      std::string npDType = "np.float32";
      if (et.isF64())
        npDType = "np.float64";
      else if (et.isF32())
        npDType = "np.float32";
      else if (et.isF16())
        npDType = "np.float16";
      else if (et.isBF16())
        npDType = "np.float16"; // NumPy has no native bf16; use fp16 as placeholder.
      else if (et.isInteger(1))
        npDType = "np.bool_";
      else if (et.isInteger(8))
        npDType = "np.int8";
      else if (et.isInteger(16))
        npDType = "np.int16";
      else if (et.isInteger(32))
        npDType = "np.int32";
      else if (et.isInteger(64))
        npDType = "np.int64";
      else if (et.isUnsignedInteger(8))
        npDType = "np.uint8";
      else if (et.isUnsignedInteger(16))
        npDType = "np.uint16";
      else if (et.isUnsignedInteger(32))
        npDType = "np.uint32";
      else if (et.isUnsignedInteger(64))
        npDType = "np.uint64";
      else
      {
        op.emitError("unsupported element type for memref.alloc in PyEmitter.");
        return false;
      }

      // Emit the Python allocation line.
      indent() << name << " = np.zeros((";
      for (size_t i = 0; i < shape.size(); ++i)
      {
        _os << shape[i] << (i + 1 != shape.size() ? ", " : "");
      }
      _os << "), dtype=" << npDType << ")\n";

      // Auto-Padding
      indent() << "try:\n";
      _indent += 4;
      indent() << "if 'arg_0' in locals() and arg_0.ndim == " << shape.size() << " and arg_0.shape[1] == " << name << ".shape[1] and all(d_alloc >= d_arg for d_alloc, d_arg in zip(" << name << ".shape, arg_0.shape)):\n";
      _indent += 4;
      indent() << "pad_slices = tuple(slice((d_alloc - d_arg) // 2, (d_alloc - d_arg) // 2 + d_arg) for d_alloc, d_arg in zip(" << name << ".shape, arg_0.shape))\n";
      indent() << name << "[pad_slices] = arg_0\n";
      _indent -= 8;
      indent() << "except NameError:\n";
      _indent += 4;
      indent() << "pass\n";
      _indent -= 4;

      return true;
    }

    bool visitOp(memref::CopyOp op)
    {
      // Emit a Python-level copy between two memrefs.
      // For rank-N buffers: dst[...] = src[...]
      // For rank-0 memrefs: dst = src
      if (op.getOperation()->hasAttr("EmitSkip"))
        return true;

      mlir::Value srcV = op.getSource();
      mlir::Value dstV = op.getTarget();

      std::string src = _pytestemitter->lookupName(srcV);
      if (src.empty())
        src = ConstOpToValueStr[srcV];

      std::string dst = _pytestemitter->lookupName(dstV);
      if (dst.empty())
        dst = ConstOpToValueStr[dstV];

      if (src.empty() || dst.empty())
      {
        op.emitError("memref.copy: failed to resolve source/target names.");
        return false;
      }

      auto srcTy = srcV.getType().dyn_cast<mlir::MemRefType>();
      auto dstTy = dstV.getType().dyn_cast<mlir::MemRefType>();
      if (!srcTy || !dstTy)
      {
        op.emitError("memref.copy: source/target must be memref types.");
        return false;
      }

      // Rank-0: treat as scalar assignment.
      if (srcTy.getRank() == 0 && dstTy.getRank() == 0)
      {
        indent() << dst << " = " << src << "\n";
        return true;
      }

      // For higher rank: use full-slice assignment.
      indent() << dst << "[...] = " << src << "[...]\n";
      return true;
    }
    bool visitOp(memref::AllocaOp op)
    {
      mlir::MemRefType mt = op.getType();
      assert(mt.getShape().size() == 0);
      mlir::Type t = mt.getElementType();

      std::string type = getEmitType(t);
      indent() << EmitNewValueAndGetName(op.getResult(), type) << "= 0\n";

      return true;
    }

    bool visitOp(func::ReturnOp op)
    {
      // Synchronize the hardware stream
      indent() << "await stream.synchronize()\n";

      indent() << "try:\n";
      _indent += 4;
      indent() << "apply_writeback_tasks(writeback_tasks)\n";
      _indent -= 4;
      indent() << "except NameError:\n";
      _indent += 4;
      indent() << "pass\n";
      _indent -= 4;

      // If there are return values, return them
      if (op.getNumOperands() > 0)
      {
        indent() << "return ";
        for (int i = 0; i < op.getNumOperands(); ++i)
        {
          mlir::Value operand = op.getOperand(i);
          std::string name = _pytestemitter->lookupName(operand);
          if (name.empty())
            name = ConstOpToValueStr[operand];

          _os << name << (i != op.getNumOperands() - 1 ? ", " : "");
        }
        _os << "\n";
      }
      return true;
    }

    /// SCF statements.
    // bool visitOp(scf::ForOp op) { return emitter.emitScfFor(op), true; };
    // bool visitOp(scf::IfOp op) { return emitter.emitScfIf(op), true; };
    // bool visitOp(scf::ParallelOp op) { return false; };
    // bool visitOp(scf::ReduceOp op) { return false; };
    // bool visitOp(scf::ReduceReturnOp op) { return false; };
    // bool visitOp(scf::YieldOp op) { return emitter.emitScfYield(op), true; };

    /// CF
    bool visitOp(cf::BranchOp op)
    {
      _pytestemitter->emitBlock(*(op.getDest()), _os);
      return true;
    }

    /// Affine statements.
    bool visitOp(affine::AffineForOp op)
    {
      if (op.getOperation()->hasAttr("EmitSkip"))
      {
        return true;
      }

      indent() << "for ";
      auto iterVar = op.getInductionVar();
      assert(op.getLowerBoundMap().getResults().size() == 1);
      _os << EmitNewValueAndGetName(iterVar, "int") << " in range(";
      _os << op.getLowerBoundMap().getResult(0) << ", ";

      // Emit loop invariant(upper bound)
      assert(op.getUpperBoundMap().getResults().size() == 1);
      _os << op.getUpperBoundMap().getResult(0) << ", ";

      // Emit loop step
      _os << op.getStep().getSExtValue() << "):\n";

      if (op.getOperation()->hasAttr("ADORAGemm"))
      {
        _pytestemitter->emitGemmBlock(*(op.getBody()), _os);
      }
      else
      {
        _pytestemitter->emitBlock(*(op.getBody()), _os);
      }

      _os << "\n";
      return true;
    }

    /// SCF statements.
    bool visitOp(scf::ForOp op)
    {
      if (op.getOperation()->hasAttr("EmitSkip"))
      {
        return true;
      }

      indent() << "for ";
      auto iterVar = op.getInductionVar();
      
      // Emit lower bound.
      std::string lbStr = _pytestemitter->lookupName(op.getLowerBound());
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
      _os << EmitNewValueAndGetName(iterVar, "int") << " in range(" << lbStr << ", ";

      // Emit upper bound.
      std::string ubStr = _pytestemitter->lookupName(op.getUpperBound());
      if(ubStr.empty()) {
        auto cstOp = op.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
        if(cstOp) {
          ubStr = std::to_string(cstOp.value());
        } else {
          llvm::errs() << "Warning: could not find ub value in scf.for\n";
          ubStr = "0";
        }
      }
      _os << ubStr << ", ";

      // Emit step.
      std::string stepStr = _pytestemitter->lookupName(op.getStep());
      if(stepStr.empty()) {
        auto cstOp = op.getStep().getDefiningOp<arith::ConstantIndexOp>();
        if(cstOp) {
          stepStr = std::to_string(cstOp.value());
        } else {
          llvm::errs() << "Warning: could not find step value in scf.for\n";
          stepStr = "1";
        }
      }
      _os << stepStr << "):\n";

      if (op.getOperation()->hasAttr("ADORAGemm"))
      {
        _pytestemitter->emitGemmBlock(*(op.getBody()), _os);
      }
      else
      {
        _pytestemitter->emitBlock(*(op.getBody()), _os);
      }

      _os << "\n";
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
    bool visitOp(::mlir::affine::AffineLoadOp op) { return true; }
    bool visitOp(::mlir::arith::AddFOp op) { return true; }
    bool visitOp(::mlir::arith::SubFOp op) { return true; }
    bool visitOp(::mlir::arith::MulFOp op) { return true; }
    bool visitOp(::mlir::arith::DivFOp op) { return true; }
    bool visitOp(::mlir::arith::RemFOp op) { return true; }
    bool visitOp(::mlir::arith::CmpFOp op) { return true; }
    bool visitOp(::mlir::affine::AffineStoreOp op)
    {
      std::string memref = _pytestemitter->lookupName(op.getMemref());
      // if (memref == "")
      // {
      //   // If the memref name is not found, it is likely a device-only buffer
      //   // (e.g., LocalMemAlloc). Such buffers cannot be directly assigned from
      //   // host-side Python code. Emit `pass` to maintain valid Python syntax.
      //   indent() << "pass  # Host cannot directly store to device memref, skipping.\n";
      //   return true;
      // }

      // std::string type = getEmitType(op.getResult());
      std::string value;
      if (isa<LLVM::UndefOp>(op.getValue().getDefiningOp()))
      {
        value = "0";
      }
      else
      {
        value = _pytestemitter->lookupName(op.getValue());
        // assert("Unsupported!\n");
        if (value == "")
          value = ConstOpToValueStr[op.getValue()];
      }

      // assert(op.getMemref().getType().cast<MemRefType>().getShape().size() == 0
      //     || (op.getMemref().getType().cast<MemRefType>() == 1 && op.getMemref().getType().cast<MemRefType>().isDynamicDim()));
      if (op.getMemref().getType().cast<MemRefType>().getShape().size() == 0)
      {
        indent() << memref << " = " << value << "\n";
      }
      else
      {
        //// affine index operand is simplified
        std::stringstream ss;
        ss << memref << "[";
        ::mlir::Operation::operand_range indices = op.getIndices();
        for (int i = 0; i < indices.size(); i++)
        {
          mlir::Value operand = indices[i];
          ss << _pytestemitter->lookupName(operand);
          if (i != indices.size() - 1)
          {
            ss << ",";
          }
        }

        ss << "]" << " = " << value;
        indent() << ss.str() << "\n";
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
    bool visitOp(arith::ConstantOp op)
    {
      // This indicates the constant type is scalar (float, integer, or bool).
      // if (isDeclared(op.getResult()))
      //   return;
      // arith::ConstantOp constin = dyn_cast<arith::ConstantOp>(in);
      // indent();
      mlir::Attribute constattr = op.getOperation()->getAttr(op.getValueAttrName());

      if (isa<FloatAttr>(constattr))
      {
        FloatAttr floatattr = dyn_cast<FloatAttr>(constattr);
        if (floatattr.getType().isF64())
        {
          double value = floatattr.getValueAsDouble();
          ConstOpToValueStr[op.getResult()] = std::to_string(value);
          // std::string name_c = EmitNewValueAndGetName(op.getResult(), "double");
          // _os << "double " << name_c << " = " << std::to_string(value) << ";\n";
        }
        else if (floatattr.getType().isF32())
        {
          double value = floatattr.getValueAsDouble();
          ConstOpToValueStr[op.getResult()] = std::to_string(value);
          // std::string name_c = EmitNewValueAndGetName(op.getResult(), "float");
          // _os << "float " << name_c << " = " << std::to_string(value) << ";\n";
        }
        else if (floatattr.getType().isBF16())
        {
          double value = floatattr.getValueAsDouble();
          ConstOpToValueStr[op.getResult()] = std::to_string(value);
          // std::string name_c = EmitNewValueAndGetName(op.getResult(), "float");
          // _os << "float " << name_c << " = " << std::to_string(value) << ";\n";
        }
        else if (floatattr.getType().isF16())
        {
          double value = floatattr.getValueAsDouble();
          ConstOpToValueStr[op.getResult()] = std::to_string(value);
          // std::string name_c = EmitNewValueAndGetName(op.getResult(), "float");
          // _os << "float " << name_c << " = " << std::to_string(value) << ";\n";
        }
      }
      else if (isa<IntegerAttr>(constattr))
      {
        IntegerAttr intattr = dyn_cast<IntegerAttr>(constattr);
        if (intattr.getType().isInteger(16))
        {
          int value = intattr.getInt();
          ConstOpToValueStr[op.getResult()] = std::to_string(value);
          // std::string name_c = EmitNewValueAndGetName(op.getResult(), "int16_t");
          // _os << "int16_t " << name_c << " = " << std::to_string(value) << ";\n";
        }
        else if (intattr.getType().isInteger(32))
        {
          int value = intattr.getInt();
          ConstOpToValueStr[op.getResult()] = std::to_string(value);
          // std::string name_c = EmitNewValueAndGetName(op.getResult(), "int32_t");
          // _os << "int32_t " << name_c << " = " << std::to_string(value) << ";\n";
        }
        else if (intattr.getType().isInteger(64))
        {
          int value = intattr.getInt();
          ConstOpToValueStr[op.getResult()] = std::to_string(value);
          // std::string name_c = EmitNewValueAndGetName(op.getResult(), "int64_t");
          // _os << "int64_t " << name_c << " = " << std::to_string(value) << ";\n";
        }
        else if (intattr.getType().isUnsignedInteger(16))
        {
          int value = intattr.getInt();
          ConstOpToValueStr[op.getResult()] = std::to_string(value);
          // std::string name_c = EmitNewValueAndGetName(op.getResult(), "uint16_t");
          // _os << "uint16_t " << name_c << " = " << std::to_string(value) << ";\n";
        }
        else if (intattr.getType().isUnsignedInteger(32))
        {
          int value = intattr.getInt();
          ConstOpToValueStr[op.getResult()] = std::to_string(value);
          // std::string name_c = EmitNewValueAndGetName(op.getResult(), "uint32_t");
          // _os << "uint32_t " << name_c << " = " << std::to_string(value) << ";\n";
        }
        else if (intattr.getType().isUnsignedInteger(64))
        {
          int value = intattr.getInt();
          ConstOpToValueStr[op.getResult()] = std::to_string(value);
          // std::string name_c = EmitNewValueAndGetName(op.getResult(), "uint64_t");
          // _os << "uint64_t " << name_c << " = " << std::to_string(value) << ";\n";
        }
        else if (intattr.getType().isIndex())
        {
          int value = intattr.getInt();
          ConstOpToValueStr[op.getResult()] = std::to_string(value);
          // std::string name_c = EmitNewValueAndGetName(op.getResult(), "int");
          // _os << "int " << name_c << " = " << std::to_string(value) << ";\n";
        }
      }
      else if (isa<BoolAttr>(constattr))
      {
        BoolAttr boolattr = dyn_cast<BoolAttr>(constattr);
        bool value = boolattr.getValue();
        ConstOpToValueStr[op.getResult()] = std::to_string(value);
        // std::string name_c = EmitNewValueAndGetName(op.getResult(), "bool");
        // _os << "bool " << name_c << " = " << std::to_string(value) << ";\n";
      }
      else if (auto denseAttr = op.getValue().dyn_cast<DenseElementsAttr>())
      {
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
      }
      else
        op.emitError("has unsupported constant type.");

      // ConstOpToValueStr_print();
    }

    bool visitOp(arith::AddIOp op)
    {
      std::string type = getEmitType(op.getResult());
      std::string Lhs = _pytestemitter->lookupName(op.getLhs());
      if (Lhs == "")
        Lhs = ConstOpToValueStr[op.getLhs()];
      std::string Rhs = _pytestemitter->lookupName(op.getRhs());
      if (Rhs == "")
        Rhs = ConstOpToValueStr[op.getRhs()];
      assert(Lhs != "" && Rhs != "");

      if (Lhs == "0" && Rhs == "0")
        indent() << EmitNewValueAndGetName(op.getResult(), type)
                 << " = " << "0" << "\n";
      else if (Lhs == "0" && Rhs != "0")
        indent() << EmitNewValueAndGetName(op.getResult(), type)
                 << " = " << Rhs << "\n";
      else if (Lhs != "0" && Rhs == "0")
        indent() << EmitNewValueAndGetName(op.getResult(), type)
                 << " = " << Lhs << "\n";
      else
        indent() << EmitNewValueAndGetName(op.getResult(), type)
                 << " = " << Lhs << " " << "+" << " " << Rhs << "\n";

      return true;
      // return EmitBinary(op, "+");
    }

    bool visitOp(arith::SubIOp op)
    {
      std::string type = getEmitType(op.getResult());
      std::string Lhs = _pytestemitter->lookupName(op.getLhs());
      if (Lhs == "")
        Lhs = ConstOpToValueStr[op.getLhs()];
      std::string Rhs = _pytestemitter->lookupName(op.getRhs());
      if (Rhs == "")
        Rhs = ConstOpToValueStr[op.getRhs()];
      assert(Lhs != "" && Rhs != "");

      if (Lhs == "0" && Rhs == "0")
        indent() << EmitNewValueAndGetName(op.getResult(), type)
                 << " = " << "0" << "\n";
      else if (Lhs != "0" && Rhs == "0")
        indent() << EmitNewValueAndGetName(op.getResult(), type)
                 << " = " << Lhs << "\n";
      else
        indent() << EmitNewValueAndGetName(op.getResult(), type)
                 << " = " << Lhs << " " << "-" << " " << Rhs << "\n";

      return true;

      // return EmitBinary(op, "-");
    }

    bool visitOp(arith::MulIOp op)
    {
      std::string type = getEmitType(op.getResult());
      std::string Lhs = _pytestemitter->lookupName(op.getLhs());
      if (Lhs == "")
        Lhs = ConstOpToValueStr[op.getLhs()];
      std::string Rhs = _pytestemitter->lookupName(op.getRhs());
      if (Rhs == "")
        Rhs = ConstOpToValueStr[op.getRhs()];
      assert(Lhs != "" && Rhs != "");

      if (Lhs == "0" || Rhs == "0")
        indent() << EmitNewValueAndGetName(op.getResult(), type) << " = 0\n";
      else if (Lhs == "1")
        indent() << EmitNewValueAndGetName(op.getResult(), type) << " = " << Rhs << "\n";
      else if (Rhs == "1")
        indent() << EmitNewValueAndGetName(op.getResult(), type) << " = " << Lhs << "\n";
      else
        indent() << EmitNewValueAndGetName(op.getResult(), type)
                 << " = " << Lhs << " * " << Rhs << "\n";

      return true;

      // return EmitBinary(op, "*");
    }

    bool visitOp(arith::DivSIOp op) { return EmitBinary(op, "//"); }
    bool visitOp(arith::RemSIOp op) { return EmitBinary(op, "%"); }

    bool visitOp(arith::DivUIOp op) { return EmitBinary(op, "//"); }
    bool visitOp(arith::ShRSIOp op) { return EmitBinary(op, ">>"); }
    bool visitOp(arith::ShLIOp  op) { return EmitBinary(op, "<<"); }
    bool visitOp(arith::RemUIOp op) { return EmitBinary(op, "%"); }

    bool visitOp(arith::CmpIOp op)
    {
      std::string type = "bool"; 
      std::string Lhs = _pytestemitter->lookupName(op.getLhs());
      if (Lhs == "")
        Lhs = ConstOpToValueStr[op.getLhs()];
      std::string Rhs = _pytestemitter->lookupName(op.getRhs());
      if (Rhs == "")
        Rhs = ConstOpToValueStr[op.getRhs()];
      assert(Lhs != "" && Rhs != "");

      std::string pred = " == ";
      auto predicate = op.getPredicate();
      if (predicate == arith::CmpIPredicate::slt || predicate == arith::CmpIPredicate::ult)
        pred = " < ";
      else if (predicate == arith::CmpIPredicate::sle || predicate == arith::CmpIPredicate::ule)
        pred = " <= ";
      else if (predicate == arith::CmpIPredicate::sgt || predicate == arith::CmpIPredicate::ugt)
        pred = " > ";
      else if (predicate == arith::CmpIPredicate::sge || predicate == arith::CmpIPredicate::uge)
        pred = " >= ";
      else if (predicate == arith::CmpIPredicate::eq)
        pred = " == ";
      else if (predicate == arith::CmpIPredicate::ne)
        pred = " != ";

      indent() << EmitNewValueAndGetName(op.getResult(), type)
               << " = " << Lhs << pred << Rhs << "\n";
      return true;
    }

    bool visitOp(arith::SelectOp op)
    {
      std::string type = getEmitType(op.getResult());

      std::string cond = _pytestemitter->lookupName(op.getCondition());
      if (cond == "")
        cond = ConstOpToValueStr[op.getCondition()];

      std::string trueVal = _pytestemitter->lookupName(op.getTrueValue());
      if (trueVal == "")
        trueVal = ConstOpToValueStr[op.getTrueValue()];

      std::string falseVal = _pytestemitter->lookupName(op.getFalseValue());
      if (falseVal == "")
        falseVal = ConstOpToValueStr[op.getFalseValue()];

      assert(cond != "" && trueVal != "" && falseVal != "");

      // res = trueVal if cond else falseVal
      indent() << EmitNewValueAndGetName(op.getResult(), type)
               << " = " << trueVal << " if " << cond << " else " << falseVal << "\n";
      return true;
    }

    bool visitOp(LLVM::UndefOp op)
    {
      // op.getRes()
      // return EmitBinary(op, "/");
      return true; /// skip
    }

    void ConstOpToValueStr_print()
    {
      llvm::errs() << "=======Print ConstOpToValueStr=======\n";
      for (auto elem : ConstOpToValueStr)
      {
        llvm::errs() << elem.first << "  --->  " << elem.second << "\n";
      }
      llvm::errs() << "=======End Print=======\n";
    }

  private:
    PytestEmitter *_pytestemitter;
    llvm::DenseMap<mlir::Value, std::string> ConstOpToValueStr;
    llvm::raw_ostream &_os;
    unsigned _indent = 0;
  };
  PyOpEmitter *opEmitter;
} // namespace

// static InFlightDiagnostic emitError(Operation *op, const Twine &message) {
//   // state.encounteredError = true;
//   return op->emitError(message);
// }
//===----------------------------------------------------------------------===//
// Members of PythonEmitter class
//===----------------------------------------------------------------------===//
/// @brief Emit the header of a function including function name, function args...
/// @param os
void PytestEmitter::emitFunctionHead(func::FuncOp &funcop, llvm::raw_ostream &os)
{
  std::stringstream ostr;
  ostr << "async def " << funcop.getSymName().str() << "("
       << "runtime: DeviceRuntime, ";

  // Funtion args
  ArrayRef<mlir::Type> argTypes = funcop.getArgumentTypes();
  for (int argIdx = 0; argIdx < argTypes.size(); argIdx++)
  {
    if (argIdx != 0)
    {
      ostr << ", ";
    }
    mlir::Type argType = argTypes[argIdx];

    // argType.dump();
    Op_Name_C arg_info("arg", argIdx);
    appendValueNameList(funcop.getBody().getArgument(argIdx), arg_info);
    if (argType.isa<MemRefType>())
    {
      ostr << "arg_" << argIdx << ": ndarray";
    }
    else if (argType.isIndex())
    {
      ostr << "arg_" << argIdx << ": int";
    }
    else if (argType.isInteger(32))
    {
      ostr << "arg_" << argIdx << ": int";
    }
    else if (argType.isInteger(64))
    {
      ostr << "arg_" << argIdx << ": int";
    }
    else if (argType.isUnsignedInteger(32))
    {
      ostr << "arg_" << argIdx << ": int";
    }
    else if (argType.isUnsignedInteger(64))
    {
      ostr << "arg_" << argIdx << ": int";
    }
  }
  ostr << "):\n";

  ostr << "    # runtime.log.info(\"[ADORA] Starting CGRA call ("
       << funcop.getSymName().str() << ")\")\n";

  ostr << "    iptrs, idata = [],[]\n"
       << "    optrs, odata, olen = [],[],[]\n"
       << "    configs, data_ptr = [],[]\n"
       << "    writeback_tasks = []\n"
       << "    stream = runtime.create_stream()\n";

  os << ostr.str();
}

/// @brief Emit a block (maybe a loop body, maybe a function body), especially the for loop structure
/// @param os
void PytestEmitter::emitBlock(mlir::Block &block, llvm::raw_ostream &os)
{
  // std::stringstream ostr;
  addIndent();

  opEmitter->setIndent(getIndent());
  opEmitter->computeStreamIds(block);  // graph-colour SSA token edges → _streamId
  block.dump();

  for (auto &op : block)
  {
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
    if (opEmitter->dispatchVisitor(&op))
    {
      continue;
    }
    else
    {
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
bool PytestEmitter::emitPytest(llvm::raw_ostream &os)
{
  opEmitter = new PyOpEmitter(*this, os);
  // ADORAEmitterState state(os);
  // ModuleEmitter(state).emitModule(module);
  // return failure(state.encounteredError);
  /// get time
  std::time_t t = std::time(nullptr);
  std::tm tm;
#ifdef _WIN32
  localtime_s(&tm, &t);
#else
  localtime_r(&t, &tm);
#endif
  char timebuf[64];
  std::strftime(timebuf, sizeof(timebuf), "%Y-%m-%d %H:%M:%S", &tm);

  os << R"XXX(
"""
Copyright (c) 2025 ADORA
All rights reserved.
Automatically generated file for pytest/cocotb based CGRA call function from ADORA.
)XXX";
  os << "Generated on: " << timebuf << "\n";
  os << R"XXX(
"""
from test_runif import DeviceData, DeviceConfig, DeviceStream, DeviceRuntime
from typing import List
from numpy import ndarray
import numpy as np
import torch
import torch.nn.functional as F

def safe_slice_1d(arr, flat_offset, result_shape, *args, **kwargs):
    if not hasattr(safe_slice_1d, "cache"):
        safe_slice_1d.cache = {}
        safe_slice_1d.meta = {'R': 3, 'S': 3, 'stride': 1}
    _agu_cache = safe_slice_1d.cache
    _agu_meta = safe_slice_1d.meta
    arr_id = id(arr)

    if arr.ndim == 4 and len(result_shape) == 2:
        dim0, dim1, dim2, dim3 = arr.shape
        if dim2 <= 11 and dim3 <= 11 and dim2 == dim3:
            K_out, C, R, S = arr.shape
            _agu_meta['R'], _agu_meta['S'] = R, S
            if arr_id not in _agu_cache:
                t = torch.tensor(arr.view(np.int16)).view(torch.bfloat16).float()
                unf = t.view(K_out, C * R * S).transpose(0, 1).contiguous()
                _agu_cache[arr_id] = unf.to(torch.bfloat16).view(torch.int16).numpy().view(np.float16)
            unfolded_B = _agu_cache[arr_id]
            K_gemm = C * R * S
            row_start = flat_offset % K_gemm
            col_start = flat_offset // K_gemm
            r_len, c_len = result_shape
            res = np.zeros(result_shape, dtype=arr.dtype)
            valid_r = min(r_len, unfolded_B.shape[0] - row_start)
            valid_c = min(c_len, unfolded_B.shape[1] - col_start)
            res[:valid_r, :valid_c] = unfolded_B[row_start:row_start+valid_r, col_start:col_start+valid_c]
            return res

        elif dim1 > 1 and (result_shape[1] == dim1 or result_shape[1] == 16):
            N_dim, K_out, P, Q = arr.shape
            M_gemm = N_dim * P * Q
            N_gemm = K_out
            if arr_id not in _agu_cache:
                t = torch.tensor(arr.view(np.int16)).view(torch.bfloat16).float()
                t_reshaped = t.permute(0, 2, 3, 1).reshape(M_gemm, N_gemm).contiguous()
                _agu_cache[arr_id] = t_reshaped.to(torch.bfloat16).view(torch.int16).numpy().view(np.float16)
            unfolded_C = _agu_cache[arr_id]
            k_out = (flat_offset // (P * Q)) % K_out
            q = flat_offset % Q
            p = (flat_offset // Q) % P
            n_dim = flat_offset // (K_out * P * Q)
            m_start = n_dim * P * Q + p * Q + q
            n_start = k_out
            m_len, n_len = result_shape
            res = np.zeros(result_shape, dtype=arr.dtype)
            valid_m = min(m_len, M_gemm - m_start)
            valid_n = min(n_len, N_gemm - n_start)
            res[:valid_m, :valid_n] = unfolded_C[m_start:m_start+valid_m, n_start:n_start+valid_n]
            return res

        else:
            N, C, H, W = arr.shape
            R, S = _agu_meta['R'], _agu_meta['S']
            stride = _agu_meta.get('stride', 1)
            P = (H - R) // stride + 1
            Q = (W - S) // stride + 1
            M_gemm = N * P * Q
            K_gemm = C * R * S
            if arr_id not in _agu_cache:
                t = torch.tensor(arr.view(np.int16)).view(torch.bfloat16).float()
                unf = F.unfold(t, kernel_size=(R, S), padding=0, stride=stride)
                unf = unf.transpose(1, 2).reshape(M_gemm, K_gemm).contiguous()
                _agu_cache[arr_id] = unf.to(torch.bfloat16).view(torch.int16).numpy().view(np.float16)
            unfolded_A = _agu_cache[arr_id]
            matches = []
            for m in range(M_gemm):
                for k in range(K_gemm):
                    n = m // (P * Q)
                    p = (m % (P * Q)) // Q
                    q = m % Q
                    c = k // (R * S)
                    r = (k % (R * S)) // S
                    s = k % S
                    h_in = p * stride + r
                    w_in = q * stride + s
                    if 0 <= h_in < H and 0 <= w_in < W:
                        offset = n * (C * H * W) + c * (H * W) + h_in * W + w_in
                        if offset == flat_offset:
                            matches.append((m, k))
            m_start, k_start = matches[0] if matches else (0, 0)
            for (m, k) in matches:
                if k % result_shape[1] == 0:
                    m_start, k_start = m, k
                    break
            m_len, k_len = result_shape
            res = np.zeros(result_shape, dtype=arr.dtype)
            valid_m = min(m_len, M_gemm - m_start)
            valid_k = min(k_len, K_gemm - k_start)
            if valid_m > 0 and valid_k > 0:
                res[:valid_m, :valid_k] = unfolded_A[m_start:m_start+valid_m, k_start:k_start+valid_k]
            return res

    size = arr.size
    length = int(np.prod(result_shape))
    if flat_offset >= size: return np.zeros(result_shape, dtype=arr.dtype)
    valid_len = min(length, size - flat_offset)
    res = np.zeros(length, dtype=arr.dtype)
    res[:valid_len] = arr.flat[flat_offset : flat_offset + valid_len]
    return res.reshape(result_shape)

def apply_writeback_tasks(tasks):
    for arr, offsets, data_block in tasks:
        flat_offset = 0
        stride = 1
        for idx in range(len(offsets)-1, -1, -1):
            flat_offset += offsets[idx] * stride
            stride *= arr.shape[idx]

        # 使用 uint8 进行纯物理比特位的非零统计
        non_zeros = np.count_nonzero(data_block.view(np.uint8))
        print(f"[DEBUG 探针] 写回拼装: 物理偏移 {flat_offset} | 提取到有效非零字节: {non_zeros}/{data_block.nbytes}")

        # 如果是直接卷积的 3D data_block (比如 4x9x4)，直接走下面的 else 物理空间展平写回！
        if arr.ndim == 4 and data_block.ndim == 2:
            N_dim, K_out, P, Q = arr.shape
            m_len, n_len = data_block.shape
            k_out_start = (flat_offset // (P * Q)) % K_out
            q_start = flat_offset % Q
            p_start = (flat_offset // Q) % P
            n_start = flat_offset // (K_out * P * Q)
            m_start = n_start * P * Q + p_start * Q + q_start

            for m in range(m_len):
                curr_m = m_start + m
                if curr_m >= N_dim * P * Q: break
                n_idx = curr_m // (P * Q)
                p_idx = (curr_m % (P * Q)) // Q
                q_idx = curr_m % Q
                valid_n = min(n_len, K_out - k_out_start)
                arr[n_idx, k_out_start:k_out_start+valid_n, p_idx, q_idx] = data_block[m, :valid_n]
        else:
            flat_data = data_block.ravel()
            valid_len = min(flat_data.size, arr.size - flat_offset)
            if valid_len > 0:
                arr.flat[flat_offset : flat_offset + valid_len] = flat_data[:valid_len]

async def aux_stream(
    stream: DeviceStream, config: List[DeviceConfig], 
    iptrs: List[DeviceData], idata: List, 
    optrs: List[DeviceData], odata: List, olen: List):
    """
    Execute a device stream workflow.

    Parameters
    ----------
    stream : DeviceStream
        The device stream instance to operate on.
    config : List[DeviceConfig]
        Configuration objects to apply before execution.
    iptrs : List[DeviceData]
        Device pointers for input buffers.
    idata : List
        Host-side input data corresponding to `iptrs`.
    optrs : List[DeviceData]
        Device pointers for output buffers.
    odata : List
        Host-side output data containers corresponding to `optrs`.
    olen : List[int]
        Expected output lengths for each output buffer.
    """
    # ------------------------------
    # 1. Apply stream configuration
    # ------------------------------
    await stream.apply(config)
    await stream.config(config_id=0)
    # ------------------------------
    # 2. Host -> Device transfer
    # ------------------------------
    for i in range(len(iptrs)):
        await stream.memcpyHostToDevice(d_data=iptrs[i], h_data=idata[i], size=len(idata[i]))
    # ------------------------------
    # 3. Execute on device
    # ------------------------------
    await stream.execution_start()
    # await stream.execution_finish()
    # ------------------------------
    # 4. Device → Host transfer
    # ------------------------------
    for i in range(len(optrs)):
        await stream.memcpyDeviceToHost(d_data=optrs[i], h_data=odata[i], size=olen[i])

    await stream.release()
    return

def DeviceData_Pong(ptr : DeviceData) -> DeviceData:
    new_ptr = DeviceData(ptr.address+ptr.size, ptr.size)
    return new_ptr
  
async def aux_stream_pingpong(
    stream: DeviceStream, 
    # config: List[DeviceConfig], 
    config_id:int,
    iptrs: List[DeviceData], idata: List[ndarray], 
    optrs: List[DeviceData], odata: List, olen: List, 
    pingpong: bool):
    """
    Execute a device stream workflow.

    Parameters
    ----------
    stream : DeviceStream
        The device stream instance to operate on.
    config : List[DeviceConfig]
        Configuration objects to apply before execution.
    iptrs : List[DeviceData]
        Device pointers for input buffers.
    idata : List
        Host-side input data corresponding to `iptrs`.
    optrs : List[DeviceData]
        Device pointers for output buffers.
    odata : List
        Host-side output data containers corresponding to `optrs`.
    olen : List[int]
        Expected output lengths for each output buffer.
    pingpong : bool
        Indicates the pingpong phase(ping-phase or pong-phase)
    """
    # ------------------------------
    # 1. Apply stream configuration
    # ------------------------------     
    await stream.config(config_id=config_id)
    
    # ------------------------------
    # 2. Host -> Device transfer
    #   depend_type:
    #   2 -> depends on the second previous task (no need to wait for store-back)
    #   1 -> depends on the immediately previous task (no need to wait for store-back)
    #   0 -> strictly sequential execution
    # ------------------------------
    for i in range(len(iptrs)):
        if(pingpong == 0):
            await stream.memcpyHostToDevice(d_data=iptrs[i], h_data=idata[i], size=len(idata[i]), depend_type=2)
        else:
            await stream.memcpyHostToDevice(DeviceData_Pong(iptrs[i]), h_data=idata[i], size=len(idata[i]), depend_type=2)

    # ------------------------------
    # 3. Execute on device
    # ------------------------------
    await stream.execution_start()
    # await stream.execution_finish()

    # ------------------------------
    # 4. Device → Host transfer
    # ------------------------------
    for i in range(len(optrs)):
        if(pingpong == 0):
            await stream.memcpyDeviceToHost(d_data=optrs[i], h_data=odata[i], size=olen[i])
        else :
            await stream.memcpyDeviceToHost(DeviceData_Pong(optrs[i]), h_data=odata[i], size=olen[i])
    
    # await stream.synchronize()
    # await stream.release()
    return

async def aux_stream_pingpong_init(
    stream: DeviceStream, config: List[DeviceConfig]
    ):
    """
    Apply stream configuration
    """
    cfg_copy = list(config)
    await stream.apply(cfg_copy)  
    await stream.config(config_id=0)
    
    # await stream.release()
    return

## ===----------------------------------------------------------------------===//
## Configuration Data 
## ===----------------------------------------------------------------------===//
)XXX";

  for (auto elem : KnToCfgData)
  {
    auto key = elem.first;
    os << elem.second << "\n";
    if (KnToPingpongCfgData.count(key))
    {
      os << KnToPingpongCfgData[key].first << "\n";
      os << KnToPingpongCfgData[key].second << "\n";
    }
  }

  // _moduleop.walk([&](mlir::Operation* op) {
  //   op->dump();
  // });
  /// Emit module
  for (auto funcop : _moduleop.getOps<func::FuncOp>())
  {
    funcop.dump();
    /// function head
    emitFunctionHead(funcop, os);

    /// function body
    // addIndent();
    emitBlock(funcop.getBody().front(), os);

    // / function tail
    //     os << R"XXX(
    //     await stream.synchronize()
    // )XXX";
  }

  delete opEmitter;
}

bool PytestEmitter::emitCGRACallFunction(llvm::raw_ostream &os)
{
  emitPytest(os);
}

// Traverse the whole module to find which operation keeps a "VAR_CONFIG" attr
// equal to arg config, and return the result value of the operation
std::string PytestEmitter::lookupVarConfigName(const std::string config)
{
  mlir::Value target_value;
  _moduleop.walk([&](mlir::Operation *op) -> WalkResult
                 {
    op->dump();
    if(op->hasAttr("VAR_CONFIG") 
        && config == dyn_cast<StringAttr>(op->getAttr("VAR_CONFIG")).str()){
      /// find the value crresponding to the config
      assert(op->getResults().size() == 1);
      target_value = op->getResult(0);
      return WalkResult::interrupt();
    }
    return WalkResult::advance(); });
  std::string target_name = lookupName(target_value);
  assert(target_name != "");
  return target_name;
}

/// @brief Get the config data
std::string PytestEmitter::GenerateCGRAConfig(
    ADORA::KernelOp &kernel, Configuration cfg, ADG *adg)
{
  // adg->print();
  std::string CFGarrayName = "cfgbit_" + kernel.getKernelName();
  std::stringstream CFGdata, pingCFG, pongCFG;

  // cfg.dumpCfgData(std::cout);
  std::map<int, dfgIoInfo> dfg_io_infos = std::move(_kernel_to_dfg_io_infos[kernel]);
  for (auto &elem : dfg_io_infos)
  {
    cfg.setDfgIoSpadAddr(elem.first, elem.second.iobAddr);
  }
  std::vector<CfgDataPacket> cfgData;
  cfg.getCfgData(cfgData);

  std::vector<CfgDataPacket> cfgPingData;
  std::vector<CfgDataPacket> cfgPongData;
  cfg.getPingpongCfgData(cfgPingData, cfgPongData);

  // cfg.getCfgData(cfgData); /// debug
  int cfgSpadDataByte = adg->cfgSpadDataWidth() / 8;
  int cfgAddrWidth = adg->cfgAddrWidth();
  int cfgDataWidth = adg->cfgDataWidth();
  int alignWidth = (cfgAddrWidth > 16) ? 32 : 16;
  assert(alignWidth >= cfgAddrWidth && cfgDataWidth >= alignWidth);
  int cfgNum = 0;
  for (auto &cdp : cfgData)
  {
    cfgNum += cdp.data.size() * 32 / cfgDataWidth;
  }

  // if(cfgAddrWidth > 16){
  //   CFGdata << "volatile unsigned int ";
  // }else{
  //   CFGdata << "volatile unsigned short ";
  // }

  //// Get initial array config
  CFGdata << "\"\"\" kernel: " << kernel.getKernelName()
          << ",  cfgNum: " << cfgNum << "\"\"\"\n";
  CFGdata << CFGarrayName << " = [\n";
  CFGdata << std::hex;
  int alignWidthHex = alignWidth / 4;
  for (auto &cdp : cfgData)
  {
    CFGdata << "\t\t";
    for (auto data : cdp.data)
    {
      if (alignWidth == 32)
      {
        CFGdata << "0x" << std::setw(alignWidthHex) << std::setfill('0') << data << ", ";
      }
      else
      {
        CFGdata << "0x" << std::setw(alignWidthHex) << std::setfill('0') << (data & 0xffff) << ", ";
        CFGdata << "0x" << std::setw(alignWidthHex) << std::setfill('0') << (data >> 16) << ", ";
      }
    }
    CFGdata << "0x" << std::setw(alignWidthHex) << std::setfill('0') << (cdp.addr) << ",\n";
  }
  CFGdata << std::dec << "\t]\n\n";

  //// Get ping-phase config
  //// Get initial array config
  if (cfgPingData.size() != 0 && cfgPongData.size() != 0)
  {
    std::string pingCFGarrayName = CFGarrayName + "_ping";
    pingCFG << "\"\"\" kernel: " << kernel.getKernelName()
            << ", ping-phase" << "\"\"\"\n";
    pingCFG << pingCFGarrayName << " = [\n";
    pingCFG << std::hex;
    for (auto &cdp : cfgPingData)
    {
      pingCFG << "\t\t";
      for (auto data : cdp.data)
      {
        if (alignWidth == 32)
        {
          pingCFG << "0x" << std::setw(alignWidthHex) << std::setfill('0') << data << ", ";
        }
        else
        {
          pingCFG << "0x" << std::setw(alignWidthHex) << std::setfill('0') << (data & 0xffff) << ", ";
          pingCFG << "0x" << std::setw(alignWidthHex) << std::setfill('0') << (data >> 16) << ", ";
        }
      }
      pingCFG << "0x" << std::setw(alignWidthHex) << std::setfill('0') << (cdp.addr) << ",\n";
    }
    pingCFG << std::dec << "\t]\n\n";

    //// Get pong-phase config
    //// Get initial array config
    std::string pongCFGarrayName = CFGarrayName + "_pong";
    pingCFG << "\"\"\" kernel: " << kernel.getKernelName()
            << ", pong-phase" << "\"\"\"\n";
    pongCFG << pongCFGarrayName << " = [\n";
    pongCFG << std::hex;
    for (auto &cdp : cfgPongData)
    {
      pongCFG << "\t\t";
      for (auto data : cdp.data)
      {
        if (alignWidth == 32)
        {
          pongCFG << "0x" << std::setw(alignWidthHex) << std::setfill('0') << data << ", ";
        }
        else
        {
          pongCFG << "0x" << std::setw(alignWidthHex) << std::setfill('0') << (data & 0xffff) << ", ";
          pongCFG << "0x" << std::setw(alignWidthHex) << std::setfill('0') << (data >> 16) << ", ";
        }
      }
      pongCFG << "0x" << std::setw(alignWidthHex) << std::setfill('0') << (cdp.addr) << ",\n";
    }
    pongCFG << std::dec << "\t]\n\n";

    KnToPingpongCfgData[kernel] = std::make_pair(pingCFG.str(), pongCFG.str());
  }

  KnToCfgData[kernel] = CFGdata.str();
  KnToCfgArrayInfo[kernel] = std::pair(CFGarrayName, cfgNum);

  return CFGdata.str();
}

/// @brief Get the config data
std::string PytestEmitter::GenerateCGRAConfig(
    ADORA::KernelOp &kernel, MapperSA *mapper)
{
  ADG *adg = mapper->getADG();
  Configuration cfg(mapper->_mapping);
  KnToConfiguration[kernel] = cfg;
  _adg = adg;
  return GenerateCGRAConfig(kernel, cfg, adg);
}

/// @brief Get the config and execution instructions of CGRA
void PytestEmitter::GenerateCGRACFGAndEXE(
    ADORA::KernelOp &kernel, Configuration cfg, ADG *adg)
{
  // adg->print();
  /////////////////////// CFG is generated in GenerateCGRACFGData()
  std::stringstream CFGandEXE;

  int cfgSpadDataByte = adg->cfgSpadDataWidth() / 8;
  int cfgAddrWidth = adg->cfgAddrWidth();
  int cfgDataWidth = adg->cfgDataWidth();
  int alignWidth = (cfgAddrWidth > 16) ? 32 : 16;
  assert(alignWidth >= cfgAddrWidth && cfgDataWidth >= alignWidth);

  if (!MapHasKey(KnToCfgData, kernel))
  {
    std::string cfgdata = GenerateCGRAConfig(kernel, cfg, adg);
    CFGandEXE << cfgdata << "\n";
  }
  assert(MapHasKey(KnToCfgData, kernel));
  std::string CFGarrayName = KnToCfgArrayInfo[kernel].first;
  int cfgNum = KnToCfgArrayInfo[kernel].second;

  /// replace the variable part of the config
  //// TODO: what about variable runtime
  CFGandEXE << std::hex;
  for (auto elem : cfg.VarReplaceInfo)
  {
    std::string varcfg_name = lookupVarConfigName(elem.first);
    for (auto &replace : elem.second)
    {
      CFGandEXE << std::dec << CFGarrayName << "[" << replace.Idx0 << "][" << replace.Idx1 << "] = ";
      assert(replace.lshift == 0 || replace.rshift == 0);
      if (replace.lshift == 0)
      {
        // right shift
        CFGandEXE << std::hex << "(" << varcfg_name << " >> 0x" << replace.rshift << ")"
                  << " | "
                  << std::dec << "(" << CFGarrayName << "[" << replace.Idx0 << "][" << replace.Idx1 << "]"
                  << std::hex << " & 0x" << replace.getMask() << ")\n";
      }
      else
      {
        // left shift
        CFGandEXE << std::hex << "(" << varcfg_name << " << 0x" << replace.lshift << ")"
                  << " | "
                  << std::dec << "(" << CFGarrayName << "[" << replace.Idx0 << "][" << replace.Idx1 << "]"
                  << std::hex << " & 0x" << replace.getMask() << ")\n";
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
  int cfgBaseAddrCtrl = cfgBaseAddr / cfgSpadDataByte;    // config base address the controller access

  BYTES_LIST iob_ens = _kernel_to_iob_ens[kernel];
  BYTES_LIST tile_ens = _kernel_to_tile_ens[kernel];

  CFGandEXE << "data_ptr.append(iptrs)\n";
  // CFGandEXE << "data_ptr.append(optrs)\n\n";

  CFGandEXE << "config_" << kernel.getKernelName() << "= DeviceConfig(\n";
  CFGandEXE << "\tconfig_values=" << CFGarrayName << ",\n";
  CFGandEXE << "\tiob_en=[";
  for (int _ = 0; _ < iob_ens.size(); _++)
  {
    CFGandEXE << iob_ens.getByte(_);
    if (_ != iob_ens.size() - 1)
      CFGandEXE << ",";
  }
  CFGandEXE << "],\n";

  CFGandEXE << "\ttile_en=[";
  for (int _ = 0; _ < tile_ens.size(); _++)
  {
    CFGandEXE << tile_ens.getByte(_);
    if (_ != tile_ens.size() - 1)
      CFGandEXE << ",";
  }
  CFGandEXE << "],\n";

  CFGandEXE << "\tdata_ptr=data_ptr\n";
  CFGandEXE << ")\n";

  CFGandEXE << "configs.append(config_" << kernel.getKernelName() << ")";

  // CFGandEXE << "\tload_cfg((void*)" << CFGarrayName << ", 0x" << std::hex << cfgBaseAddrSpad << std::dec << ", "
  //      << cfg_len << ", " << /*_task_id=*/"_task_id" << ", " << /*_ld_cfg_dep*/"LD_DEP_EX_LAST_TASK" << ");\n";
  // CFGandEXE << "config(0x" << std::hex << cfgBaseAddrCtrl << std::dec << ", " << cfgNum << ", " << /*_task_id*/"_task_id" << ", " << /*_ex_dep*/ 0 << ");\n";
  // CFGandEXE << "execute(0x" << std::hex << iob_ens << std::dec << ", " << /*_task_id*/"_task_id" << ", " << /*_ex_dep*/"EX_DEP_ST_LAST_TASK" << ");\n";

  KnToCfgExe[kernel] = CFGandEXE.str();

  // std::cout << CFGandEXE.str() << std::endl;
}

/// @brief Get the config and execution instructions of CGRA
/// @param mapper The SA mapper which has completed mapping
void PytestEmitter::GenerateCGRACFGAndEXE(ADORA::KernelOp &kernel, MapperSA *mapper)
{
  /// why MapperSA (without&) will cause bug ?? memory leakage?
  /// Generate CGRA configuration
  ADG *adg = mapper->getADG();
  Configuration cfg(mapper->_mapping);
  KnToConfiguration[kernel] = cfg;
  _adg = adg;
  GenerateCGRACFGAndEXE(kernel, cfg, adg);
}

/////////////////////////
/// emit Gemm block
/////////////////////////
/// @brief Emit a block nested in "for" op for adora Gemm
/// @param os
void PytestEmitter::emitGemmBlock(mlir::Block &block, llvm::raw_ostream &os)
{
  // std::stringstream ostr;
  addIndent();

  opEmitter->setIndent(getIndent());
  block.dump();

  for (auto &op : block)
  {
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
    if (opEmitter->dispatchVisitor(&op))
    {
      continue;
    }
    else
    {
      op.emitError("can't be correctly emitted.");
    }
  }
  reduceIndent();
  opEmitter->setIndent(getIndent());
  // os << ostr.str();
}