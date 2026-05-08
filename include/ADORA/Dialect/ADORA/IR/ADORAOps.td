//===-- ADORAOps.td - ADORA dialect operation definitions ------*- tablegen -*-===//
//===----------------------------------------------------------------------===//
//
// Defines some operations of the ADORA dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ADORA_OPS
#define ADORA_OPS

include "ADORA/Dialect/ADORA/IR/ADORABase.td"

include "mlir/Pass/PassBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

include "mlir/Dialect/Linalg/IR/LinalgBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Dialect/Arith/IR/ArithBase.td"
include "mlir/Dialect/Arith/IR/ArithOpsInterfaces.td"

// include "mlir/Dialect/Affine/IR/AffineOps.td"

// include "mlir/Dialect/Func/IR/FuncOps.td"
// include "mlir/Dialect/Func/IR/FuncOps.h"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

// include "mlir/Dialect/MemRef/IR/MemRefOps.td"
// #include "mlir/Dialect/Func/IR/FuncOps.h"


//===----------------------------------------------------------------------===//
// Explicit Data Block Movement Operation (Load and Store)
//===----------------------------------------------------------------------===//

def ADORA_DataBlockLoadOp: ADORA_Op<"BlockLoad", [AttrSizedOperandSegments]>{
  let summary = "Load a part of the whole array data block to a faster memory where HW accelerator can access.";
  let description = [{
    This operation takes in an array memref,a header index of the data block which will be moved, and the block size.
    Returns the new memref from the buffer given certain header index and size.
    This operation will corporate with BlockStore and Call of KernelFunc. 

    Example: To fix !!!
    ```mlir
     %0 = ADORA.BlockLoad %arg1 [%arg9, 0] : memref<32x32xf32> -> memref<1x32xf32> {three_mm_32_kernel_0} 
     %1 = ADORA.BlockLoad %arg2 [0, %arg10] : memref<32x32xf32> -> memref<32x16xf32> {three_mm_32_kernel_0}
    ```
  }];
  
  let builders = [   
    /// Builds an affine load op with the specified map and operands.
    // OpBuilder<(ins "AffineMap":$map, "ValueRange":$operands, "unsigned":$BlockLen)>,
    /// Builds an affine load op with an identity map and operands.
    // OpBuilder<(ins "Value":$DataBlock, "ValueRange":$operands, "unsigned":$BlockLen)>,
    /// Builds a block load op with the specified map and its operands.
    OpBuilder<(ins "Value":$DataBlock, "AffineMap":$map, "ValueRange":$operands, 
                   "MemRefType":$resultType , "std::string":$KernelName )>,

    //// with strides
    OpBuilder<(ins "Value":$DataBlock, "AffineMap":$map, "ValueRange":$operands, 
                   "MemRefType":$resultType, "DenseI64ArrayAttr":$strides, "std::string":$KernelName )>,
    OpBuilder<(ins "Value":$DataBlock, "AffineMap":$map, "ValueRange":$operands, 
                   "MemRefType":$resultType, "ArrayRef<int64_t>":$strides, "std::string":$KernelName )>,
    // OpBuilder<(ins "Value":$DataBlock, "AffineMap":$map, "ValueRange":$operands, 
    //                 "int64_t":$BlockLen, "std::string":$KernelName )>,
    // /// Builds an block load op without KernelName.
    OpBuilder<(ins "Value":$DataBlock, "AffineMap":$map, "ValueRange":$operands, 
                   "MemRefType":$resultType)>,

    //// with strides
    OpBuilder<(ins "Value":$DataBlock, "AffineMap":$map, "ValueRange":$operands, 
                   "MemRefType":$resultType, "DenseI64ArrayAttr":$strides)>,
    OpBuilder<(ins "Value":$DataBlock, "AffineMap":$map, "ValueRange":$operands, 
                   "MemRefType":$resultType, "ArrayRef<int64_t>":$strides)>,

    //// async form (strides may be null, KernelName may be empty)
    OpBuilder<(ins "Value":$DataBlock, "AffineMap":$map, "ValueRange":$operands,
                   "MemRefType":$resultType, "DenseI64ArrayAttr":$strides,
                   "std::string":$KernelName,
                   "ValueRange":$asyncDependencies, "bool":$produceToken)>,
  ];
  
	//Ignore Variadic<Index>:$offset at this moment
  let arguments = (ins Arg<AnyMemRef, "the reference to load from", [MemRead]>:$OriginalMemref
                  , Variadic<Index>:$indices
                  , Variadic<ADORA_TokenType>:$asyncDependencies);
  
  let results = (outs AnyStaticShapeMemRef:$result,
                      Optional<ADORA_TokenType>:$asyncToken);
  let extraClassDeclaration = [{
    // unsigned int BlockLen = 0;
    // void setBlockLen(unsigned l){ BlockLen = l;};
    // unsigned int getBlockLen(){ return BlockLen;};
    // Value getOriginalMemref(){ return OriginalMemref();};

    static StringRef getMapAttrStr() { return "map"; }
    static StringRef getKernelNameAttrStr() { return "KernelName"; }
    static StringRef getIdAttrStr() { return "Id"; }
    // static StringRef getBlockLenAttrStr() { return "BlockLen"; }

    // int32_t getBlockLen() {
    //   return this->getOperation()->getAttr("BlockLen")
    //     .cast<IntegerAttr>().getValue().getSExtValue();
    // }

    bool hasKernelName(){
      return this->getOperation()->hasAttr("KernelName");
    }

    StringRef getKernelName() {
      Attribute KnNameAttr = this->getOperation()->getAttr("KernelName");
      if(KnNameAttr == nullptr)
        return "";
      else
        return KnNameAttr.cast<StringAttr>().strref();
    }
    void setKernelName(std::string s) {
      StringAttr KernelNameAttr = StringAttr::get(getOperation()->getContext(),s);
      getOperation()->setAttr(getKernelNameAttrStr(), KernelNameAttr);
    }

    SmallVector<std::string> getKernelNameAsStrVector();

    std::string addAnotherKernelName(const std::string& newKernel);

    StringRef getId() {
      return this->getOperation()->getAttr("Id")
        .cast<StringAttr>().strref();
    }

    void setId(int i) {
      setId(std::to_string(i));
    }

    void setId(std::string s) {
      StringAttr IdAttr = StringAttr::get(getOperation()->getContext(),s);
      getOperation()->setAttr(getIdAttrStr(), IdAttr);
    }

    MemRefType getOriginalMemrefType() {
      return getOriginalMemref().getType().cast<MemRefType>();
    }
    MemRefType getResultType() {
      return getResult().getType().cast<MemRefType>();
    }
    MemRefType getMemRefType() {
      return getOriginalMemrefType();
    }
    operand_range getMapOperands() {
      return getIndices();
    }
    void setMapOperands(mlir::ValueRange newoperands);

    /// Returns the affine map used to index the memref for this operation.
    AffineMap getAffineMap() { return getAffineMapAttr().getValue(); }
    AffineMapAttr getAffineMapAttr() {
      return (*this)->getAttr(getMapAttrStr()).cast<AffineMapAttr>();
    }
    void setAffineMap(AffineMap map){
      getOperation()->setAttr(getMapAttrStr(), AffineMapAttr::get(map));
    }

    /// stride
    bool hasStrides();
    bool hasStridesAttr();
    void setStrides(ArrayRef<int64_t> stride);
    void setStrides(Attribute stride);
    void setStrides(DenseI64ArrayAttr stride);
    Attribute getStrides();
    ArrayRef<int64_t> getStridesAsArrayRef();
    static StringRef getStridesAttrStr() { return "strides"; }

  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasCanonicalizer = 1; // PR2: DropUnusedAsyncToken / DedupAsyncDeps
}


def ADORA_DataBlockStoreOp: ADORA_Op<"BlockStore", [AttrSizedOperandSegments]>{
  let summary = "Store a smaller memref to a bigger memref at a specified position.";
  let description = [{
    This operation takes in a source array memref which comes from a BlockLoad Op. After the excution 
    of the kernel, this array memref is changed.
    This operation take in another target array memref and a header index of the data block pointing the place
    the source memref will be placed.
    No return value exists for BlockStoreOp.
    This operation will corporate with BlockLoad and Call of KernelFunc. 

    Example: To fix !!!
    ```mlir
    %value = "ADORA.BlockLoad" (%buffer, %j):(memref<10xf32>, index, index) -> memref<5xf32> 
    ```
  }];

  let builders = [   
    OpBuilder<(ins "Value":$SourceMemref, "Value":$TargetMemref, 
                   "AffineMap":$map, "ValueRange":$operands, 
                   "std::string":$KernelName )>,

    OpBuilder<(ins "Value":$SourceMemref, "Value":$TargetMemref, 
                   "AffineMap":$map, "ValueRange":$operands, "DenseI64ArrayAttr":$strides, 
                   "std::string":$KernelName )>,    

    OpBuilder<(ins "Value":$SourceMemref, "Value":$TargetMemref, 
                   "AffineMap":$map, "ValueRange":$operands, "ArrayRef<int64_t>":$strides, 
                   "std::string":$KernelName )>,

    // /// Builds an block load op without KernelName.
    OpBuilder<(ins "Value":$SourceMemref, "Value":$TargetMemref, 
                   "AffineMap":$map, "ValueRange":$operands)>,
    
    OpBuilder<(ins "Value":$SourceMemref, "Value":$TargetMemref, 
                   "AffineMap":$map, "ValueRange":$operands, "DenseI64ArrayAttr":$strides)>,

    OpBuilder<(ins "Value":$SourceMemref, "Value":$TargetMemref, 
                   "AffineMap":$map, "ValueRange":$operands, "ArrayRef<int64_t>":$strides)>,

    //// async form
    OpBuilder<(ins "Value":$SourceMemref, "Value":$TargetMemref,
                   "AffineMap":$map, "ValueRange":$operands,
                   "DenseI64ArrayAttr":$strides, "std::string":$KernelName,
                   "ValueRange":$asyncDependencies, "bool":$produceToken)>,
  ];

	//Ignore Variadic<Index>:$offset at this moment
  let arguments = (ins Arg<AnyMemRef, "A memref from a BlockLoad Op", [MemWrite]>:$SourceMemref
                  , Arg<AnyMemRef, "the reference to store to", [MemWrite]>:$TargetMemref
                  , Variadic<Index>:$indices
                  , Variadic<ADORA_TokenType>:$asyncDependencies);
  let results = (outs Optional<ADORA_TokenType>:$asyncToken);
  let extraClassDeclaration = [{
    // unsigned int BlockLen = 0;
    // void setBlockLen(unsigned l){ BlockLen = l;};
    // unsigned int getBlockLen(){ return BlockLen;};
    // Value getSourceMemref(){ return SourceMemref();};    
    // Value getTargetMemref(){ return TargetMemref();};

    static StringRef getMapAttrStr() { return "map"; }
    static StringRef getKernelNameAttrStr() { return "KernelName"; }
    static StringRef getIdAttrStr() { return "Id"; }
    // static StringRef getBlockLenAttrStr() { return "BlockLen"; }

    // int32_t getBlockLen() {
    //   return this->getOperation()->getAttr("BlockLen")
    //     .cast<IntegerAttr>().getValue().getSExtValue();
    // }

    bool hasKernelName(){
      return this->getOperation()->hasAttr("KernelName");
    }

    StringRef getKernelName() {
      Attribute KnNameAttr = this->getOperation()->getAttr("KernelName");
      if(KnNameAttr == nullptr)
        return "";
      else
        return KnNameAttr.cast<StringAttr>().strref();
    }
    
    void setKernelName(std::string s) {
      StringAttr KernelNameAttr = StringAttr::get(getOperation()->getContext(),s);
      getOperation()->setAttr(getKernelNameAttrStr(), KernelNameAttr);
    }

    StringRef getId() {
      return this->getOperation()->getAttr("Id")
        .cast<StringAttr>().strref();
    }

    void setId(int i) {
      setId(std::to_string(i));
    }

    void setId(std::string s) {
      StringAttr IdAttr = StringAttr::get(getOperation()->getContext(),s);
      getOperation()->setAttr(getIdAttrStr(), IdAttr);
    }

    MemRefType getSourceMemrefType() {
      return getSourceMemref().getType().cast<MemRefType>();
    }
    MemRefType getTargetMemrefType() {
      return getTargetMemref().getType().cast<MemRefType>();
    }
    MemRefType getMemRefType() {
      return getSourceMemrefType();
    }

    operand_range getMapOperands() {
      return getIndices();
    }
    void setMapOperands(mlir::ValueRange newoperands);

    /// Returns the affine map used to index the memref for this operation.
    AffineMap getAffineMap() { return getAffineMapAttr().getValue(); }
    AffineMapAttr getAffineMapAttr() {
      return (*this)->getAttr(getMapAttrStr()).cast<AffineMapAttr>();
    }
    void setAffineMap(AffineMap map){
      getOperation()->setAttr(getMapAttrStr(), AffineMapAttr::get(map));
    }

    /// stride
    bool hasStrides();
    bool hasStridesAttr();
    void setStrides(ArrayRef<int64_t> stride);
    void setStrides(Attribute stride);
    void setStrides(DenseI64ArrayAttr stride);
    Attribute getStrides();
    ArrayRef<int64_t> getStridesAsArrayRef();
    static StringRef getStridesAttrStr() { return "strides"; }

  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasCanonicalizer = 1; // PR2: DropUnusedAsyncToken / DedupAsyncDeps
}


def LocalMemAllocOp : ADORA_Op<"LocalMemAlloc", []>{
  let summary = "local memory allocation operation";
  let description = [{
    The `LocalMemAlloc` operation allocates a region of local memory, as specified by its
    memref type. Usually be used as a memref for blockloadop.

    Example:

    ```mlir
    %0 = ADORA.LocalMemAlloc memref<8x64xf32, 1>
    ```
  }];
  let builders = [   
    OpBuilder<(ins "MemRefType":$ResultType, 
                   "std::string":$KernelName )>,
    // /// Builds an block load op without KernelName.
    OpBuilder<(ins "MemRefType":$ResultType)>,
  ];

	//Ignore Variadic<Index>:$offset at this moment
  let arguments = (Variadic<Index>:$indices);
  
  let results = (outs AnyStaticShapeMemRef: $Result);
  // let results = (outs AnyStaticShapeMemRef: $result); /// No results for 
  let extraClassDeclaration = [{
    // unsigned int BlockLen = 0;
    // void setBlockLen(unsigned l){ BlockLen = l;};
    // unsigned int getBlockLen(){ return BlockLen;};
    // Value getOriginalMemref(){ return OriginalMemref();};

    static StringRef getMapAttrStr() { return "map"; }
    static StringRef getKernelNameAttrStr() { return "KernelName"; }
    static StringRef getIdAttrStr() { return "Id"; }
    // static StringRef getBlockLenAttrStr() { return "BlockLen"; }

    // int32_t getBlockLen() {
    //   return this->getOperation()->getAttr("BlockLen")
    //     .cast<IntegerAttr>().getValue().getSExtValue();
    // }

    /////////////////////////////////
    /// Handles of kernel name
    /////////////////////////////////
    bool hasKernelName(){
      return this->getOperation()->hasAttr("KernelName");
    }
    StringRef getKernelName() {
      Attribute KnNameAttr = this->getOperation()->getAttr("KernelName");
      if(KnNameAttr == nullptr)
        return "";
      else
        return KnNameAttr.cast<StringAttr>().strref();
    }

    void setKernelName(std::string s) {
      StringAttr KernelNameAttr = StringAttr::get(getOperation()->getContext(),s);
      getOperation()->setAttr(getKernelNameAttrStr(), KernelNameAttr);
    }

    SmallVector<std::string> getKernelNameAsStrVector();

    std::string addAnotherKernelName(const std::string& newKernel);


    /////////////////////////////////
    /// Handles of ID
    /////////////////////////////////
    StringRef getId() {
      return this->getOperation()->getAttr("Id")
        .cast<StringAttr>().strref();
    }

    void setId(int i) {
      setId(std::to_string(i));
    }

    void setId(std::string s) {
      StringAttr IdAttr = StringAttr::get(getOperation()->getContext(),s);
      getOperation()->setAttr(getIdAttrStr(), IdAttr);
    }
    
    MemRefType getResultType() {
      return getResult().getType().cast<MemRefType>();
    }

    MemRefType getMemrefType() {
      return getResultType();
    }

    /// Returns the affine map used to index the memref for this operation.
    // AffineMap getAffineMap() { return getAffineMapAttr().getValue(); }
    // AffineMapAttr getAffineMapAttr() {
    //   return (*this)->getAttr(getMapAttrStr()).cast<AffineMapAttr>();
    // }

  }];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// KernelCallOp
//===----------------------------------------------------------------------===//

def KernelCallOp : ADORA_Op<"KernelCall",
    [CallOpInterface, MemRefsNormalizable,
     DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "call operation(different from call in func)";
  let description = [{
    The `func.call` operation represents a direct call to a function that is
    within the same symbol scope as the call. The operands and result types of
    the call must match the specified function type. The callee is encoded as a
    symbol reference attribute named "callee".

    Example:

    ```mlir
    %2 = ADORA.kernelcall @my_add(%0, %1) : (f32, f32) -> f32
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);

  let builders = [
    OpBuilder<(ins "func::FuncOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", SymbolRefAttr::get(callee));
      $_state.addTypes(callee.getFunctionType().getResults());
    }]>,
    OpBuilder<(ins "SymbolRefAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", callee);
      $_state.addTypes(results);
    }]>,
    OpBuilder<(ins "StringAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, SymbolRefAttr::get(callee), results, operands);
    }]>,
    OpBuilder<(ins "StringRef":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, StringAttr::get($_builder.getContext(), callee),
            results, operands);
    }]>];

  let extraClassDeclaration = [{
    FunctionType getCalleeType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }
    ::mlir::MutableOperandRange getArgOperandsMutable(){
      return getOperandsMutable();
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }
    void setCalleeFromCallable(CallInterfaceCallable callee) {
      (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
    }

  }];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}



//===----------------------------------------------------------------------===//
// ISEL op
//===----------------------------------------------------------------------===//
def IselOp : ADORA_Op<"isel">{
  let summary = "Insert a Isel Op for CGRA Dataflow Graph";
  let description = [{
  }];

  let builders = [   
    // /// Builds isel with one input.
    OpBuilder<(ins "::mlir::Value":$in)>,
  ];

	//Ignore Variadic<Index>:$offset at this moment
  let arguments = (ins AnyType: $in);
  
  let results = (outs AnyType: $out);
  // let results = (outs AnyStaticShapeMemRef: $result); /// No results for 
  
  let extraClassDeclaration = [{
    ::mlir::Value getResult(){
      return getOut();
    }
  }];

  // let hasCustomAssemblyFormat = 1;
  let assemblyFormat = "$in attr-dict `:` type($in) `->` type($out)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// InterleaverOp op
//===----------------------------------------------------------------------===//
def InterleaverOp : ADORA_Op<"interleaver">{
  let summary = "Merge/interleave node in CGRA processing elements (PE) used in systolic execution.";
  let description = [{
    "Interleave" combines multiple inputs into a single sequence of outputs, akin to Parallel-to-Sequential conversion. 
    The interleave operation should be linked to an output node (or a add?).
  }];

  let builders = [
    OpBuilder<(ins "ValueRange":$inputs)>,
    OpBuilder<(ins "ValueRange":$inputs, "ArrayRef<int64_t>": $outShape)>
  ];

 // Define inputs as a variadic argument.
  let arguments = (ins Variadic<AnyType>: $inputs);
  
  // Define outputs.
  let results = (outs AnyType: $out);
  // let results = (outs AnyStaticShapeMemRef: $result); /// No results for 
  
  let extraClassDeclaration = [{
    ::mlir::Value getResult(){
      return getOut();
    }

    ::mlir::VectorType getOutVectorType() {
      return ::llvm::cast<VectorType>(getOut().getType());
    }

    ArrayRef<int64_t> getOutVectorShape() {
      return getOutVectorType().getShape();
    }

    Type getElementType(){
      return getOutVectorType().getElementType();
    }

    ::mlir::Value getInput(int idx){
      return getInputs()[idx];
    }

    int getInterleaveNumber(){
      return getInputs().size();
    }

    int getMergeNumber(){
      return getInputs().size();
    }

  }];

  // let hasCustomAssemblyFormat = 1;
  // let assemblyFormat = "$in attr-dict `:` type($in) `->` type($out)";
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// DeinterleaverOp op
//===----------------------------------------------------------------------===//
def DeinterleaverOp : ADORA_Op<"deinterleaver">{
  let summary = "split/deinterleave node in CGRA processing elements (PE) used in systolic execution.";
  let description = [{
    "Deinterleave": "The Deinterleave node distributes a time-multiplexed (sequential) input 
    stream across multiple parallel output channels. It effectively performs a 
    Sequential-to-Parallel conversion, allowing data arriving over several cycles on a 
    single input port to be fanned out into multiple parallel outputs in the same cycle. 
    This is opposite to interleave op."
  }];

  let builders = [
    OpBuilder<(ins "Value":$source)>,
    OpBuilder<(ins "::mlir::TypedValue<::mlir::VectorType>":$source)>,
  ];

  /// serial input
  let arguments = (ins
    AnyVectorOfAnyRank:$input
  );

  /// mutiple output
  let results = (outs Variadic<AnyType>:$results); 
  
  let extraClassDeclaration = [{
    ValueRange getOutputs(){
      return getResults();
    }

    ::mlir::VectorType getInputVectorType() {
      return ::llvm::cast<VectorType>(getInput().getType());
    }

    ArrayRef<int64_t> getInputVectorShape() {
      return getInputVectorType().getShape();
    }

    Type getElementType(){
      return getInputVectorType().getElementType();
    }

    ::mlir::Value getOutput(int idx){
      return getOutputs()[idx];
    }

    ::mlir::Value getResult(int idx){
      return getResults()[idx];
    }

    int getDeinterleaveNumber(){
      return getOutputs().size();
    }

    int getSplitNumber(){
      return getOutputs().size();
    }

  }];

  // let hasCustomAssemblyFormat = 1;
  // let assemblyFormat = "$in attr-dict `:` type($in) `->` type($out)";
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// PR3 — Async event primitives
//
// Appear only after `adora-lower-async-tokens` runs. They replace SSA
// !ADORA.token def-use chains with explicit runtime-level synchronisation.
// Lowered to llvm.call @adoraEvent*/adoraEventRecord/adoraEventWait by
// `adora-to-llvm-async-runtime` (PR3 commit C).
//===----------------------------------------------------------------------===//

def ADORA_EventCreateOp : ADORA_Op<"event.create", []> {
  let summary = "Allocate an ADORA async event handle (!ADORA.token).";
  let results = (outs ADORA_TokenType:$token);
  let assemblyFormat = "attr-dict `->` type($token)";
}

def ADORA_EventDestroyOp : ADORA_Op<"event.destroy", []> {
  let summary = "Release an ADORA async event handle.";
  let arguments = (ins ADORA_TokenType:$token);
  let assemblyFormat = "$token attr-dict `:` type($token)";
}

def ADORA_SignalOp : ADORA_Op<"signal", []> {
  let summary = "Signal (record) an ADORA async event on a stream.";
  let arguments = (ins ADORA_TokenType:$token, I32Attr:$stream);
  let assemblyFormat = "$token `on` `stream` $stream attr-dict `:` type($token)";
}

def ADORA_WaitOp : ADORA_Op<"wait", []> {
  let summary = "Wait for an ADORA async event on a stream before proceeding.";
  let arguments = (ins ADORA_TokenType:$token, I32Attr:$stream);
  let assemblyFormat = "$token `on` `stream` $stream attr-dict `:` type($token)";
}

#endif // ADORA_OPS
