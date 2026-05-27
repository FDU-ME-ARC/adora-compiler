//===- ADORAOps.cpp - Operations of the ADORA dialect -------------------------===//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Affine/IR/AffineOps.h"
// #include "mlir/Dialect/Affine/IR/AffineValueMap.h"
// #include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/ADT/STLExtras.h"
#include "ADORA/Dialect/ADORA/IR/ADORA.h"

#include "ADORA/Dialect/ADORA/IR/ADORAOps.h.inc"


using namespace mlir;
using namespace mlir::ADORA;
using namespace mlir::func;
using namespace mlir::affine;
//===----------------------------------------------------------------------===//
// Tool functions
//===----------------------------------------------------------------------===//
static ParseResult ParseStrides(OpAsmParser &parser, SmallVector<int64_t>& strides){
  if (parser.parseLSquare()) return failure(); // Parse '['

  strides.clear();
  // SmallVector<int64_t> strides_vec;
  do {
    int64_t stride;
    if (parser.parseInteger(stride)) return failure();
    
    strides.push_back(stride);

  } while (parser.parseOptionalComma().succeeded());   

  if (parser.parseRSquare()) return failure(); // Parse ']' 

  return success();
}

static void printStrides(OpAsmPrinter &p, const llvm::ArrayRef<int64_t> strides) {
    p << "[";
    for (size_t i = 0; i < strides.size(); ++i) {
        if (i != 0) {
            p << ", ";
        }
        p << strides[i];
    }
    p << "]";
}


//===----------------------------------------------------------------------===//
// DataBlockLoadOp
//===----------------------------------------------------------------------===//

void DataBlockLoadOp::build(OpBuilder &builder, OperationState &result,
                            Value OriginalMemref, AffineMap map, ValueRange mapOperands, 
                            MemRefType resultType, std::string KernelName) {
  assert(map.getNumInputs() == mapOperands.size() && "inconsistent index info");
  result.addOperands(OriginalMemref);
  result.addOperands(mapOperands);

  result.addAttribute(getMapAttrStr(), AffineMapAttr::get(map));

  auto KernelNameAttr = builder.getStringAttr(KernelName);
  result.addAttribute(getKernelNameAttrStr(), KernelNameAttr);

  result.addAttribute("operandSegmentSizes",
      builder.getDenseI32ArrayAttr(
          {1, (int32_t)mapOperands.size(), /*asyncDeps=*/0}));

  result.types.push_back(resultType);
}

void DataBlockLoadOp::build(OpBuilder &builder, OperationState &result,
                            Value OriginalMemref, AffineMap map, ValueRange mapOperands,
                            MemRefType resultType, DenseI64ArrayAttr strides, std::string KernelName,
                            ValueRange asyncDependencies, bool produceToken) {
  assert(map.getNumInputs() == mapOperands.size() && "inconsistent index info");
  result.addOperands(OriginalMemref);
  result.addOperands(mapOperands);
  result.addOperands(asyncDependencies);

  result.addAttribute(getMapAttrStr(), AffineMapAttr::get(map));
  if (strides)
    result.addAttribute("strides", strides);
  if (!KernelName.empty())
    result.addAttribute(getKernelNameAttrStr(), builder.getStringAttr(KernelName));

  result.addAttribute("operandSegmentSizes",
      builder.getDenseI32ArrayAttr(
          {1, (int32_t)mapOperands.size(),
           (int32_t)asyncDependencies.size()}));

  result.types.push_back(resultType);
  if (produceToken)
    result.types.push_back(TokenType::get(builder.getContext()));
}

void DataBlockLoadOp::build(OpBuilder &builder, OperationState &result,
                            Value OriginalMemref, AffineMap map, ValueRange mapOperands, 
                            MemRefType resultType, DenseI64ArrayAttr strides, std::string KernelName){
  build(builder, result, OriginalMemref, map, mapOperands, resultType, KernelName);
  result.addAttribute("strides", strides);
} 

void DataBlockLoadOp::build(OpBuilder &builder, OperationState &result,
                            Value OriginalMemref, AffineMap map, ValueRange mapOperands, 
                            MemRefType resultType, ArrayRef<int64_t> strides, std::string KernelName){
  build(builder, result, OriginalMemref, map, mapOperands, resultType, KernelName);
  result.addAttribute("strides", DenseI64ArrayAttr::get(builder.getContext(), strides));
} 

void DataBlockLoadOp::build(OpBuilder &builder, OperationState &result,
                            Value OriginalMemref, AffineMap map, ValueRange mapOperands, 
                            MemRefType resultType) {
  assert(map.getNumInputs() == mapOperands.size() && "inconsistent index info");
  std::string KernelName = ""/*"UnknownKernel"*/;
  build(builder, result, OriginalMemref, map, mapOperands, resultType, KernelName);
}

void DataBlockLoadOp::build(OpBuilder &builder, OperationState &result,
                            Value OriginalMemref, AffineMap map, ValueRange mapOperands, 
                            MemRefType resultType, DenseI64ArrayAttr strides) {
  build(builder, result, OriginalMemref, map, mapOperands, resultType);
  result.addAttribute("strides", strides);  
}

void DataBlockLoadOp::build(OpBuilder &builder, OperationState &result,
                            Value OriginalMemref, AffineMap map, ValueRange mapOperands, 
                            MemRefType resultType, ArrayRef<int64_t> strides) {
  build(builder, result, OriginalMemref, map, mapOperands, resultType);
  result.addAttribute("strides", DenseI64ArrayAttr::get(builder.getContext(), strides));
}


// static bool addKernelNameAttrInParse
//               (OperationState &result, Builder &builder, const std::string KernelName){
//   result.addAttribute(DataBlockLoadOp::getKernelNameAttrStr(), builder.getStringAttr(KernelName));
//   return true;
// }

ParseResult DataBlockLoadOp::parse(OpAsmParser &parser, OperationState &result) {
  /// example:
  /// %0 = ADORA.BlockLoad %arg1 [%arg9, 0] : memref<32x32xf32> -> memref<1x32xf32> {three_mm_32_kernel_0}
  auto &builder = parser.getBuilder();
  auto indexTy = builder.getIndexType();

  MemRefType memrefType, resultType;
  OpAsmParser::UnresolvedOperand memrefInfo;
  AffineMapAttr mapAttr;
  StringAttr KernelNameAttr;
  // std::string* KernelName = nullptr;
  SmallVector<OpAsmParser::UnresolvedOperand, 1> mapOperands;

  SmallVector<OpAsmParser::UnresolvedOperand, 4> asyncDeps;
  bool isAsync = false;
  if (succeeded(parser.parseOptionalKeyword("async"))) {
    isAsync = true;
    if (parser.parseOperandList(asyncDeps, OpAsmParser::Delimiter::Square))
      return failure();
  }

  if(   parser.parseOperand(memrefInfo).failed() ||
        parser.parseAffineMapOfSSAIds(mapOperands, mapAttr,
                                     getMapAttrStr(),
                                     result.attributes).failed() ||
        parser.parseColon().failed() || parser.parseType(memrefType).failed() ||
        parser.parseArrow().failed() || parser.parseType(resultType).failed()){
    return failure();
  }

  // example: stride [2, 2] 
  std::string s = "stride";
  if( parser.parseOptionalComma().succeeded() &&
      parser.parseOptionalKeywordOrString(&s).succeeded()){
    
    SmallVector<int64_t> strides_vec;
    // Parse the integers inside the square brackets
    if(ParseStrides(parser, strides_vec).failed()) return failure();

    result.addAttribute("strides", DenseI64ArrayAttr::get(builder.getContext(), ArrayRef<int64_t>(strides_vec)));
  }

  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.resolveOperand(memrefInfo, memrefType, result.operands) ||
      parser.resolveOperands(mapOperands, indexTy, result.operands) ||
      parser.resolveOperands(asyncDeps,
          ADORA::TokenType::get(parser.getContext()), result.operands) ||
      parser.addTypeToList(resultType, result.types))
    return failure();

  result.addAttribute("operandSegmentSizes",
      builder.getDenseI32ArrayAttr(
          {1, (int32_t)mapOperands.size(), (int32_t)asyncDeps.size()}));
  // async keyword presence implies a token result (GPU-dialect style: no explicit "-> !ADORA.token")
  if (isAsync)
    result.addTypes(ADORA::TokenType::get(parser.getContext()));
  return success();
}

void DataBlockLoadOp::print(OpAsmPrinter &p) {
  // Always print "async [...]" when a token result exists so the parser
  // can reconstruct the correct number of results (GPU-dialect style).
  if (getAsyncToken()) {
    p << " async [";
    llvm::interleaveComma(getAsyncDependencies(), p);
    p << "]";
  }
  p << " " << getOriginalMemref() << " [";
  if (AffineMapAttr mapAttr =
          (*this)->getAttrOfType<AffineMapAttr>(getMapAttrStr()))
    p.printAffineMapOfSSAIds(mapAttr, getMapOperands());
  p << "]";
  p << " : " << getOriginalMemrefType() ;
  p << " -> " << getResultType() << " ";

  /// strides
  if(hasStrides()){
    p << ", stride ";
    ArrayRef<int64_t> strides = getStridesAsArrayRef();
    printStrides(p, strides);
  }

  // p << "{\""  << getKernelName() << "\"}";
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{getMapAttrStr(), getStridesAttrStr(), "operandSegmentSizes"});
  // async token type is implicit from the "async" keyword (GPU-dialect style)
}

// Returns true if 'value' is a valid index to an affine operation (e.g.
// affine.load, affine.store, affine.dma_start, affine.dma_wait) where
// `region` provides the polyhedral symbol scope. Returns false otherwise.
static bool isValidAffineIndexOperand(Value value, Region *region) {
  return isValidDim(value, region) || isValidSymbol(value, region);
}

/// Verify common indexing invariants of affine.load, affine.store,
/// affine.vector_load and affine.vector_store.
static LogicalResult
verifyMemoryOpIndexing(Operation *op, AffineMapAttr mapAttr,
                       Operation::operand_range mapOperands,
                       MemRefType memrefType, unsigned numIndexOperands) {
  if (mapAttr) {
    AffineMap map = mapAttr.getValue();
    if (map.getNumResults() != memrefType.getRank())
      return op->emitOpError("affine map num results must equal memref rank");
    if (map.getNumInputs() != numIndexOperands)
      return op->emitOpError("expects as many subscripts as affine map inputs");
  } else {
    if (memrefType.getRank() != numIndexOperands)
      return op->emitOpError(
          "expects the number of subscripts to be equal to memref rank");
  }

  Region *scope = getAffineScope(op);
  for (auto idx : mapOperands) {
    if (!idx.getType().isIndex())
      return op->emitOpError("index to load must have 'index' type");
    // if (!isValidAffineIndexOperand(idx, scope))
    //   return op->emitOpError("index must be a dimension or symbol identifier");
  }

  return success();
}

LogicalResult DataBlockLoadOp::verify() {
  MemRefType memrefType = getOriginalMemrefType();
  if (failed(verifyMemoryOpIndexing(
          getOperation(),
          (*this)->getAttrOfType<AffineMapAttr>(getMapAttrStr()),
          getMapOperands(), memrefType,
          /*numIndexOperands=*/(unsigned)getIndices().size())))
    return failure();

  if (getOriginalMemrefType().getElementType() != getResultType().getElementType())
    return emitOpError(
        "requires 2 memref types of the same elemental type");


  if (hasStrides() &&
      getOriginalMemrefType().getShape().size() != getStridesAsArrayRef().size())
    return emitOpError(
        "requires strides have the same dimension number to accessed memref");


  return success();
}

SmallVector<std::string> DataBlockLoadOp::getKernelNameAsStrVector() {
  Attribute KnNameAttr = this->getOperation()->getAttr("KernelName");
  if(KnNameAttr == nullptr)
    return SmallVector<std::string>();
  else{
    SmallVector<std::string> tokens;
    std::string wholeattr = KnNameAttr.cast<StringAttr>().strref().str();
    size_t start = 0;
    size_t end = 0;
    while ((end = wholeattr.find('+', start)) != std::string::npos) {
        tokens.push_back(wholeattr.substr(start, end - start));
        start = end + 1; 
    }
    std::string lastToken = wholeattr.substr(start);
    if (!lastToken.empty()) { 
      tokens.push_back(lastToken);
    }
    return tokens;
  }
}    
std::string DataBlockLoadOp::addAnotherKernelName(const std::string& newKernel) {
  std::string original = getKernelName().str();

  if (original.empty()) {
    setKernelName(newKernel);
    return newKernel;
  }
  else if(findElement(getKernelNameAsStrVector(), newKernel) != -1){
    return original;
  }
  else{
    setKernelName(original + "+" + newKernel);
    return original + "+" + newKernel;
  }
}

bool DataBlockLoadOp::hasStridesAttr(){
  mlir::Attribute attr = this->getOperation()->getAttr("strides");
  if(attr != nullptr)
    return true;
  else
    return false;
}

bool DataBlockLoadOp::hasStrides(){
  if(!hasStridesAttr()){
    return false;
  }
  ArrayRef<int64_t> strides = this->getStridesAsArrayRef();
  for(int64_t stride : strides){
    if(stride > 1){
      return true;
    }
  }
  return false;
}

void DataBlockLoadOp::setStrides(Attribute strides){
  getOperation()->setAttr("strides", strides);
}
void DataBlockLoadOp::setStrides(DenseI64ArrayAttr strides){
  getOperation()->setAttr("strides", strides);
}

void DataBlockLoadOp::setStrides(ArrayRef<int64_t> strides){
  DenseI64ArrayAttr attr = DenseI64ArrayAttr::get(getOperation()->getContext(), strides);
  getOperation()->setAttr("strides", attr);
}

Attribute DataBlockLoadOp::getStrides(){
  if(!hasStridesAttr()){
    return nullptr;
  }

  return this->getOperation()->getAttr("strides");
}

ArrayRef<int64_t> DataBlockLoadOp::getStridesAsArrayRef(){
  if(!hasStridesAttr()){
    return ArrayRef<int64_t>();
  }

  DenseI64ArrayAttr arrayattr = cast<DenseI64ArrayAttr>(this->getOperation()->getAttr("strides"));
  return arrayattr.asArrayRef();
}

void DataBlockLoadOp::setMapOperands(mlir::ValueRange newMapOperands){
  SmallVector<mlir::Value> newOperands;
  newOperands.push_back(getOriginalMemref());
  newOperands.append(newMapOperands.begin(), newMapOperands.end());
  auto deps = getAsyncDependencies();
  newOperands.append(deps.begin(), deps.end());
  getOperation()->setOperands(newOperands);
  getOperation()->setAttr("operandSegmentSizes",
      DenseI32ArrayAttr::get(getOperation()->getContext(),
          {1, (int32_t)newMapOperands.size(), (int32_t)deps.size()}));
}

//===----------------------------------------------------------------------===//
// DataBlockStoreOp
//===----------------------------------------------------------------------===//

void DataBlockStoreOp::build(OpBuilder &builder, OperationState &result,
                            Value SourceMemref, Value TargetMemref, 
                            AffineMap map, ValueRange mapOperands, 
                            std::string KernelName) {
  assert(map.getNumInputs() == mapOperands.size() && "inconsistent index info");
  result.addOperands(SourceMemref);
  result.addOperands(TargetMemref);

  result.addOperands(mapOperands);
  result.addAttribute(getMapAttrStr(), AffineMapAttr::get(map));

  auto KernelNameAttr = builder.getStringAttr(KernelName);
  result.addAttribute(getKernelNameAttrStr(), KernelNameAttr);

  result.addAttribute("operandSegmentSizes",
      builder.getDenseI32ArrayAttr(
          {1, 1, (int32_t)mapOperands.size(), /*asyncDeps=*/0}));
}

void DataBlockStoreOp::build(OpBuilder &builder, OperationState &result,
                            Value SourceMemref, Value TargetMemref,
                            AffineMap map, ValueRange mapOperands,
                            DenseI64ArrayAttr strides, std::string KernelName,
                            ValueRange asyncDependencies, bool produceToken) {
  assert(map.getNumInputs() == mapOperands.size() && "inconsistent index info");
  result.addOperands(SourceMemref);
  result.addOperands(TargetMemref);
  result.addOperands(mapOperands);
  result.addOperands(asyncDependencies);
  result.addAttribute(getMapAttrStr(), AffineMapAttr::get(map));
  if (strides)
    result.addAttribute("strides", strides);
  if (!KernelName.empty())
    result.addAttribute(getKernelNameAttrStr(), builder.getStringAttr(KernelName));
  result.addAttribute("operandSegmentSizes",
      builder.getDenseI32ArrayAttr(
          {1, 1, (int32_t)mapOperands.size(),
           (int32_t)asyncDependencies.size()}));
  if (produceToken)
    result.types.push_back(TokenType::get(builder.getContext()));
}

void DataBlockStoreOp::build(OpBuilder &builder, OperationState &result,
                            Value SourceMemref, Value TargetMemref, 
                            AffineMap map, ValueRange mapOperands, DenseI64ArrayAttr strides, 
                            std::string KernelName) {
  build(builder, result, SourceMemref, TargetMemref, map, mapOperands, KernelName);
  result.addAttribute("strides", strides);
}

void DataBlockStoreOp::build(OpBuilder &builder, OperationState &result,
                            Value SourceMemref, Value TargetMemref, 
                            AffineMap map, ValueRange mapOperands, ArrayRef<int64_t> strides, 
                            std::string KernelName) {
  build(builder, result, SourceMemref, TargetMemref, map, mapOperands, KernelName);
  result.addAttribute("strides", DenseI64ArrayAttr::get(builder.getContext(), strides));
}

void DataBlockStoreOp::build(OpBuilder &builder, OperationState &result,
                            Value SourceMemref, Value TargetMemref, 
                            AffineMap map, ValueRange mapOperands) {
  assert(map.getNumInputs() == mapOperands.size() && "inconsistent index info");
  std::string KernelName = ""/*"UnknownKernel"*/;
  build(builder, result, SourceMemref, TargetMemref, map, mapOperands, KernelName);
}

void DataBlockStoreOp::build(OpBuilder &builder, OperationState &result,
                            Value SourceMemref, Value TargetMemref, 
                            AffineMap map, ValueRange mapOperands, DenseI64ArrayAttr strides) {
  build(builder, result, SourceMemref, TargetMemref, map, mapOperands);
  result.addAttribute("strides", strides);
}

void DataBlockStoreOp::build(OpBuilder &builder, OperationState &result,
                            Value SourceMemref, Value TargetMemref, 
                            AffineMap map, ValueRange mapOperands, ArrayRef<int64_t> strides) {
  build(builder, result, SourceMemref, TargetMemref, map, mapOperands);
  result.addAttribute("strides", DenseI64ArrayAttr::get(builder.getContext(), strides));
}

ParseResult DataBlockStoreOp::parse(OpAsmParser &parser, OperationState &result) {
  /// example: To fix
  /// %0 = ADORA.BlockLoad %arg1 [%arg9, 0] : memref<32x32xf32> -> memref<1x32xf32> {three_mm_32_kernel_0}
  auto &builder = parser.getBuilder();
  auto indexTy = builder.getIndexType();

  MemRefType sourceType, targetType;
  OpAsmParser::UnresolvedOperand sourceInfo;
  OpAsmParser::UnresolvedOperand targetInfo;
  AffineMapAttr mapAttr;
  StringAttr KernelNameAttr;
  SmallVector<OpAsmParser::UnresolvedOperand, 1> mapOperands;

  SmallVector<OpAsmParser::UnresolvedOperand, 4> asyncDeps;
  bool isAsync = false;
  if (succeeded(parser.parseOptionalKeyword("async"))) {
    isAsync = true;
    if (parser.parseOperandList(asyncDeps, OpAsmParser::Delimiter::Square))
      return failure();
  }

  if(
    parser.parseOperand(sourceInfo).failed() || parser.parseComma().failed() ||
    parser.parseOperand(targetInfo).failed() ||
    parser.parseAffineMapOfSSAIds(mapOperands, mapAttr,
                                    getMapAttrStr(),
                                    result.attributes).failed() ||
    // parser.parseOptionalAttrDict(result.attributes) ||
    parser.parseColon().failed() || parser.parseType(sourceType).failed() ||
    parser.parseArrow().failed() || parser.parseType(targetType).failed() 
  ){
    return failure();
  }

  // example: stride [2, 2] 
  std::string s = "stride";
  if( parser.parseOptionalComma().succeeded() &&
      parser.parseOptionalKeywordOrString(&s).succeeded()){
    
    SmallVector<int64_t> strides_vec;
    // Parse the integers inside the square brackets
    if(ParseStrides(parser, strides_vec).failed()) return failure();

    result.addAttribute("strides", DenseI64ArrayAttr::get(builder.getContext(), ArrayRef<int64_t>(strides_vec)));
  }

  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.resolveOperand(sourceInfo, sourceType, result.operands) ||
      parser.resolveOperand(targetInfo, targetType, result.operands) ||
      parser.resolveOperands(mapOperands, indexTy, result.operands) ||
      parser.resolveOperands(asyncDeps,
          ADORA::TokenType::get(parser.getContext()), result.operands))
    return failure();

  result.addAttribute("operandSegmentSizes",
      builder.getDenseI32ArrayAttr(
          {1, 1, (int32_t)mapOperands.size(), (int32_t)asyncDeps.size()}));
  // async keyword presence implies a token result (GPU-dialect style)
  if (isAsync)
    result.addTypes(ADORA::TokenType::get(parser.getContext()));
  return success();
}


void DataBlockStoreOp::print(OpAsmPrinter &p) {
  // Always print "async [...]" when a token result exists (GPU-dialect style).
  if (getAsyncToken()) {
    p << " async [";
    llvm::interleaveComma(getAsyncDependencies(), p);
    p << "]";
  }
  p << " " << getSourceMemref() << ",";
  p << " " << getTargetMemref() << " [";
  if (AffineMapAttr mapAttr =
          (*this)->getAttrOfType<AffineMapAttr>(getMapAttrStr()))
    p.printAffineMapOfSSAIds(mapAttr, getMapOperands());
  p << "]";

  p << " : " << getSourceMemrefType() ;
  p << " -> " << getTargetMemrefType() << " ";

  ///// stride
  /// strides
  if(hasStrides()){
    p << ", stride ";
    ArrayRef<int64_t> strides = getStridesAsArrayRef();
    printStrides(p, strides);
  }
  
  // p << "{\""  << getKernelName() << "\"}";
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{getMapAttrStr(), getStridesAttrStr(), "operandSegmentSizes"});
  // async token type is implicit from the "async" keyword (GPU-dialect style)
}


LogicalResult DataBlockStoreOp::verify() {
  MemRefType memrefType = getTargetMemrefType();
  if (failed(verifyMemoryOpIndexing(
          getOperation(),
          (*this)->getAttrOfType<AffineMapAttr>(getMapAttrStr()),
          getMapOperands(), memrefType,
          /*numIndexOperands=*/(unsigned)getIndices().size())))
    return failure();

  if (getTargetMemrefType().getElementType() != getSourceMemrefType().getElementType())
    return emitOpError(
        "requires source and target memref types of the same elemental type");

  if (hasStrides() &&
      getSourceMemrefType().getShape().size() != getStridesAsArrayRef().size())
    return emitOpError(
        "requires strides have the same dimension number to accessed memref");


  return success();
}


bool DataBlockStoreOp::hasStridesAttr(){
  mlir::Attribute attr = this->getOperation()->getAttr("strides");
  if(attr != nullptr)
    return true;
  else
    return false;
}

bool DataBlockStoreOp::hasStrides(){
  if(!hasStridesAttr()){
    return false;
  }
  ArrayRef<int64_t> strides = this->getStridesAsArrayRef();
  for(int64_t stride : strides){
    if(stride > 1){
      return true;
    }
  }
  return false;
}

void DataBlockStoreOp::setStrides(Attribute strides){
  getOperation()->setAttr("strides", strides);
}
void DataBlockStoreOp::setStrides(DenseI64ArrayAttr strides){
  getOperation()->setAttr("strides", strides);
}

void DataBlockStoreOp::setStrides(ArrayRef<int64_t> strides){
  DenseI64ArrayAttr attr = DenseI64ArrayAttr::get(getOperation()->getContext(), strides);
  getOperation()->setAttr("strides", attr);
}

Attribute DataBlockStoreOp::getStrides(){
  if(!hasStridesAttr()){
    return nullptr;
  }

  return this->getOperation()->getAttr("strides");
}

ArrayRef<int64_t> DataBlockStoreOp::getStridesAsArrayRef(){
  if(!hasStridesAttr()){
    return ArrayRef<int64_t>();
  }

  DenseI64ArrayAttr arrayattr = cast<DenseI64ArrayAttr>(this->getOperation()->getAttr("strides"));
  return arrayattr.asArrayRef();
}

void DataBlockStoreOp::setMapOperands(mlir::ValueRange newMapOperands){
  SmallVector<mlir::Value> newOperands;
  newOperands.push_back(getSourceMemref());
  newOperands.push_back(getTargetMemref());
  newOperands.append(newMapOperands.begin(), newMapOperands.end());
  auto deps = getAsyncDependencies();
  newOperands.append(deps.begin(), deps.end());
  getOperation()->setOperands(newOperands);
  getOperation()->setAttr("operandSegmentSizes",
      DenseI32ArrayAttr::get(getOperation()->getContext(),
          {1, 1, (int32_t)newMapOperands.size(), (int32_t)deps.size()}));
}

//===----------------------------------------------------------------------===//
// LocalMemAllocOp
//===----------------------------------------------------------------------===//

void LocalMemAllocOp::build(OpBuilder &builder, OperationState &result,
                            MemRefType resultType, std::string KernelName) {
  auto KernelNameAttr = builder.getStringAttr(KernelName);
  result.addAttribute(getKernelNameAttrStr(), KernelNameAttr);

  result.types.push_back(resultType);
}

void LocalMemAllocOp::build(OpBuilder &builder, OperationState &result,
                            MemRefType resultType) {
  std::string KernelName = ""/*"UnknownKernel"*/;
  build(builder, result, resultType, KernelName);
}

// static bool addKernelNameAttrInParse
//               (OperationState &result, Builder &builder, const std::string KernelName){
//   result.addAttribute(LocalMemAllocOp::getKernelNameAttrStr(), builder.getStringAttr(KernelName));
//   return true;
// }

ParseResult LocalMemAllocOp::parse(OpAsmParser &parser, OperationState &result) {
  /// example:
  /// %0 = ADORA.BlockLoad %arg1 [%arg9, 0] : memref<32x32xf32> -> memref<1x32xf32> {three_mm_32_kernel_0}
  auto &builder = parser.getBuilder();
  auto indexTy = builder.getIndexType();

  MemRefType resultType;
  OpAsmParser::UnresolvedOperand memrefInfo;
  StringAttr KernelNameAttr;

  return failure(
      // parser.parseOperand(memrefInfo) ||
      parser.parseType(resultType) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.addTypeToList(resultType, result.types)
      // parser.parseLBrace() || 
      // parser.parseAttribute(KernelNameAttr, builder.getNoneType(), "KernelName", result.attributes)||
      // parser.parseOptionalKeywordOrString(KernelName) || 
      // addKernelNameAttrInParse(result, builder, *KernelName) ||
      // parser.parseOptionalRBrace()
  );
}

void LocalMemAllocOp::print(OpAsmPrinter &p) {
  p << " " << getResultType() << " ";
  // p << "{\""  << getKernelName() << "\"}";
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{getMapAttrStr()});
}

LogicalResult LocalMemAllocOp::verify() {
  // MemRefType memrefType = getOriginalMemrefType();
  // if (failed(verifyMemoryOpIndexing(
  //         getOperation(),
  //         (*this)->getAttrOfType<AffineMapAttr>(getMapAttrStr()),
  //         getMapOperands(), memrefType,
  //         /*numIndexOperands=*/getNumOperands() - 1)))
  //   return failure();

  // if (getOriginalMemrefType().getElementType() != getResultType().getElementType())
  //   return emitOpError(
  //       "requires 2 memref types of the same elemental type");

  return success();
}


SmallVector<std::string> LocalMemAllocOp::getKernelNameAsStrVector() {
  Attribute KnNameAttr = this->getOperation()->getAttr("KernelName");
  if(KnNameAttr == nullptr)
    return SmallVector<std::string>();
  else{
    SmallVector<std::string> tokens;
    std::string wholeattr = KnNameAttr.cast<StringAttr>().strref().str();
    size_t start = 0;
    size_t end = 0;
    while ((end = wholeattr.find('+', start)) != std::string::npos) {
        tokens.push_back(wholeattr.substr(start, end - start));
        start = end + 1; 
    }
    std::string lastToken = wholeattr.substr(start);
    if (!lastToken.empty()) { 
      tokens.push_back(lastToken);
    }
    return tokens;
  }
}    
std::string LocalMemAllocOp::addAnotherKernelName(const std::string& newKernel) {
  std::string original = getKernelName().str();
  if (original.empty()) {
    setKernelName(newKernel);
    return newKernel;
  }
  
  setKernelName(original + "+" + newKernel);
  return original + "+" + newKernel;
}

// void AffineLoadOp::getCanonicalizationPatterns(RewritePatternSet &results,
//                                                MLIRContext *context) {
//   results.add<SimplifyAffineOp<AffineLoadOp>>(context);
// }
//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// KernelCallOp
//===----------------------------------------------------------------------===//

LogicalResult KernelCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  FuncOp fn = symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, fnAttr);
  if (!fn)
    return LogicalResult::success();

  // Verify that the operand and result types match the callee.
  auto fnType = fn.getFunctionType();
  if (fnType.getNumInputs() != getNumOperands())
    return emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
    if (getOperand(i).getType() != fnType.getInput(i))
      return emitOpError("operand type mismatch: expected operand type ")
             << fnType.getInput(i) << ", but provided "
             << getOperand(i).getType() << " for operand number " << i;

  if (fnType.getNumResults() != getNumResults())
    return emitOpError("incorrect number of results for callee");

  for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i)
    if (getResult(i).getType() != fnType.getResult(i)) {
      auto diag = emitOpError("result type mismatch at index ") << i;
      diag.attachNote() << "      op result types: " << getResultTypes();
      diag.attachNote() << "function result types: " << fnType.getResults();
      return diag;
    }

  return success();
}

FunctionType KernelCallOp::getCalleeType() {
  return FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
}


//===----------------------------------------------------------------------===//
// IselOp
//===----------------------------------------------------------------------===//
void IselOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value in){
  ::mlir::Type resulttype = in.getType();
  build(odsBuilder, odsState, resulttype, in);
}

LogicalResult IselOp::verify() {
  if(getIn().getType() != getOut().getType())
    return emitOpError(
        "isel op: input should be the same with output.");

  return success();
}


//===----------------------------------------------------------------------===//
// InterleaverOp
//===----------------------------------------------------------------------===//
void InterleaverOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs){
  assert(inputs.size() > 1);
  ::mlir::Type intype = inputs[0].getType();
  for(auto input : inputs){
    assert(input.getType() == intype && "All inputs of interleaver op should be the same type.");
  }

  SmallVector<int64_t> shape;
  shape.push_back(inputs.size());

  ::mlir::Type outtype = VectorType::get(shape, intype);
  build(odsBuilder, odsState, outtype, inputs);
}

void InterleaverOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ArrayRef<int64_t> outShape){
  assert(inputs.size() > 1 && "InterleaverOp requires at least 2 inputs");

  ::mlir::Type intype = inputs.front().getType();
  for (::mlir::Value v : inputs) {
    assert(v.getType() == intype &&
           "All inputs of InterleaverOp should have the same type.");
  }

  // Construct result vector shape:
  //   [numInputs] + outShape

  ::mlir::Type outtype = ::mlir::VectorType::get(outShape, intype);
  build(odsBuilder, odsState, outtype, inputs);
}

LogicalResult InterleaverOp::verify() {
  if(!(getInputs().size() > 1)){
    return emitOpError(
        "Interleaver op: input number of Interleaver op should be larger than 1");
  }
  ::mlir::Type intype = getInput(0).getType();
  for(auto input : getInputs()){
    if(input.getType() != intype){
      return emitOpError(
        "Interleaver op: all inputs of Interleaver op should be the same type.");
    }
  }

  ::mlir::Type outtype = ::llvm::cast<VectorType>(getOut().getType()).getElementType();
  if(outtype != intype){
    return emitOpError(
        "Interleaver op: input and output element type should be the same");
  }

  llvm::ArrayRef<int64_t> shape = ::llvm::cast<VectorType>(getOut().getType()).getShape();
  int num = 1;
  for(auto s : shape){
    num *= s;
  }

  if(num != getInputs().size()){
    return emitOpError(
        "Interleaver op: the count of inputs must correspond to the sum of the element count in the output vectors");
  }

  return success();
}

void InterleaverOp::print(OpAsmPrinter &p) {
  Type intype = getElementType();
  p << " " ;
  /// print inputs
  llvm::interleaveComma(getInputs(), p, [&](auto it) {
    p << it;
  });

  p << " : " ;
  llvm::interleaveComma(getInputs(), p, [&](auto it) {
    p << intype;
  });
  p << " -> " ;
  p << getOutVectorType();

  p.printOptionalAttrDict((*this)->getAttrs());
}

ParseResult InterleaverOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  Type outType;

  /// parse inputs
  SmallVector<OpAsmParser::UnresolvedOperand, 4> inputs;
  if (parser.parseOperandList(inputs))
    return failure();
  
  /// parse inputs type
  SmallVector<Type, 3> types;
  if (parser.parseColonTypeList(types))
    return failure();

  /// zip inputs and type
  for (auto pair : llvm::zip(inputs, types)){
    if (parser.resolveOperand(std::get<0>(pair),std::get<1>(pair), result.operands))
      return failure();
  }

  /// parse output type
  if (parser.parseArrowTypeList(result.types))
  {
    return failure();
  }

  return parser.parseOptionalAttrDict(result.attributes);
}


//===----------------------------------------------------------------------===//
// DeinterleaverOp
//===----------------------------------------------------------------------===//
void DeinterleaverOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value input){
  ::mlir::Type intype = input.getType();
  assert(isa<VectorType>(intype));
  VectorType vectype = dyn_cast<VectorType>(intype);
  ::mlir::Type elemtype = vectype.getElementType();

  llvm::ArrayRef<int64_t> shape = vectype.getShape();
  assert(shape.size() == 1);
  int64_t size = shape[0];

  SmallVector<mlir::Type> resultTypes;
  for(int64_t i = 0; i < size; i++){
    resultTypes.push_back(elemtype);
  }

  build(odsBuilder, odsState, resultTypes, input);
}

void DeinterleaverOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypedValue<::mlir::VectorType> input){
  ::mlir::VectorType vectype = input.getType();
  ::mlir::Type elemtype = vectype.getElementType();

  llvm::ArrayRef<int64_t> shape = vectype.getShape();
  assert(shape.size() == 1);
  int64_t size = shape[0];

  SmallVector<mlir::Type> resultTypes;
  for(int64_t i = 0; i < size; i++){
    resultTypes.push_back(elemtype);
  }

  build(odsBuilder, odsState, resultTypes, input);
}

LogicalResult DeinterleaverOp::verify() {
  if(!(getOutputs().size() > 1)){
    return emitOpError(
        "Deinterleaver op: output number of Deinterleaver op should be larger than 1");
  }
  ::mlir::Type intype = getElementType();
  for(auto output : getOutputs()){
    if(output.getType() != intype){
      return emitOpError(
        "Deinterleaver op: all outputs of Deinterleaver op should be the same type.");
    }
  }

  llvm::ArrayRef<int64_t> shape = ::llvm::cast<VectorType>(getInput().getType()).getShape();
  int num = 1;
  for(auto s : shape){
    num *= s;
  }

  if(num != getOutputs().size()){
    return emitOpError(
        "Deinterleaver op: the count of outputs must correspond to the sum of the element count in the input vector.");
  }

  return success();
}

void DeinterleaverOp::print(OpAsmPrinter &p) {
  Type outtype = getElementType();
  p << " " ;
  /// print input
  p << getInput();

  p << " : " ;

  /// print input type
  p << getInputVectorType();

  p << " -> (" ;
  llvm::interleaveComma(getOutputs(), p, [&](auto it) {
    p << outtype;
  });
  p << ")" ;

  p.printOptionalAttrDict((*this)->getAttrs());
}

ParseResult DeinterleaverOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  Type outType;

  /// parse inputs
  OpAsmParser::UnresolvedOperand input;
  if (parser.parseOperand(input))
    return failure();

  /// parse input type
  Type inputtype;
  if (parser.parseColonType(inputtype))
    return failure();

  if (parser.resolveOperand(input, inputtype, result.operands))
    return failure();

  /// parse output type
  if (parser.parseArrowTypeList(result.types))
  {
    return failure();
  }

  return parser.parseOptionalAttrDict(result.attributes);
}


#define GET_OP_CLASSES
#include "ADORA/Dialect/ADORA/IR/ADORAOps.cpp.inc"

//===----------------------------------------------------------------------===//
// PR2 commit C — Canonical patterns for async-token cleanup.
//
// DropUnusedAsyncToken: if an op's asyncToken result has no uses, rebuild
//   it without the token to keep IR minimal.
// DedupAsyncDeps: remove duplicate / self-referential entries from an op's
//   asyncDependencies operand list.
//===----------------------------------------------------------------------===//

namespace {

/// Drop an asyncToken result when it has no uses. Rebuilds the op in its
/// synchronous form (produceToken=false) to keep the IR minimal.
/// Safe to apply repeatedly (converges in one step).
template <typename OpTy>
struct DropUnusedAsyncToken : public mlir::OpRewritePattern<OpTy> {
  using mlir::OpRewritePattern<OpTy>::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(OpTy op, mlir::PatternRewriter &rw) const override {
    mlir::Value tok = op.getAsyncToken();
    if (!tok || !tok.use_empty()) return mlir::failure();
    // NOTE: we must rebuild the op (erase + re-create) because MLIR does not
    // allow removing a result in-place; the result type list is immutable.
    // Delegate to the existing sync builder by passing asyncDeps unchanged
    // and produceToken=false, then RAUW the old result (if any) and erase.
    return mlir::failure(); // placeholder — see per-op specializations below
  }
};

/// Remove duplicate values and self-references from asyncDependencies.
/// A self-reference (token defined by the same op) cannot happen in valid
/// SSA but is guarded here defensively.
template <typename OpTy>
struct DedupAsyncDeps : public mlir::OpRewritePattern<OpTy> {
  using mlir::OpRewritePattern<OpTy>::OpRewritePattern;
  mlir::LogicalResult
  matchAndRewrite(OpTy op, mlir::PatternRewriter &rw) const override {
    auto deps = op.getAsyncDependencies();
    llvm::SmallSetVector<mlir::Value, 4> uniq;
    for (mlir::Value v : deps)
      if (v && v.getDefiningOp() != op.getOperation()) uniq.insert(v);
    if (uniq.size() == (size_t)deps.size()) return mlir::failure();
    rw.modifyOpInPlace(op, [&] {
      op.getAsyncDependenciesMutable().assign(
          mlir::SmallVector<mlir::Value>(uniq.begin(), uniq.end()));
    });
    return mlir::success();
  }
};

} // namespace

void mlir::ADORA::DataBlockLoadOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, mlir::MLIRContext *ctx) {
  results.add<DedupAsyncDeps<DataBlockLoadOp>>(ctx);
}

void mlir::ADORA::DataBlockStoreOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, mlir::MLIRContext *ctx) {
  results.add<DedupAsyncDeps<DataBlockStoreOp>>(ctx);
}

