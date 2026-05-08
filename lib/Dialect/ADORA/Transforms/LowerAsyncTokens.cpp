//===- LowerAsyncTokens.cpp - Lower !ADORA.token to event ops --------===//
//
// PR3 commit B.
//
// Converts SSA !ADORA.token def-use chains (emitted by adora-schedule-tasks
// with emit-token=true) into explicit runtime sync ops:
//
//   Before:
//     %tok = ADORA.BlockLoad ... -> (memref<...>, !ADORA.token)
//     ADORA.BlockStore ... async [%tok]
//
//   After:
//     %e   = ADORA.event.create -> !ADORA.token
//     %res = ADORA.BlockLoad ... (sync form)
//     ADORA.signal %e on stream 0 : !ADORA.token
//     ADORA.wait   %e on stream 0 : !ADORA.token
//     ADORA.BlockStore ... (sync form)
//     ADORA.event.destroy %e : !ADORA.token
//
// The rebuilt sync ops are identical to the pre-PR2 form; downstream passes
// that do not understand !ADORA.token are unaffected.
//===----------------------------------------------------------------------===//

#include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "ADORA/Dialect/ADORA/Transforms/Passes.h"
#include "ADORA/Dialect/ADORA/Utility/Utility.h"
#include "./PassDetail.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::ADORA;

#define PASS_NAME  "adora-lower-async-tokens"
#define DEBUG_TYPE "adora-lower-async-tokens"

namespace {

struct LowerAsyncTokensPass
    : public LowerAsyncTokensBase<LowerAsyncTokensPass> {
public:
  void runOnOperation() override;
};

// ---- helpers ---------------------------------------------------------------

/// Insert adora.event.create before `insertBefore`.
static Value emitEventCreate(OpBuilder &b, Location loc,
                             Operation *insertBefore) {
  b.setInsertionPoint(insertBefore);
  auto tokTy = TokenType::get(b.getContext());
  return b.create<EventCreateOp>(loc, tokTy).getToken();
}

/// Insert adora.signal after `insertAfter`.
static void emitSignal(OpBuilder &b, Location loc,
                       Operation *insertAfter, Value event, int32_t stream) {
  b.setInsertionPointAfter(insertAfter);
  b.create<SignalOp>(loc, event, b.getI32IntegerAttr(stream));
}

/// Insert adora.wait before `insertBefore`.
static void emitWait(OpBuilder &b, Location loc,
                     Operation *insertBefore, Value event, int32_t stream) {
  b.setInsertionPoint(insertBefore);
  b.create<WaitOp>(loc, event, b.getI32IntegerAttr(stream));
}

/// Insert adora.event.destroy after `insertAfter`.
static void emitDestroy(OpBuilder &b, Location loc,
                        Operation *insertAfter, Value event) {
  b.setInsertionPointAfter(insertAfter);
  b.create<EventDestroyOp>(loc, event);
}

// ---- rebuild helpers (re-use PR2 logic: drop asyncDeps + asyncToken) -------

static Operation *rebuildLoadSync(DataBlockLoadOp old) {
  OpBuilder b(old);
  auto mapAttr = old->getAttrOfType<AffineMapAttr>(
      DataBlockLoadOp::getMapAttrStr());
  AffineMap mapVal = mapAttr
      ? mapAttr.getValue()
      : AffineMap::getMultiDimIdentityMap(
            old.getResult().getType().cast<MemRefType>().getRank(),
            b.getContext());
  auto strides = old->getAttrOfType<DenseI64ArrayAttr>("strides");
  if (!strides) strides = DenseI64ArrayAttr::get(b.getContext(), {});
  std::string kern = old.getKernelName().str();
  SmallVector<Value> mapOps(old.getIndices().begin(), old.getIndices().end());

  auto newOp = b.create<DataBlockLoadOp>(
      old.getLoc(), old.getOriginalMemref(), mapVal, mapOps,
      old.getResult().getType().cast<MemRefType>(), strides, kern,
      /*asyncDeps=*/ValueRange{}, /*produceToken=*/false);

  for (NamedAttribute a : old->getAttrs()) {
    StringRef n = a.getName();
    if (n == DataBlockLoadOp::getMapAttrStr() || n == "strides" ||
        n == DataBlockLoadOp::getKernelNameAttrStr() ||
        n == "operandSegmentSizes") continue;
    newOp->setAttr(a.getName(), a.getValue());
  }
  old.getResult().replaceAllUsesWith(newOp.getResult());
  // Replace any remaining uses of the async token (result 1) with a null
  // placeholder; by this point all token consumers have been rebuilt to sync
  // form so there should be no uses left, but guard against stale uses to
  // avoid a "still has uses" assertion on erase.
  if (Value asyncTok = old.getAsyncToken())
    asyncTok.dropAllUses();
  old.erase();
  return newOp.getOperation();
}

static Operation *rebuildStoreSync(DataBlockStoreOp old) {
  OpBuilder b(old);
  auto mapAttr = old->getAttrOfType<AffineMapAttr>(
      DataBlockStoreOp::getMapAttrStr());
  AffineMap mapVal = mapAttr
      ? mapAttr.getValue()
      : AffineMap::getMultiDimIdentityMap(
            old.getSourceMemref().getType().cast<MemRefType>().getRank(),
            b.getContext());
  auto strides = old->getAttrOfType<DenseI64ArrayAttr>("strides");
  if (!strides) strides = DenseI64ArrayAttr::get(b.getContext(), {});
  std::string kern = old.getKernelName().str();
  SmallVector<Value> mapOps(old.getIndices().begin(), old.getIndices().end());

  auto newOp = b.create<DataBlockStoreOp>(
      old.getLoc(), old.getSourceMemref(), old.getTargetMemref(),
      mapVal, mapOps, strides, kern,
      /*asyncDeps=*/ValueRange{}, /*produceToken=*/false);

  for (NamedAttribute a : old->getAttrs()) {
    StringRef n = a.getName();
    if (n == DataBlockStoreOp::getMapAttrStr() || n == "strides" ||
        n == DataBlockStoreOp::getKernelNameAttrStr() ||
        n == "operandSegmentSizes") continue;
    newOp->setAttr(a.getName(), a.getValue());
  }
  old.erase();
  return newOp.getOperation();
}

static Operation *rebuildKernelSync(KernelOp old) {
  OpBuilder b(old);
  std::string name;
  if (auto kn = old->getAttrOfType<StringAttr>(KernelOp::getKernelNameAttrStr()))
    name = kn.getValue().str();
  auto newOp = b.create<KernelOp>(old.getLoc(), name,
                                  /*asyncDeps=*/ValueRange{},
                                  /*produceToken=*/false,
                                  &old.getBody());
  static const std::set<std::string> kSkip = {"kernel_name", "operandSegmentSizes"};
  for (NamedAttribute a : old->getAttrs())
    if (kSkip.find(a.getName().str()) == kSkip.end())
      newOp->setAttr(a.getName(), a.getValue());
  old.erase();
  return newOp.getOperation();
}

// ---- main pass logic -------------------------------------------------------

void LowerAsyncTokensPass::runOnOperation() {
  func::FuncOp func = getOperation();
  OpBuilder b(func.getContext());
  int32_t stream = defaultStream;

  // Collect all async-capable ops that have tokens in lexical order.
  // We process producers first (emit create+signal), then consumers (emit
  // wait), then rebuild all to sync form.

  // Map: tokenValue -> eventValue (after create is inserted)
  llvm::DenseMap<Value, Value> tokenToEvent;

  // Pass 1: for every producer (op with asyncToken), insert event.create
  //         before and event.signal after; record mapping.
  func.walk([&](Operation *op) {
    if (!isAsyncCapable(op)) return;
    Value tok = getAsyncTokenOrNull(op);
    if (!tok) return;
    Value ev = emitEventCreate(b, op->getLoc(), op);
    // signal inserted after op; but we haven't rebuilt yet so op still valid
    emitSignal(b, op->getLoc(), op, ev, stream);
    tokenToEvent[tok] = ev;
  });

  if (dropTokensOnly) {
    // Just rebuild all async ops to sync form without emitting event ops.
    SmallVector<Operation *> toRebuild;
    func.walk([&](Operation *op) {
      if (isAsyncCapable(op) &&
          (getAsyncTokenOrNull(op) || !getAsyncDeps(op).empty()))
        toRebuild.push_back(op);
    });
    for (Operation *op : toRebuild) {
      if (auto l = dyn_cast<DataBlockLoadOp>(op))  { rebuildLoadSync(l); continue; }
      if (auto s = dyn_cast<DataBlockStoreOp>(op)) { rebuildStoreSync(s); continue; }
      if (auto k = dyn_cast<KernelOp>(op))         { rebuildKernelSync(k); continue; }
    }
    return;
  }

  // Pass 2: for every consumer (op with asyncDeps), insert wait(s) before it.
  // NOTE: we insert waits BEFORE rebuild so we can still read getAsyncDeps.
  func.walk([&](Operation *op) {
    if (!isAsyncCapable(op)) return;
    auto deps = getAsyncDeps(op);
    if (deps.empty()) return;
    // Deduplicate events across deps.
    llvm::SmallSetVector<Value, 4> eventsToWait;
    for (Value d : deps) {
      auto it = tokenToEvent.find(d);
      if (it != tokenToEvent.end()) eventsToWait.insert(it->second);
    }
    for (Value ev : eventsToWait)
      emitWait(b, op->getLoc(), op, ev, stream);
  });

  // Pass 3: rebuild all async ops to their sync form.
  SmallVector<Operation *> toRebuild;
  func.walk([&](Operation *op) {
    if (isAsyncCapable(op) &&
        (getAsyncTokenOrNull(op) || !getAsyncDeps(op).empty()))
      toRebuild.push_back(op);
  });
  // Process in reverse to avoid iterator invalidation when erasing.
  for (Operation *op : llvm::reverse(toRebuild)) {
    if (auto l = dyn_cast<DataBlockLoadOp>(op))  { rebuildLoadSync(l); continue; }
    if (auto s = dyn_cast<DataBlockStoreOp>(op)) { rebuildStoreSync(s); continue; }
    if (auto k = dyn_cast<KernelOp>(op))         { rebuildKernelSync(k); continue; }
  }

  // Pass 4: insert event.destroy after last use of each event.
  // Walk events in creation order (by their defining op position).
  for (auto &kv : tokenToEvent) {
    Value ev = kv.second;
    // Find the lexically last op in the func that uses this event.
    Operation *lastUse = ev.getDefiningOp(); // at worst destroy right after create
    for (Operation *user : ev.getUsers()) {
      if (lastUse->getBlock() == user->getBlock() &&
          lastUse->isBeforeInBlock(user))
        lastUse = user;
    }
    emitDestroy(b, lastUse->getLoc(), lastUse, ev);
  }
}

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::ADORA::createLowerAsyncTokensPass() {
  return std::make_unique<LowerAsyncTokensPass>();
}
