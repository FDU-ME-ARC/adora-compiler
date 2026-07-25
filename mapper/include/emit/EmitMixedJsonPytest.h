#ifndef ADORA_EMIT_MIXED_JSON_PYTEST_H
#define ADORA_EMIT_MIXED_JSON_PYTEST_H

#include <string>
#include "llvm/Support/raw_ostream.h"
#include "mapper/mapping.h"

// Emit a standalone pytest/cocotb helper for the --dfg-input JSON path.
// Unlike PytestEmitter, this emitter does not require an MLIR ModuleOp.
class MixedJsonPytestEmitter {
public:
    void emit(llvm::raw_ostream& os, Mapping* mapping,
              const std::string& designName);
};

#endif
