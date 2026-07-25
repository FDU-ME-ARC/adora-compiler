#ifndef __AUFORA_YOSYS_FRONTEND_H__
#define __AUFORA_YOSYS_FRONTEND_H__

#include <string>
#include "dfg/dfg.h"

// Fine-grained technology mapping bridge.  Coarse cells are emitted as
// preserved black boxes, while one-bit logic is left as Yosys internal cells
// and mapped to $lut cells by ABC.
class YosysFrontend {
public:
    static DFG* synthesize(
        DFG* input,
        const std::string& workDir,
        const std::string& designName,
        int maxLutInputs,
        const std::string& yosysExecutable,
        const std::string& cellLibrary);
};

#endif
