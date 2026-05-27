import os
import lit.formats

config.name = "ADORA"
config.test_format = lit.formats.ShTest(True)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)

# ---- Find tool dirs injected by lit.site.cfg.py ----
adora_tools_dir = getattr(config, "adora_tools_dir", None)
llvm_tools_dir  = getattr(config, "llvm_tools_dir", None)

# ---- Update PATH so tools can be found ----
paths = []
if adora_tools_dir:
  paths.append(adora_tools_dir)
if llvm_tools_dir:
  paths.append(llvm_tools_dir)

config.environment["PATH"] = os.pathsep.join(paths + [config.environment.get("PATH", "")])

# ---- DFG opcode-name table (consumed via getenv in include/ADORA/Misc/DFG.h) ----
_adora_src_root = os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir))
_op_name_file = os.path.join(_adora_src_root, "lib", "DFG", "Documents", "GeneralOpName.txt")
if os.path.isfile(_op_name_file):
    config.environment["GeneralOpNameFile"] = _op_name_file

# ---- Substitutions used by RUN lines ----
if adora_tools_dir:
  config.substitutions.append(("%cgra-opt", os.path.join(adora_tools_dir, "cgra-opt")))
  config.substitutions.append(("%cgra-mapper", os.path.join(adora_tools_dir, "cgra-mapper")))
  config.substitutions.append(("%adoracc", os.path.join(adora_tools_dir, "adoracc.py")))
  config.substitutions.append(("%tensor-opt", os.path.join(adora_tools_dir, "tensor-opt")))

if llvm_tools_dir:
  config.substitutions.append(("%FileCheck", os.path.join(llvm_tools_dir, "FileCheck")))
