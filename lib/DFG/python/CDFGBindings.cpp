//===----------------------------------------------------------------------===//
// Python bindings that expose the ADORA CDFG to the cycle-estimator, replacing
// the dot-file + regex round-trip with direct structured access.
//===----------------------------------------------------------------------===//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <map>
#include <string>
#include <vector>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "../inc/mlir_cdfg.h"
#include "ADORA/Misc/DFG.h"

namespace py = pybind11;
using namespace mlir;

namespace {

// Plain-old-data mirror of an LLVMCDFGNode, decoupled from the C++ object so the
// Python side never holds a dangling pointer after the CDFG is freed.
struct PyNode {
  int id;
  std::string opcode;     // LLVMCDFGNode::getTypeName()
  std::string name;       // LLVMCDFGNode::getName()
  int loop_level;
  int data_bits;
  int memref_size;
  std::string memref_name;
  bool ls_affine;
};

struct PyEdge {
  int id;
  int src;
  int dst;
  std::string type;       // EdgeType name
  int iter_dist;
};

struct PyKernel {
  std::string name;
  std::vector<PyNode> nodes;
  std::vector<PyEdge> edges;
  std::vector<int> loop_counts;            // per-level trip counts
  std::vector<std::pair<int, int>> loop_bounds;  // (left, right)
  std::vector<int> loop_strides;
};

std::string edgeTypeName(EdgeType t) {
  switch (t) {
  case EDGE_TYPE_DATA: return "DATA";
  case EDGE_TYPE_CTRL: return "CONTROL";
  case EDGE_TYPE_MEM:  return "MEMORY";
  default:             return "UNKNOWN";
  }
}

PyKernel buildPyKernel(const std::string &kernelName, LLVMCDFG *cdfg) {
  PyKernel pk;
  pk.name = kernelName;

  for (auto &kv : cdfg->nodes()) {
    LLVMCDFGNode *n = kv.second;
    PyNode pn;
    pn.id = n->id();
    pn.opcode = n->getTypeName();
    pn.name = n->getName();
    pn.loop_level = n->getLoopLevel();
    pn.data_bits = n->dataBits();
    pn.memref_size = n->getMemrefSize();
    pn.memref_name = n->getMemrefName();
    pn.ls_affine = n->isLSaffine();
    pk.nodes.push_back(std::move(pn));
  }

  for (auto &kv : cdfg->edges()) {
    LLVMCDFGEdge *e = kv.second;
    PyEdge pe;
    pe.id = e->id();
    pe.src = e->src() ? e->src()->id() : -1;
    pe.dst = e->dst() ? e->dst()->id() : -1;
    pe.type = edgeTypeName(e->type());
    pe.iter_dist = e->IterDist();
    pk.edges.push_back(std::move(pe));
  }

  for (auto &kv : cdfg->getLoopsAffineCounts())
    pk.loop_counts.push_back(kv.second);
  for (auto &kv : cdfg->getLoopsAffineBounds())
    pk.loop_bounds.emplace_back(kv.second.first, kv.second.second);
  for (auto &kv : cdfg->getLoopsAffineStrides())
    pk.loop_strides.push_back(kv.second);

  return pk;
}

std::vector<PyKernel> parseKernels(const std::string &mlirPath,
                                   const std::string &opNameFile) {
  DialectRegistry registry;
  registry.insert<ADORA::ADORADialect, arith::ArithDialect,
                  affine::AffineDialect, func::FuncDialect, scf::SCFDialect,
                  memref::MemRefDialect, math::MathDialect>();
  MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();

  OwningOpRef<ModuleOp> module =
      parseSourceFile<ModuleOp>(mlirPath, &ctx);
  if (!module)
    throw std::runtime_error("failed to parse MLIR file: " + mlirPath);

  std::vector<PyKernel> kernels;
  int kernelCnt = 0;
  module->walk([&](ADORA::KernelOp kernel) {
    std::string kernelName = kernel.getKernelName();
    if (kernelName.empty())
      kernelName = "kernel_" + std::to_string(kernelCnt);
    LLVMCDFG *cdfg = new LLVMCDFG(kernelName, opNameFile);
    generateCDFGfromKernel(cdfg, kernel, /*verbose=*/false);
    kernels.push_back(buildPyKernel(kernelName, cdfg));
    delete cdfg;
    kernelCnt++;
  });

  return kernels;
}

} // namespace

PYBIND11_MODULE(_adora_cdfg, m) {
  m.doc() = "Direct access to ADORA CDFG for the cycle estimator";

  py::class_<PyNode>(m, "Node")
      .def_readonly("id", &PyNode::id)
      .def_readonly("opcode", &PyNode::opcode)
      .def_readonly("name", &PyNode::name)
      .def_readonly("loop_level", &PyNode::loop_level)
      .def_readonly("data_bits", &PyNode::data_bits)
      .def_readonly("memref_size", &PyNode::memref_size)
      .def_readonly("memref_name", &PyNode::memref_name)
      .def_readonly("ls_affine", &PyNode::ls_affine);

  py::class_<PyEdge>(m, "Edge")
      .def_readonly("id", &PyEdge::id)
      .def_readonly("src", &PyEdge::src)
      .def_readonly("dst", &PyEdge::dst)
      .def_readonly("type", &PyEdge::type)
      .def_readonly("iter_dist", &PyEdge::iter_dist);

  py::class_<PyKernel>(m, "Kernel")
      .def_readonly("name", &PyKernel::name)
      .def_readonly("nodes", &PyKernel::nodes)
      .def_readonly("edges", &PyKernel::edges)
      .def_readonly("loop_counts", &PyKernel::loop_counts)
      .def_readonly("loop_bounds", &PyKernel::loop_bounds)
      .def_readonly("loop_strides", &PyKernel::loop_strides);

  m.def("parse_kernels", &parseKernels, py::arg("mlir_path"),
        py::arg("op_name_file"),
        "Parse an ADORA-dialect MLIR file and return the CDFG of each kernel");
}
