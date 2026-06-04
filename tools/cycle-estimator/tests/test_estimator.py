"""Unit tests locking in the cycle-estimator core logic.

Run: python3 -m unittest discover -s tests   (from project root)
No external deps -- stdlib unittest only.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dot_parser import parse_dot
from core.latency_table import op_latency, DIV_LATENCY
from core.ii_model import estimate_ii, rec_mii, res_mii
from core.cycle_model import (estimate_cycles, critical_path_latency,
                              io_bytes_from_dot, config_overhead, dma_cycles)
from extract.loop_info import parse_kernels, kernel_for_dot
from scripts.cfgnum import parse_cfgnums

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
GEMM_DOT = os.path.join(FIX, "gemm_1_0_CDFG.dot")
GEMM_MLIR = ("/data00/home/loujiahang/adora/adora-compiler/experiment/"
             "demo/gemm/gemm/IR/0_kernels/gemm_kernel.mlir")


class TestLatencyTable(unittest.TestCase):
    def test_known_latencies(self):
        self.assertEqual(op_latency("ADD"), 1)
        self.assertEqual(op_latency("FMUL32"), 3)
        self.assertEqual(op_latency("FDIV32"), 7)
        self.assertEqual(op_latency("MAC"), 2)
        self.assertEqual(op_latency("LOAD"), 2)
        self.assertEqual(op_latency("STORE"), 1)

    def test_div_latency_constant(self):
        self.assertEqual(DIV_LATENCY, 6)
        self.assertEqual(op_latency("SDIV"), 6)

    def test_structural_zero(self):
        self.assertEqual(op_latency("for"), 0)
        self.assertEqual(op_latency("yield"), 0)

    def test_unknown_raises(self):
        with self.assertRaises(KeyError):
            op_latency("NOSUCHOP")


class TestDotParser(unittest.TestCase):
    def setUp(self):
        self.g = parse_dot(GEMM_DOT)

    def test_counts(self):
        self.assertEqual(len(self.g.nodes), 14)
        self.assertEqual(len(self.g.edges), 11)

    def test_backedge_has_iterdist(self):
        bes = self.g.backedges()
        self.assertEqual(len(bes), 1)
        self.assertEqual(bes[0].iterdist, 48)

    def test_compute_nodes(self):
        # 2x FMUL32 + 2x FADD32
        self.assertEqual(len(self.g.compute_nodes()), 4)

    def test_io_sizes_present(self):
        load, store = io_bytes_from_dot(self.g)
        self.assertEqual(load, 3220)   # 120 + 3000 + 100
        self.assertEqual(store, 100)


class TestIIModel(unittest.TestCase):
    def setUp(self):
        self.g = parse_dot(GEMM_DOT)

    def test_accumulator_ii_is_one(self):
        # iterdist=48 >> recurrence latency -> RecMII collapses to 1.
        self.assertEqual(rec_mii(self.g), 1)

    def test_resmii_small_dfg(self):
        # 4 compute nodes on 16 ALUs -> 1.
        self.assertEqual(res_mii(self.g, num_alus=16), 1)

    def test_resmii_scales_down_array(self):
        # 4 nodes on 2 ALUs -> ceil(4/2)=2.
        self.assertEqual(res_mii(self.g, num_alus=2), 2)

    def test_estimate_ii_takes_max(self):
        info = estimate_ii(self.g, num_alus=2)
        self.assertEqual(info["II"], max(info["RecMII"], info["ResMII"]))


class TestCycleModel(unittest.TestCase):
    def test_config_overhead(self):
        self.assertEqual(config_overhead(30), 10)
        self.assertEqual(config_overhead(None), 0)

    def test_dma_cycles_ceil(self):
        self.assertEqual(dma_cycles(3220, bw=4), 805)
        self.assertEqual(dma_cycles(1, bw=4), 1)
        self.assertEqual(dma_cycles(0), 0)

    def test_critical_path(self):
        g = parse_dot(GEMM_DOT)
        # longest op-latency path through 2 FMUL32 + 2 FADD32 chain + io.
        self.assertGreater(critical_path_latency(g), 0)

    def test_full_estimate_formula(self):
        g = parse_dot(GEMM_DOT)
        kernels = parse_kernels(GEMM_MLIR)
        loops = kernels["gemm_1"]

        # serial: total = cfg + load + compute + store
        ser = estimate_cycles(g, loops, overlap=False)
        compute = ser.outer_trip * (ser.II * ser.inner_trip + ser.drain)
        self.assertEqual(ser.total, ser.config + ser.load + compute + ser.store)

        # overlap (default): mem and compute run concurrently -> max()
        ov = estimate_cycles(g, loops)
        self.assertEqual(ov.total,
                         ov.config + max(ov.load + ov.store, compute))

        self.assertEqual(ov.inner_trip, 30)
        self.assertEqual(ov.outer_trip, 25)


class TestMlirLoopInfo(unittest.TestCase):
    def test_gemm_kernels(self):
        ks = parse_kernels(GEMM_MLIR)
        self.assertEqual(ks["gemm_1"].trip_counts, [25, 30])
        self.assertEqual(ks["gemm_1"].total_trip, 750)

    def test_dot_name_suffix_match(self):
        # 'gemm_1_0' should resolve back to MLIR kernel 'gemm_1'.
        k = kernel_for_dot(GEMM_MLIR, "gemm_1_0")
        self.assertIsNotNone(k)
        self.assertEqual(k.name, "gemm_1")


class TestCfgNum(unittest.TestCase):
    def test_parse_inline(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write('""" kernel: foo_kernel_0,  cfgNum: 42"""\n')
            path = f.name
        self.assertEqual(parse_cfgnums(path), {"foo_kernel_0": 42})
        os.unlink(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
