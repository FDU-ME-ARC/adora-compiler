//===----------------------------------------------------------------------===//
//
// This file implements automatic dataflow strategy decision for ADORA Tensor
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/OperationSupport.h"
// #include "mlir/IR/OpDefinition.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Support/LLVM.h"

#include <iostream>
#include <string>
#include <bit>
#include <math.h>
#include <vector>
#include <algorithm>
#include <optional>

// #include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

// For op transformation
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

#include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "ADORA/Dialect/ADORA/Utility/Utility.h"
#include "ADORA/Dialect/ADORATensor/IR/ADORATensor.h"
#include "PassDetail.h"

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::ADORA;
using namespace mlir::ADORA::ADORATensor;

#define DEBUG_TYPE "adora-tensor-op-strategy-decision"

namespace mlir
{
  namespace ADORA
  {
    namespace ADORATensor
    {

      class ADORAOpStrategyDecisionPass : public ADORAOpStrategyDecisionPassBase<ADORAOpStrategyDecisionPass>
      {
        void runOnOperation() override;
      };

      std::unique_ptr<OperationPass<ModuleOp>> createADORAOpStrategyDecisionPass()
      {
        return std::make_unique<ADORAOpStrategyDecisionPass>();
      }

      // ==============================================================================
      // 1. Helper Functions
      // ==============================================================================

      // ceiling division
      inline int64_t ceilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

      // element bytes from MLIR type (int/float); fallback 4B
      static int64_t getDataBytes(mlir::Type t)
      {
        if (isa<mlir::MemRefType>(t))
          return t.cast<mlir::MemRefType>().getElementTypeBitWidth() / 8;
        if (isa<mlir::RankedTensorType>(t))
          return t.cast<mlir::RankedTensorType>().getElementTypeBitWidth() / 8;
        return 4;
      }

      static ArrayRef<int64_t> getShape(mlir::Type t)
      {
        if (isa<mlir::MemRefType>(t))
          return t.cast<mlir::MemRefType>().getShape();
        if (isa<mlir::RankedTensorType>(t))
          return t.cast<mlir::RankedTensorType>().getShape();
        return {};
      }

      static inline llvm::SmallVector<int64_t, 2> trimTo2D(llvm::ArrayRef<int64_t> shape)
      {
        llvm::SmallVector<int64_t, 2> result;
        if (shape.size() >= 2)
        {
          result.push_back(shape[shape.size() - 2]);
          result.push_back(shape[shape.size() - 1]);
        }
        else if (shape.size() == 1)
        {
          // Handle [N] -> [1, N] to maintain 2D invariant
          result.push_back(1);
          result.push_back(shape[0]);
        }
        else
        {
          result.push_back(1);
          result.push_back(1);
        }
        return result;
      }

      static inline llvm::SmallVector<int64_t, 32> getAllDivisor(int64_t n)
      {
        llvm::SmallVector<int64_t, 32> divs;
        if (n <= 0)
          n = std::abs(n);
        if (n == 0)
        {
          divs.push_back(1);
          return divs;
        }

        int64_t r = static_cast<int64_t>(std::sqrt(static_cast<long double>(n)));
        for (int64_t d = 1; d <= r; ++d)
        {
          if (n % d == 0)
          {
            divs.push_back(d);
            if (n / d != d)
              divs.push_back(n / d);
          }
        }
        llvm::sort(divs.begin(), divs.end());
        return divs;
      }

      std::vector<std::pair<int, int>> findFeasibleSpatialMap(int num_pe, int num_iob, int limit = 80)
      {
        std::vector<std::pair<int, int>> cand;
        for (int r = 1; r <= num_iob; ++r)
        {
          for (int c = 1; c <= num_iob; ++c)
          {
            // Basic check: total PEs and boundary constraints
            if (r * c < num_pe && (r + 2 * c <= num_iob || 2 * r + c <= num_iob))
            {
              cand.emplace_back(r, c);
            }
          }
        }
        // Sort by area (utilization)
        std::sort(cand.begin(), cand.end(), [](auto a, auto b)
                  { 
                    double sa = a.first * a.second;
                    double sb = b.first * b.second;
                    return sa < sb; });

        if ((int)cand.size() > limit)
          cand.resize(limit);
        return cand;
      }

      bool legalRowCol(DataflowStrategy stationarykind, std::pair<int, int> sa, int num_pe, int num_io)
      {
        /////// check PE limits and IO
        int row = sa.first;
        int col = sa.second;
        int io_cost = 0;
        bool legal = false;

        switch (stationarykind)
        {
        case DataflowStrategy::InputStationary:
          io_cost = row * col / 4 + col + 2 * row;
          if (row * (2 + col) <= num_pe && io_cost <= num_io)
          {
            legal = true;
          }
          break;
        case DataflowStrategy::WeightStationary:
          io_cost = row * col / 4 + col * 2 + row;
          if (col * (2 + row) <= num_pe && io_cost <= num_io)
          {
            legal = true;
          }
          break;
        case DataflowStrategy::OutputStationary:
          io_cost = 2 * row * col / 4 + col + row;
          if (col * (2 + row) <= num_pe && io_cost <= num_io)
          {
            legal = true;
          }
          break;
        default:
          assert(false && "No defined DataflowStrategy");
          break;
        }
        return legal;
      }

      // choose which dims map to (row,col) under each strategy
      struct MapDims
      {
        int64_t Rmap, Cmap, innermostNonStationary, T0_dimension;
      };

      MapDims mappingFor(DataflowStrategy s, int64_t M, int64_t N, int64_t K)
      {
        switch (s)
        {
        case DataflowStrategy::InputStationary: // A is MxK; stream along N
          return {M, K, N, M};
        case DataflowStrategy::WeightStationary: // B is KxN; stream along M
          return {K, N, M, K};
        case DataflowStrategy::OutputStationary: // C is MxN; stream along K
          return {M, N, K, N};
        default:
          return {M, N, K, M};
        }
      }

      bool CheckTileDivided(DataflowStrategy s, int64_t M, int64_t N, int64_t K,
                            int64_t row, int64_t col, int64_t T0, int64_t T1)
      {
        switch (s)
        {
        case DataflowStrategy::InputStationary:
          return M % (T0 * row) == 0 && N % T1 == 0 && K % col == 0;
        case DataflowStrategy::WeightStationary:
          return M % T1 == 0 && N % col == 0 && K % (T0 * row) == 0;
        case DataflowStrategy::OutputStationary:
          return M % row == 0 && N % (T0 * col) == 0 && K % T1 == 0;
        default:
          abort();
        }
      }

      // ==============================================================================
      // 2. Cost Models
      // ==============================================================================

      // pipeline fill estimate
      inline int64_t pipeFill(int64_t row, int64_t col)
      {
        return std::max<int64_t>(0, (int)lround(2.0 * sqrt((double)row * col)));
      }

      // EC cycles with reconfig amortization
      int64_t execCycles(DataflowStrategy s, int64_t M, int64_t N, int64_t K,
                         int64_t row, int64_t col, int64_t T0, int64_t T1,
                         int numPEs, int numIOBs)
      {
        int64_t mac = (M * N * K) / std::max<int64_t>(1, row * col);
        int cgra_col = numIOBs / 2;
        int cgra_row = numPEs / std::max(1, cgra_col);
        int64_t S = pipeFill(cgra_row, cgra_col);

        long double reconfig = 3 * row + col + S;
        switch (s)
        {
        case DataflowStrategy::InputStationary:
          reconfig = 3 * row + col + S;
          break;
        case DataflowStrategy::WeightStationary:
          reconfig = 3 * col + row + S;
          break;
        case DataflowStrategy::OutputStationary:
          reconfig = 3 * col + row + S;
          break;
        default:
          reconfig = 3 * row + col + S;
          break;
        }
        // amortize by T0*T1
        return mac * (1 + (reconfig) / ((long double)(T0 * T1)));
      }

      // [Standard GEMM / Im2Col] Transfer Volume
      int64_t transferVolumn(DataflowStrategy s, int64_t M, int64_t N, int64_t K, int64_t row, int64_t col)
      {
        switch (s)
        {
        case DataflowStrategy::InputStationary:
          // Input(MK) + Weight(NK) + PartialSum
          return (int64_t)((long double)M * K * N * (1.0 / N + 1.0 / M + 1.0 / col));
        case DataflowStrategy::WeightStationary:
          return (int64_t)((long double)M * K * N * (1.0 / col + 1.0 / M + 1.0 / K));
        case DataflowStrategy::OutputStationary:
          return (int64_t)((long double)M * K * N * (1.0 / N + 1.0 / row + 1.0 / K));
        default:
          return M * K + K * N + M * N;
        }
      }

      // // Reduce input transfer volume by kernel_size factor
      // int64_t transferVolumnDirectConv(DataflowStrategy s, int64_t M, int64_t N, int64_t K,
      //                                  int64_t row, int64_t col, int64_t KH, int64_t KW)
      // {
      //   double kernel_size = (double)(KH * KW);
      //   if (kernel_size < 1.0)
      //     kernel_size = 1.0;

      //   switch (s)
      //   {
      //   case DataflowStrategy::InputStationary:
      //     // IS Formula: MNK(1/N + 1/M + 1/col)
      //     // 1/N term corresponds to Input Matrix (MK). Scale by 1/kernel_size
      //     return (int64_t)((long double)M * K * N * (1.0 / N / kernel_size + 1.0 / M + 1.0 / col));

      //   case DataflowStrategy::WeightStationary:
      //     // WS Formula: MNK(1/col + 1/M + 1/K)
      //     // 1/col term corresponds to Input Streaming. Scale by 1/kernel_size
      //     return (int64_t)((long double)M * K * N * (1.0 / col / kernel_size + 1.0 / M + 1.0 / K));

      //   case DataflowStrategy::OutputStationary:
      //     // OS Formula: MNK(1/N + 1/row + 1/K)
      //     // 1/N term corresponds to Input Matrix (MK). Scale by 1/kernel_size
      //     return (int64_t)((long double)M * K * N * (1.0 / N / kernel_size + 1.0 / row + 1.0 / K));

      //   default:
      //     return transferVolumn(s, M, N, K, row, col);
      //   }
      // }

      // ==============================================================================
      // 3. Search Solvers
      // ==============================================================================

      struct Cand
      {
        ComputeAlgorithm algorithm;
        DataflowStrategy stationaryKind;
        int64_t T0, T1, row, col;
        int64_t EC, TX, LAT;

        void dump() const
        {
          llvm::errs() << "[Cand] Algo=" << getComputeAlgorithmStrRef(algorithm)
                       << " Strategy=" << getDataflowStrategyStrRef(stationaryKind)
                       << " T0=" << T0 << " T1=" << T1
                       << " row=" << row << " col=" << col
                       << " | EC=" << EC << " TX=" << TX
                       << " LAT=" << LAT << "\n";
        }
      };

      // Generic DSE Solver (Shared by GEMM and Conv)
      static std::optional<Cand> solveDSEGemmLike(
          int64_t M, int64_t N, int64_t K, int64_t dByte,
          int num_pe, int num_iob, int bank_byte, int BandByteWidth,
          DataflowStrategy selected_strategy,
          ComputeAlgorithm selected_algo,
          int64_t KH = 0, int64_t KW = 0)
      {
        std::optional<Cand> best;
        auto pairs = findFeasibleSpatialMap(num_pe, num_iob, 30000);

        std::vector<DataflowStrategy> strategies;
        if (selected_strategy == DataflowStrategy::Undefine)
        {
          strategies.push_back(DataflowStrategy::InputStationary);
          strategies.push_back(DataflowStrategy::OutputStationary);
          strategies.push_back(DataflowStrategy::WeightStationary);
        }
        else
        {
          strategies.push_back(selected_strategy);
        }

        for (DataflowStrategy stationarykind : strategies)
        {
          auto map = mappingFor(stationarykind, M, N, K);
          for (auto [rRaw, cRaw] : pairs)
          {
            int64_t row = std::min<int64_t>(rRaw, map.Rmap);
            int64_t col = std::min<int64_t>(cRaw, map.Cmap);

            // Check hardware constraints
            if (row <= 0 || col <= 0 || !legalRowCol(stationarykind, std::make_pair(row, col), num_pe, num_iob))
              continue;

            auto T1_cands = getAllDivisor(map.innermostNonStationary);
            for (int T1 : T1_cands)
            {
              for (int T0 = 1; T0 <= map.T0_dimension; T0++)
              {
                // Memory check
                if (T0 * T1 * dByte > bank_byte / 2)
                  break;
                // Tiling divisibility check
                if (!CheckTileDivided(stationarykind, M, N, K, row, col, T0, T1))
                  continue;

                // Compute Metrics
                int64_t EC = execCycles(stationarykind, M, N, K, row, col, T0, T1, num_pe, num_iob);
                int64_t TV = transferVolumn(stationarykind, M, N, K, row, col) * dByte;

                int64_t TX = ceilDiv(TV, std::max(1, BandByteWidth));
                int64_t LAT = std::max<int64_t>(EC, TX);

                Cand cand{selected_algo, stationarykind, T0, T1, row, col, EC, TX, LAT};

                // Smart Selector logic
                auto better = [&](const Cand &a, const Cand &b)
                {
                  // 1. Latency dominant (allow 4% margin)
                  if ((double)std::abs(b.LAT - a.LAT) / (double)b.LAT <= 0.04)
                  {
                    // 2. If latency is similar, prefer larger T1 (better for inner loops)
                    return a.T1 >= b.T1;
                  }
                  // 3. Absolute Latency
                  if (a.LAT != b.LAT)
                    return a.LAT < b.LAT;
                  // 4. Transfer Time
                  if (a.TX != b.TX)
                    return a.TX < b.TX;
                  // 5. Compute Cycles
                  return a.EC < b.EC;
                };

                if (!best || better(cand, *best))
                  best = cand;
              }
            }
          }
        }
        return best;
      }

      // ==============================================================================
      // Precise Physical Cost Model for Direct Convolution
      // ==============================================================================
      static std::optional<Cand> solveDSEDirectConv(
          int64_t N_batch, int64_t IC, int64_t OH, int64_t OW, int64_t OC,
          int64_t KH, int64_t KW, int64_t stride_h, int64_t stride_w,
          int64_t dilation_h, int64_t dilation_w,
          int64_t dByte, int num_pe, int num_iob, int bank_byte, int BandByteWidth,
          DataflowStrategy selected_strategy)
      {
        std::optional<Cand> best;
        auto pairs = findFeasibleSpatialMap(num_pe, num_iob, 30000);

        DataflowStrategy stationarykind = DataflowStrategy::OutputStationary;
        if (selected_strategy != DataflowStrategy::Undefine)
        {
          stationarykind = selected_strategy;
        }

        for (auto [rRaw, cRaw] : pairs)
        {
          int64_t T_K = std::min<int64_t>(rRaw, OC);
          int64_t T_Q = std::min<int64_t>(cRaw, OW);

          if (T_K <= 0 || T_Q <= 0 || !legalRowCol(stationarykind, std::make_pair(T_K, T_Q), num_pe, num_iob))
            continue;

          auto TN_cands = getAllDivisor(N_batch);
          auto TP_cands = getAllDivisor(OH);

          for (int T_N : TN_cands)
          {
            for (int T_P : TP_cands)
            {

              // 1. Precise SRAM Capacity Model (Dilation Corrected)
              // Calculate the EFFECTIVE receptive field of the kernel in the input tensor
              int64_t effective_KH = dilation_h * (KH - 1) + 1;
              int64_t effective_KW = dilation_w * (KW - 1) + 1;

              // The physical input window height/width needed for the output tile
              int64_t T_H_in = (T_P - 1) * stride_h + effective_KH;
              int64_t T_W_in = (T_Q - 1) * stride_w + effective_KW;

              // X block size must account for the sparse but wide Dilation footprint in SPAD
              int64_t X_tile_size = T_N * IC * T_H_in * T_W_in;
              int64_t W_tile_size = T_K * IC * KH * KW; // Weights are dense in memory
              int64_t Y_tile_size = T_N * T_K * T_P * T_Q;

              int64_t total_sram_bytes = (X_tile_size + W_tile_size + Y_tile_size) * dByte;

              if (total_sram_bytes > bank_byte / 2)
              {
                continue;
              }

              // 2. Transfer Volume (TV)
              int64_t outer_iters = ceilDiv(N_batch, T_N) * ceilDiv(OC, T_K) * ceilDiv(OH, T_P) * ceilDiv(OW, T_Q);
              int64_t TV = outer_iters * (X_tile_size + W_tile_size + 2 * Y_tile_size) * dByte;
              int64_t TX = ceilDiv(TV, std::max(1, BandByteWidth));

              // 3. Execution Cycles (EC)
              int64_t mac_per_tile = T_N * T_P * IC * KH * KW;
              int cgra_col = num_iob / 2;
              int cgra_row = num_pe / std::max(1, cgra_col);
              int64_t S = pipeFill(cgra_row, cgra_col);
              int64_t reconfig = 3 * T_K + T_Q + S;

              int64_t EC = outer_iters * (mac_per_tile + reconfig);
              int64_t LAT = std::max<int64_t>(EC, TX);

              Cand cand{ComputeAlgorithm::Conv_Direct, stationarykind, T_N, T_K, T_P, T_Q, EC, TX, LAT};

              auto better = [&](const Cand &a, const Cand &b)
              {
                if ((double)std::abs(b.LAT - a.LAT) / (double)b.LAT <= 0.04)
                {
                  return a.TX < b.TX;
                }
                return a.LAT < b.LAT;
              };

              if (!best || better(cand, *best))
                best = cand;
            }
          }
        }
        return best;
      }

      // ==============================================================================
      // 4. Strategy Setting Functions
      // ==============================================================================

      void AutoSetGemmStrategy(
          mlir::ADORA::ADORATensor::GemmOp gemmop,
          int num_pe, int num_iob, int bank_byte, int BandByteWidth,
          DataflowStrategy _strategy = DataflowStrategy::Undefine)
      {
        auto A = gemmop.getA();
        auto B = gemmop.getB();
        // C is needed for shape check in original code, though unused in logic
        auto C = gemmop.getC();

        int dByte = getDataBytes(A.getType());
        auto shapeA = trimTo2D(getShape(A.getType()));
        auto shapeB = trimTo2D(getShape(B.getType()));
        auto shapeC = trimTo2D(getShape(C.getType()));

        // Original shape sanity check logic
        if (shapeC.size() == 1)
        {
          llvm::SmallVector<int64_t, 2> newShapeC;
          newShapeC.push_back(shapeA[0]);
          newShapeC.push_back(shapeC[0]);
          shapeC = newShapeC;
        }

        int64_t M = shapeA[0];
        int64_t K = shapeA[1];
        int64_t N = shapeB[1];
        assert(shapeC[0] == M && shapeC[1] == N && "C shape mismatch with A/B");

        // GEMM defaults to Standard algorithm
        auto best = solveDSEGemmLike(M, N, K, dByte, num_pe, num_iob, bank_byte, BandByteWidth,
                                     _strategy, ComputeAlgorithm::GEMM_Standard);

        if (best)
        {
          const auto &ch = *best;
          ADORATensor::SystolicImplInterface Sinterface(gemmop);
          Sinterface.setAlgorithm(ch.algorithm);
          Sinterface.setStationaryKind(ch.stationaryKind);
          Sinterface.setTileSize(ArrayRef<int64_t>({ch.T0, ch.T1, ch.row, ch.col}));
          llvm::errs() << "Best GEMM strategy: ";
          ch.dump();
        }
        else
        {
          gemmop.emitError() << "[AutoSetGemmStrategy] No strategy found.";
        }
      }

      void AutoSetConvStrategy(
          mlir::ADORA::ADORATensor::ConvOp convOp,
          int num_pe, int num_iob, int bank_byte, int BandByteWidth,
          DataflowStrategy _strategy = DataflowStrategy::Undefine,
          ComputeAlgorithm _force_algo = ComputeAlgorithm::Undefine)
      {
        auto inputVal = convOp.getX();
        auto weightVal = convOp.getW();

        int dByte = getDataBytes(inputVal.getType());
        auto inputShape = getShape(inputVal.getType());   // [N, IC, H, W]
        auto weightShape = getShape(weightVal.getType()); // [OC, IC, KH, KW]

        if (inputShape.size() < 4 || weightShape.size() < 4)
        {
          convOp.emitError() << "[AutoSetConvStrategy] Expected 4D input/weight, got rank "
                             << inputShape.size() << " and " << weightShape.size();
          return;
        }

        int64_t N_batch = inputShape[0];
        int64_t IC = inputShape[1];
        int64_t H = inputShape[2];
        int64_t W = inputShape[3];

        int64_t OC = weightShape[0];
        int64_t KH = weightShape[2];
        int64_t KW = weightShape[3];

        int64_t stride_h = 1, stride_w = 1;
        int64_t pad_h_total = 0, pad_w_total = 0;
        int64_t dilation_h = 1, dilation_w = 1;

        if (auto stridesAttr = convOp.getStridesAttr())
        {
          auto vals = stridesAttr.getValue();
          if (vals.size() >= 2)
          {
            stride_h = vals[0].cast<IntegerAttr>().getInt();
            stride_w = vals[1].cast<IntegerAttr>().getInt();
          }
        }

        if (auto dilationsAttr = convOp.getDilationsAttr())
        {
          auto vals = dilationsAttr.getValue();
          if (vals.size() >= 2)
          {
            dilation_h = vals[0].cast<IntegerAttr>().getInt();
            dilation_w = vals[1].cast<IntegerAttr>().getInt();
          }
        }

        if (auto padsAttr = convOp.getPadsAttr())
        {
          auto vals = padsAttr.getValue();
          if (vals.size() == 4)
          {
            pad_h_total = vals[0].cast<IntegerAttr>().getInt() + vals[2].cast<IntegerAttr>().getInt();
            pad_w_total = vals[1].cast<IntegerAttr>().getInt() + vals[3].cast<IntegerAttr>().getInt();
          }
        }

        int64_t effective_KH = dilation_h * (KH - 1) + 1;
        int64_t effective_KW = dilation_w * (KW - 1) + 1;

        int64_t OH = (H + pad_h_total - effective_KH) / stride_h + 1;
        int64_t OW = (W + pad_w_total - effective_KW) / stride_w + 1;

        // Dimensions mapped for Im2Col (Virtual GEMM)
        int64_t M_gemm = N_batch * OH * OW;
        int64_t K_gemm = IC * KH * KW;
        int64_t N_gemm = OC;

        std::optional<Cand> bestGlobal;
        std::optional<Cand> bestIm2Col;
        std::optional<Cand> bestDirect;

        // Fast-path heuristic: 1x1 convolutions have NO memory amplification.
        // Im2Col completely out-performs Direct Conv due to perfect matrix multiplication structure.
        bool is_1x1_conv = (KH == 1 && KW == 1 && pad_h_total == 0 && pad_w_total == 0);

        if (_force_algo != ComputeAlgorithm::Undefine)
        {
          if (_force_algo == ComputeAlgorithm::Conv_Im2Col)
          {
            bestGlobal = solveDSEGemmLike(M_gemm, N_gemm, K_gemm, dByte, num_pe, num_iob, bank_byte, BandByteWidth, _strategy, ComputeAlgorithm::Conv_Im2Col);
          }
          else if (_force_algo == ComputeAlgorithm::Conv_Direct)
          {
            bestGlobal = solveDSEDirectConv(N_batch, IC, OH, OW, OC, KH, KW, stride_h, stride_w, dilation_h, dilation_w, dByte, num_pe, num_iob, bank_byte, BandByteWidth, _strategy);
          }
        }
        else if (is_1x1_conv)
        {
          bestGlobal = solveDSEGemmLike(M_gemm, N_gemm, K_gemm, dByte, num_pe, num_iob, bank_byte, BandByteWidth, _strategy, ComputeAlgorithm::Conv_Im2Col);
        }
        else
        {
          bestIm2Col = solveDSEGemmLike(M_gemm, N_gemm, K_gemm, dByte, num_pe, num_iob, bank_byte, BandByteWidth, _strategy, ComputeAlgorithm::Conv_Im2Col);

          bestDirect = solveDSEDirectConv(N_batch, IC, OH, OW, OC, KH, KW, stride_h, stride_w, dilation_h, dilation_w, dByte, num_pe, num_iob, bank_byte, BandByteWidth, _strategy);

          // Final Decision: Duel between Im2Col and DirectConv
          if (bestIm2Col && bestDirect)
          {
            if ((double)std::abs(bestIm2Col->LAT - bestDirect->LAT) / (double)std::max(bestIm2Col->LAT, bestDirect->LAT) <= 0.04)
            {
              bestGlobal = (bestIm2Col->TX <= bestDirect->TX) ? bestIm2Col : bestDirect;
            }
            else
            {
              bestGlobal = (bestIm2Col->LAT < bestDirect->LAT) ? bestIm2Col : bestDirect;
            }
          }
          else if (bestIm2Col)
          {
            bestGlobal = bestIm2Col;
          }
          else if (bestDirect)
          {
            bestGlobal = bestDirect;
          }
        }

        if (bestGlobal)
        {
          const auto &ch = *bestGlobal;
          ADORATensor::SystolicImplInterface Sinterface(convOp);
          Sinterface.setAlgorithm(ch.algorithm);
          Sinterface.setStationaryKind(ch.stationaryKind);

          // The Tile Size attributes seamlessly bridge to the Lowering passes
          // For Im2Col: T0, T1, row, col maps to GEMM's spatial/temporal bounds
          // For Direct: T0->T_N, T1->T_K, row->T_P, col->T_Q perfectly aligning with GenericDirectConv
          Sinterface.setTileSize(ArrayRef<int64_t>({ch.T0, ch.T1, ch.row, ch.col}));

          llvm::errs() << "[DSE] Best Conv strategy won by: " << (ch.algorithm == ComputeAlgorithm::Conv_Direct ? "Direct Conv" : "Im2Col") << "\n";
          ch.dump();
        }
        else
        {
          convOp.emitError() << "[AutoSetConvStrategy] No feasible strategy found."
                             << " M=" << M_gemm << " N=" << N_gemm << " K=" << K_gemm
                             << " PE=" << num_pe << " IOB=" << num_iob
                             << " SRAM=" << bank_byte;
        }
      }

      // ==============================================================================
      // 5. Pass Entry
      // ==============================================================================

      void ADORAOpStrategyDecisionPass::runOnOperation()
      {
        mlir::ModuleOp m = getOperation();
        unsigned NumGPE = 0, NumIOB = 0, bankBytes = 0;
        unsigned bandByteWidth = _bus_byte_band_width;

        if (_adg_fn != "-")
        {
          NumGPE = getInstanceNumFromADG(_adg_fn, "GPE");
          NumIOB = getInstanceNumFromADG(_adg_fn, "IOB");
          bankBytes = getIntegerAttrFromADG(_adg_fn, "iob_spad_bank_size");
        }
        else
        {
          NumGPE = _n_pes;
          NumIOB = _n_iobs;
          bankBytes = _sram_bank_kb_size * 1024;
        }

        // Parse command line algorithm choice
        ComputeAlgorithm algorithm_kind = ComputeAlgorithm::Undefine;
        if (_algorithm_kind == "conv_direct")
          algorithm_kind = ComputeAlgorithm::Conv_Direct;
        else if (_algorithm_kind == "conv_im2col")
          algorithm_kind = ComputeAlgorithm::Conv_Im2Col;
        else if (_algorithm_kind == "conv_winograd")
          algorithm_kind = ComputeAlgorithm::Conv_Winograd;
        else if (_algorithm_kind == "gemm_standard")
          algorithm_kind = ComputeAlgorithm::GEMM_Standard;
        else if (_algorithm_kind != "undefine")
        {
          llvm::errs() << "[ADORAOpStrategyDecisionPass] Warning: Unknown algorithm kind '"
                       << _algorithm_kind << "'. Using automatic selection.\n";
        }

        // Parse command line stationary choice
        DataflowStrategy stationary_kind = DataflowStrategy::Undefine;
        if (_stationary_kind == "inputstationary")
          stationary_kind = DataflowStrategy::InputStationary;
        else if (_stationary_kind == "weightstationary")
          stationary_kind = DataflowStrategy::WeightStationary;
        else if (_stationary_kind == "outputstationary")
          stationary_kind = DataflowStrategy::OutputStationary;
        else if (_stationary_kind != "undefine")
        {
          llvm::errs() << "[ADORAOpStrategyDecisionPass] Warning: Unknown stationary kind '"
                       << _stationary_kind << "'. Using automatic selection.\n";
        }

        m.walk([&](mlir::ADORA::ADORATensor::GemmOp op)
               { AutoSetGemmStrategy(op, NumGPE, NumIOB, bankBytes, bandByteWidth, stationary_kind); });

        m.walk([&](mlir::ADORA::ADORATensor::ConvOp op)
               { AutoSetConvStrategy(op, NumGPE, NumIOB, bankBytes, bandByteWidth, stationary_kind, algorithm_kind); });
      }

    } // namespace ADORATensor
  } // namespace ADORA
} // namespace mlir
