// #map = affine_map<(d0) -> (d0)>
// module {
//   func.func @if_loop_1(%arg0: memref<100xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
//     %cst = arith.constant dense<2> : vector<4xi32>
//     %cst_0 = arith.constant dense<10> : vector<4xi32>
//     %cst_1 = arith.constant dense<0> : vector<4xi32>
//     %true = arith.constant true
//     %c0 = arith.constant 0 : index
//     %c100 = arith.constant 100 : index
//     %c4 = arith.constant 4 : index
//     %0 = dataflow.launch : i32 {
//       %1 = dataflow.task : vector<4xi32> {
//         dataflow.state %true, "loop_start" or "null" {Enable = "Loop_Start"} : i1
//         %3 = dataflow.for %arg1 = %c0 to %c100 step %c4 iter_args(%arg2 = %cst_1) -> (vector<4xi32>) {
//           %4 = dataflow.execution : vector<4xi32> {
//             %5 = dataflow.addr %arg0[%arg1] {laneNums = 4 : i32, memShape = [100]} : memref<100xi32>[index] -> i32
//             %6 = dataflow.load %5 {laneNums = 4 : i32, operand_segment_sizes = array<i32: 1, 1, 1, 0>, permutation_map = #map} : i32 -> vector<4xi32>
//             %7 = arith.muli %6, %cst : vector<4xi32>
//             %8 = arith.cmpi ugt, %7, %cst_0 : vector<4xi32>
//             %9 = arith.addi %7, %arg2 : vector<4xi32>
//             %10 = dataflow.select %8, %9, %arg2 : vector<4xi1>, vector<4xi32>
//             %11 = arith.addi %arg1, %c4 {Exe = "Loop"} : index
//             %12 = arith.cmpi eq, %11, %c100 {Exe = "Loop"} : index
//             dataflow.state %12, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
//             dataflow.yield {execution_block = 1 : i32} %10 : vector<4xi32>
//           }
//           dataflow.yield %4 : vector<4xi32>
//         } {Loop_Band = 0 : i32, Loop_Level = 0 : i32}
//         dataflow.yield %3 : vector<4xi32>
//       }
//       %2 = vector.reduction <add>, %1 : vector<4xi32> into i32
//       dataflow.yield %2 : i32
//     }
//     return %0 : i32
//   }
// }

 #map = affine_map<(d0) -> (d0)>
module {
  func.func @if_loop_1(%arg0: memref<100xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant dense<2> : vector<4xi32>
    %cst_0 = arith.constant dense<10> : vector<4xi32>
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %c0 = arith.constant 0 : index
    %c100 = arith.constant 100 : index
    %c4 = arith.constant 4 : index
    %0 = dataflow.launch : i32 {
      %1 = dataflow.task : i32 {
        dataflow.state %true, "loop_start" or "null" {Enable = "Loop_Start"} : i1
        %2 = dataflow.for %arg1 = %c0 to %c100 step %c4 iter_args(%arg2 = %c0_i32) -> (i32) {
          %3 = dataflow.execution : i32 {
            %4 = vector.broadcast %arg2 : i32 to vector<4xi32>
            %5 = dataflow.addr %arg0[%arg1] {laneNums = 4 : i32, memShape = [100]} : memref<100xi32>[index] -> i32
            %6 = dataflow.load %5 {laneNums = 4 : i32, operand_segment_sizes = array<i32: 1, 1, 1, 0>, permutation_map = #map} : i32 -> vector<4xi32>
            %7 = arith.muli %6, %cst : vector<4xi32>
            %8 = arith.cmpi ugt, %7, %cst_0 : vector<4xi32>
            %10 = dataflow.select %8, %7, %4 : vector<4xi1>, vector<4xi32>
            %11 = arith.addi %arg1, %c4 {Exe = "Loop"} : index
            %12 = arith.cmpi eq, %11, %c100 {Exe = "Loop"} : index
            dataflow.state %12, "loop_exit" or "loop_back" {Exe = "Loop"} : i1
            %13 = vector.reduction <add>, %10 : vector<4xi32> into i32
            dataflow.yield {execution_block = 1 : i32} %13 : i32
          }
          dataflow.yield %3 : i32
        } {Loop_Band = 0 : i32, Loop_Level = 0 : i32}
        dataflow.yield %2 : i32
      }
      dataflow.yield %1 : i32
    }
    return %0 : i32
  }
}