# Prompt: 修复 `adora-adjust-kernel-mem-footprint` SIGSEGV

**用途**：新开 context 时直接复制下方代码块作为启动 prompt。

---

```
修复 adora-adjust-kernel-mem-footprint pass 的 SIGSEGV bug。

背景：
- 编译器路径：/data00/home/loujiahang/adora/adora-compiler
- 构建路径：build/（ninja cgra-opt）
- FileCheck：/data00/home/loujiahang/CGRVOPT/llvm-project-onnx/build/bin/FileCheck
- 建议新分支：jhlou/fix-adjust-memory-footprint（从 develop 或 jhlou/scheduletasks 拉出）

已知症状：
- lit 测试 6 个全失败（baseline fd9b48f 就失败，与 PR4 无关）：
    test/adoracc/kernel/{atax_unroll,jacobi_2d,mvt,mvt_has_kernel,mvt_unroll}/*.mlir
    test/cgra-mapper/FPVecAdd/FPVecAdd.mlir
- 统一崩溃在：
    cgra-opt --adora-simplify-affine-loop-levels --canonicalize -cse
             --adora-simplify-loadstore --adora-math-rewrite
             --adora-adjust-kernel-mem-footprint=cachesize=128
             FPVecAdd_kernel.mlir
    → exit -11 (SIGSEGV)

最小复现：
    cgra-opt --adora-adjust-kernel-mem-footprint=cachesize=128 \
      /tmp/fpvec_e2e/adora-cc-ir/0_kernels/FPVecAdd_kernel.mlir
    （或任意含 memref<?xf32> 动态 shape 的 ADORA.kernel 输入）

已定位问题（部分）：
- 文件 lib/Dialect/ADORA/Transforms/Loop/AdjustMemoryFootprint.cpp
- `excessFactor_toCachesize`（line 337-352）3 处 std::optional 没有
  has_value() 检查就 dereference，对 memref<?xf32> 会 UB：
    *maxSpace_singleMem           ← line 339
    *fp_totalMem, *fp_singleMem  ← line 351-352

但仅修这 3 处不够——崩溃在 BlockLoad/BlockStore 生成 loop（line 744+）处
还有问题。从调试输出看：只替换了第一个 memref（%arg0 → memref<20xf32>），
处理第二个 memref 时崩溃。

任务：
1. 用 asan build 定位剩余崩溃点：
     cmake -DCMAKE_CXX_FLAGS='-fsanitize=address -g -O1'
2. 加 has_value() 检查避免 null deref
3. 对 memref<?xf32> 动态 shape：要么保守跳过 partition（excessFactor=1），
   要么报 emitError 让用户先做 shape specialization
4. 加 regression test：
     test/cgra-opt/kernel/adjust_footprint_dynamic_shape.mlir
     验证 cgra-opt --adora-adjust-kernel-mem-footprint 对 memref<?xf32> 不崩溃
5. 修好后验证 6 个失败的 lit 测试：
     cd build && python3 /data00/home/loujiahang/CGRVOPT/llvm-project-onnx/llvm/utils/lit/lit.py \
         ../test/adoracc/ ../test/cgra-mapper/ -v
   期望至少 5 个 pass（gemm_funccall 保留 XFAIL）
6. 成功后可能触发更多下游 bug（cgra-mapper 自己的 emit 逻辑），
   独立处理或标 XFAIL。

相关文件：
- lib/Dialect/ADORA/Transforms/Loop/AdjustMemoryFootprint.cpp（1787 行）
- include/ADORA/Dialect/ADORA/Transforms/Passes.td（pass option 定义）
- tools/adoracc/adoracc.py:235（kernel_opt_cmd 里的 pass 调用）

约束：
- 修复应该是**向前兼容**的：cachesize 等 option 语义不变
- 对静态 shape memref 的已有行为不能回退
- 如果某些输入无法 partition，应该 emit 诊断信息而不是崩溃
```

---

## 补充上下文（仅参考，不放入 prompt）

**为什么这个 bug 独立于 PR4 async token 工作**：

`adoracc.py` 的 kernel_opt pipeline 只走到 `adora-adjust-kernel-mem-footprint`，
**不调用** `adora-schedule-tasks`：

```
adora-simplify-affine-loop-levels
  → canonicalize → cse
  → adora-simplify-loadstore
  → adora-math-rewrite
  → adora-adjust-kernel-mem-footprint    ← SIGSEGV 发生在这
  [STOP]
```

而 async token 在 `adora-schedule-tasks` 才生成，此 pass 崩溃时 IR 里
**没有任何 `!ADORA.token`**，所以和 PR4 的工作完全解耦。

**已尝试但回滚的修复**：

曾尝试只修 `excessFactor_toCachesize` 的 3 处 null deref（加 has_value 检查），
rebuild 后仍崩溃，说明：
1. 那 3 处确实是 bug 但不是 root cause
2. BlockLoad/BlockStore 生成 loop 里还有其他 memory error
3. 需要 asan build 精确定位

尝试的修改已 `git checkout --` 回滚，未提交。
