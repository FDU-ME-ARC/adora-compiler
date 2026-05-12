# PR6 Session Handover — 归档与待办

最后一次 session 结束时的状态；下次开新 context 时 `cat` 本文件即可恢复。

---

## 已完成（已 push 到 `jhlou/pr6-loop-carried-analysis`）

| commit | 范围 | 验证 |
|---|---|---|
| `f87ac31` | **PR6.2 finish** — `emit-token` 默认 false→true、spec §0.1/§3/§5/§6 补丁、6 个 experiment run.sh 清理、默认开关哨兵测试 + **PR6.3 核心** — `rebuildStoreSync` dropAllUses 修复、`stripTokenIterArgsFromAffineFor` Pass 5、对应 strip 回归测试 + **PR6.4 Pytest 实现** — `_idToOp` / `_depSummary` 缓存、`getDepsTaskNames` SSA→dep_summary 双路径、WAR 过滤 | 构建 ✅；9/9 token 相关 lit 测试通过；5/5 experiment pipeline 的 IR 终态正确（token iter_args 全 strip，非 token reduction 正确保留） |
| `fd2887c` | 5 个存量 lit 测试去除显式 `emit-token=true`（现在走默认路径） | 8/9 pass；1 fail 是 pre-existing `schedule_gemm_tiled`，与 token 无关 |
| `4d2c05d` | **PR6.4 接线** — cgra-mapper `--enable-async` CLI（默认 false），在 mapping 前插入 `schedule-tasks + assign-streams + lower-async-tokens`；link `MLIRADORATransforms` | 构建 ✅；默认 off 时字节级等同 pre-PR6；on 时**无法端到端验证**（见未解决 1） |

---

## ⚠️ 未解决问题（需要你接手）

### 1. PR6.4 端到端管道**未通过冒烟**（阻塞 merge）

**现象**：用 `./build/bin/cgra-mapper --enable-async=true ...
test/cgra-mapper/ADORATensor/gemm_funccall/Output/gemm_funccall.mlir.tmp/opt.mlir
--output=...` 跑不出输出，cgra-mapper 在 mapping 阶段崩溃。

**根因不在 PR6.4**：同样的输入在 `--enable-async=false`（默认）下**也崩溃**，
说明是 cgra-mapper 对这个 bf16 `gemm_funccall` 的 pre-existing bug，与
token 管道无关。

**要你做的事**：
1. 找一个 cgra-mapper **能正常跑通 mapping** 的干净样例（建议 fp32、
   多 kernel、带 BlockLoad/BlockStore/kernel 依赖，最好是 `.C` 源文件经
   `adoracc.py` 产出的 `opt.mlir`）；
2. 用它跑 `--enable-async=true`，grep emit 产物中的 `await asyncio.gather`
   是否出现（若出现 = PR6.4 Path 2 dep_summary 路径真在工作）；
3. 若干净样例也有问题，把 stderr 贴出来我看。

**候选样例来源**：
- `test/cgra-mapper/FPVecAdd/` — 太简单，没有 kernel 间依赖，无法验证 gather
- `test/cgra-mapper/ADORATensor/**/` — 需逐个筛选哪个能跑通且有多 kernel

### 2. `schedule_gemm_tiled.mlir` 及其他 4 个 pre-existing lit failures

以下 5 个测试在本 session 开始前就已经失败（我验证过：revert 掉本
session 所有改动后仍然 FAIL）：

```
cgra-opt/cdfggen/gemm/gemm.mlir
cgra-opt/cdfggen/getTanh/getTanh.mlir
cgra-opt/cdfggen/interleave/mergeadd_opt.mlir
cgra-opt/kernel/gemm.mlir
cgra-opt/schedule/schedule_gemm_tiled.mlir
```

**与 PR6 无关**，独立 issue。建议：要么单独开 PR 修，要么 `XFAIL` 标记。

### 3. `EmitCGRACall` / `EmitVitisSDK` 未移植 PR6.4 双路径

只改了 `EmitPytest.cpp`。同构改造（加 `_idToOp` / `_depSummary` 缓存 +
`getDepsTaskNames` 双路径）约 ~200 行对称代码。

**要你做的事**：待 PR6.4 端到端冒烟通过（问题 1 解决）后，把改动对称
迁移到 `mapper/src/emit/EmitCGRACall.cpp` 和 `EmitVitisSDK.cpp`。

### 4. Loop-carried 依赖 emit 未翻译

`_lcDepSummary` 在 EmitPytest 已缓存但**没有消费**。跨迭代依赖在
Python 里需要 semaphore / bounded channel；首版保守行为是串行。

**要你做的事**：设计 LC→Python 的映射语义（`asyncio.Semaphore`？
`collections.deque` 作为 N-stage pipe？），然后在 `getDepsTaskNames` 里
把 `adora.lc_dep_summary` 接上。

### 5. `_idToOp` 多 func 场景未清理

缓存是进程级的，跨多个 `func.func` 时会污染。

**要你做的事**：在 emit 每个 func 开始时 reset `_idToOp / _depSummary /
_lcDepSummary`。落点在 EmitPytest 的 func-level 入口（大概 `emitFunc`
或类似），加 `_idToOp.clear(); _depSummary = nullptr; _lcDepSummary = nullptr;`。

### 6. PR6.3 留下的孤儿 `ADORA.event.create` sentinel

PR6.2 在循环前创建的 token sentinel，在 PR6.3 strip 后没有 users。
验证时看到过（例如 05_loop_carried 的 `%0, %1, %2`）。

**影响**：生成 emit 代码时可能产生冗余的 create/destroy 调用对，但
因为 never signaled/waited，等同 no-op。**非阻塞**。

**要你做的事**（可选）：在 PR6.3 Pass 5 后加一次 dead-event sweep，
或依赖 canonicalizer DCE（但 event ops 有 side-effect，默认 DCE 不处理）。

---

## 下次恢复 context 的命令

```bash
cat docs/pr6_session_handover.md                                 # 本文件
cat docs/pr6_2_thread_loop_carried_tokens_prompt.md              # PR6.2 spec
cat docs/pr6_3_lower_async_tokens_affine_for_options.md          # PR6.3 spec
git log --oneline -5                                             # 查最新提交
git log --stat 4d2c05d f87ac31 fd2887c                          # 本 session 总体改动
```

---

## 关键设计决策（别忘）

1. **无 runtime**：并发由 host 静态分析 emit 产物决定多 stream issue；
   `!ADORA.token` 是编译期依赖凭证，不是 runtime event。
2. **emit 前 lower**：PR6.3 负责把 SSA token 降为 `adora.dep_summary` 属性 +
   event ops；emit 端只读属性，不处理 SSA。
3. **默认发射 token**：`--adora-schedule-tasks` 默认 `emit-token=true`；
   `emit-token=false` 仅用于 pre-PR6 baseline 回归对照。
4. **cgra-mapper `--enable-async` 默认 off**：token 管道是 opt-in，默认
   行为字节级等同 pre-PR6；PR6.4 成熟 + 端到端冒烟通过后再议默认翻转。
