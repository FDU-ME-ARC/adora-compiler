# 下一步行动计划

> 更新时间：2026-05-17  
> 本文记录当前阻塞点、待办事项和明确的行动指令。  
> 按优先级排序，每项都说明由谁来做、怎么做、怎么验收。

---

## 0. 已完成（本 session）

- ✅ `EmitPytest`：移除 Path 2（dep_summary fallback），stream 着色移入 emit 内部
- ✅ `cgra-mapper.cpp`：`--enable-async` 流水线只保留 `schedule-tasks`
- ✅ `schedule_gemm_tiled.mlir`：CHECK 模式修复，lit passed 从 23 → 24
- ✅ 文档：`pipeline_overview.md`、`pipeline_detailed.md`、`review_current_state.md` 更新

---

## 1. 修复 cgra-mapper `is not supported!` bug【阻塞 PR6.4】

### 问题描述

cgra-mapper 在 mapping 阶段遇到 `UseAddr`（cfgIdMap index=15）操作时打印 `is not supported!` 并静默退出，不生成 Python 输出。复现命令：

```bash
cd /data00/home/loujiahang/adora/adora-compiler
./build/bin/cgra-mapper \
    --adg=test/spec/cgra_adg_fp32.json \
    --op-file=test/spec/operations_fp32.json \
    --output-type=pytest \
    experiment/taskschedule/03_3mm/input.mlir
# 末尾输出：index: 15 / is not supported!
```

### 定位方法

```bash
# 找到报错位置
grep -rn "is not supported" \
    /data00/home/loujiahang/adora/adora-compiler/mapper/src/

# 装 gdb 后单步（或加 fprintf(stderr) 打印调用栈）
sudo apt-get install -y gdb
gdb --args ./build/bin/cgra-mapper \
    --adg=test/spec/cgra_adg_fp32.json \
    --op-file=test/spec/operations_fp32.json \
    --output-type=pytest \
    experiment/taskschedule/03_3mm/input.mlir
# gdb 里：run → bt
```

VS Code 调试配置已写入 `.vscode/launch.json`（选 `cgra-mapper: 3mm is-not-supported debug`），需要先安装 `ms-vscode.cpptools` 插件 + gdb。

### 验收

修完后运行上述命令，末尾不再出现 `is not supported!`，且 `/tmp/3mm_out.py` 存在并含 `asyncio.gather`。

---

## 2. PR6.4 端到端冒烟【依赖 §1】

§1 修完之后，用以下命令验证 `--enable-async=true` 路径产出并发 Python：

```bash
cd /data00/home/loujiahang/adora/adora-compiler

./build/bin/cgra-mapper \
    --enable-async=true \
    --adg=test/spec/cgra_adg_fp32.json \
    --op-file=test/spec/operations_fp32.json \
    --output-type=pytest \
    experiment/taskschedule/03_3mm/input.mlir \
    > /tmp/3mm_async.py

grep "asyncio.gather\|create_task" /tmp/3mm_async.py | head -5
# 出现 → PR6.4 通过 → 可以 merge
```

---

## 3. Loop-carried dep 的 Python emit 语义【设计待定】

### 问题

`_lcDepSummary` 已缓存但 emit 层未消费。跨 `tk` 迭代依赖在生成的 Python 里缺失，当前保守串行。

### 候选方案

| 方案 | 实现复杂度 | 语义精确度 |
|------|-----------|-----------|
| A：保守串行（不改）| — | 正确，性能差 |
| B：`asyncio.Semaphore(1)` | 低 | 语义有歧义 |
| C：`asyncio.Queue` put/get | 中 | 最精确 |

**你需要决定选哪个方案**，选定后我来实现。

---

## 4. 4 个 pre-existing lit failures【优先级低】

以下 4 个测试与 task-schedule 无关，是 cdfggen 和 kernel 提取 pass 的老问题：

```
cgra-opt/cdfggen/gemm/gemm.mlir
cgra-opt/cdfggen/getTanh/getTanh.mlir
cgra-opt/cdfggen/interleave/mergeadd_opt.mlir
cgra-opt/kernel/gemm.mlir
```

选项：
- **(A)** 加 `// XFAIL: *` 让 CI 干净，后续单独 PR 修
- **(B)** 排查根因修复（需要你了解 cdfggen pass 的预期行为）

**建议选 A**，先让 `ninja check-adora` 全绿，我来加 XFAIL。

---

## 5. `EmitCGRACall` / `EmitVitisSDK` 对称移植【依赖 §2】

PR6.4 冒烟通过后，把 `EmitPytest` 的 stream 着色逻辑对称移植到：
- `mapper/src/emit/EmitCGRACall.cpp`
- `mapper/src/emit/EmitVitisSDK.cpp`

约 ~40 行/文件，机械改动，我来做。

---

## 6. `analyzeDependencyInGraph` 空 stub【论文相关，远期】

`ScheduleAdoraTasks.cpp:160` 函数体只有一行注释，是后续 reorder/fusion 决策的前置依赖。这是论文 gap §缺口1，需要你决定是补实现还是在论文里缩小声称范围。

---

## 快速行动清单

```
今天：
  [ ] 装 gdb（sudo apt-get install gdb 或 conda install gdb）
  [ ] grep -rn "is not supported" mapper/src/ 找到代码位置
  [ ] 决定 loop-carried dep Python 方案（A/B/C）
  [ ] 决定 4 个 pre-existing failures 处理方式（XFAIL 还是修）

修完 §1 后（我来）：
  [ ] 验证 --enable-async=true 3mm 产出 asyncio.gather
  [ ] 对称移植 EmitCGRACall / EmitVitisSDK
  [ ] 实现 loop-carried dep emit（等你选方案）
  [ ] 加 XFAIL（等你决定）
```
