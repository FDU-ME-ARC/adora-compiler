# ADORA Control Flow / CondStore 后续开发任务清单

## 0. 当前开发基线

当前工作基于：

* 上游开发分支：`upstream/jhlou/scheduletasks`
* 本地同步分支：`scheduletasks`
* 个人开发分支：`scheduletasks-h`
* 开发 worktree：`/home/jyhu/adora-compiler-scheduletasks-h`
* 当前个人远程提交：`e88f0705dd5489e92108ff6d03a619e432bf31d2`

已经完成的 memref 功能迁移**不要重复实现**：

* `memref.load` 的元素索引已转换为字节偏移：

  * i32 插入 `CONST 4 + MUL`
  * 地址连接到 load 的 operand port `0`
* `memref.store` 的元素索引已转换为字节偏移：

  * i32 插入 `CONST 4 + MUL`
  * 地址连接到 store 的 operand port `2`
* 1-byte 元素不插入无意义乘法。
* synthetic `CONST/MUL` 已补充空 `operation` 防护。
* 专用 lit 测试已通过。
* 完整 `check-adora` 没有引入新的失败。
* `simple.mlir` 已完成端到端验证。
* 当前功能仅推送至个人远程 `origin/scheduletasks-h`，**尚未向主仓库创建 PR**。

本轮工作的目标是：

> 在上述 memref 支持基础上，继续完善复杂 `if/else` 控制流和条件访存支持，重点实现并验证 `ADORA.cond_store`，同时整理本地实验目录；待所有相关功能形成完整、稳定、可回归验证的功能组后，再统一创建一次 PR。

---

# 任务一：建立复杂控制流测试基线并摸清现有编译行为

## 目标

在修改控制流代码之前，先系统确认当前 ADORA 对不同 `if/else` 形式的真实处理结果，建立可持续使用的最小测试集。

本阶段以“分析现状 + 建立 testcase”为主，不进行大规模重构。

## 待办

### 1.1 梳理现有控制流相关实现

* [x] 搜索并定位当前与以下内容有关的实现：

  * `if`
  * `if-else`
  * `scf.if`
  * `affine.if`
  * `cf.cond_br`
  * `arith.select`
  * `ADORA.isel`
  * CDFG 中的 control edge / predicate 相关逻辑
* [x] 找出现有“简单 if-else 处理”具体位于哪些函数和文件中。
* [x] 记录当前处理过程：

  * 输入 IR 形式
  * 经过哪些 pass
  * 是否发生 if-conversion
  * 最终 CDFG 中生成什么节点
  * Mapper 是否能够正常处理
* [x] 不修改无关控制流实现。

### 1.2 建立最小控制流 testcase

在 `experiment/jyhu/` 中建立或整理一组小型 testcase，用于人工观察完整编译流程。

至少覆盖：

* [x] `if` 无 `else`
* [x] 普通 `if-else`
* [x] 连续 `if / else if / else`
* [x] 两级 nested `if`
* [x] `if` 中纯计算
* [x] `if` 中 `memref.load`
* [x] `if` 中 `memref.store`
* [x] `if-else` 两侧分别存在 `memref.store`
* [x] loop 内部存在 `if`

建议目录：

```text
experiment/jyhu/control-flow/
├── if_simple/
├── if_else/
├── if_elseif_else/
├── nested_if/
├── if_compute/
├── if_load/
├── if_store/
├── if_else_store/
└── loop_if/
```

每个 testcase 尽量保持最小，不混入无关计算。

### 1.3 记录各 testcase 的实际 lowering 结果

* [x] 对每个 testcase 跑：

  * C → MLIR（如适用）
  * kernel extraction / optimization
  * CDFG generation
  * mapper（当前能够运行的 testcase）
* [x] 记录 `if-else-if` 等结构最终在 MLIR 中究竟表现为：

  * nested `scf.if`
  * `affine.if`
  * `cf.cond_br`
  * `select`
  * 其他形式
* [x] 特别观察：

```c
if (a) {
    S1;
} else if (b) {
    S2;
} else {
    S3;
}
```

是否能够正确表达：

```text
S1 : a
S2 : !a && b
S3 : !a && !b
```

* [x] 暂时不要为了追求复杂 case 全部通过而修改大量底层代码。

## 预期成果

### 本地实验文件

```text
experiment/jyhu/control-flow/
```

包含一组最小、分类清楚的控制流 testcase。

### 开发记录

建议新增：

```text
docs/development/control-flow-baseline.md
```

至少记录：

```markdown
| Case | Frontend IR | 当前 CDFG | Mapper | 状态 | 备注 |
|------|-------------|-----------|--------|------|------|
| if_simple | ... | ... | PASS | PASS | |
| if_else | ... | ... | ... | ... | |
| if_elseif_else | ... | ... | ... | ... | |
```

### 完成标准

* [x] 能清楚解释当前简单 `if-else` 的已有实现。
* [x] 能明确指出 `else if` / nested if 当前失败或处理不完整的位置。
* [x] 后续修改控制流代码时已有稳定 testcase 可用于回归。

---

# 任务二：完善 if/else 控制流处理与路径条件分析

## 目标

在已有简单 `if-else` 支持上扩展更复杂但仍可控的场景，优先保证：

```text
if
if-else
if-else if-else
nested if
```

能够获得正确的控制条件。

第一阶段重点放在**路径条件正确性**，不要一开始扩展到 `switch`、`break`、`continue` 等所有复杂场景。

## 待办

### 2.1 明确内部控制条件表示

* [x] 梳理现有代码如何表示 branch condition。
* [x] 判断能否复用现有结构，不优先新建新的复杂 Dialect。
* [x] 为每个 branch/body operation 能够获取对应的 path predicate。

例如：

```c
if (a) {
    S1;
} else if (b) {
    S2;
} else {
    S3;
}
```

内部应至少能够区分：

```text
S1 → a
S2 → !a && b
S3 → !a && !b
```

对于：

```c
if (a) {
    if (b) {
        S;
    }
}
```

应能够表达：

```text
S → a && b
```

### 2.2 扩展简单 if-else 实现

* [x] 先保证单层 `if` 行为不回归。
* [x] 保证普通 `if-else` 行为不回归。
* [x] 支持连续 `if / else if / else`。
* [x] 支持至少两层 nested if。
* [x] 对 predicate 取反逻辑统一处理，不在多个代码位置重复手写。
* [x] 避免将 mutually-exclusive branch 错误视为同时执行。
* [ ] 暂不扩展到：

  * `switch`
  * `break`
  * `continue`
  * 非结构化 CFG
  * 复杂循环退出

这些作为后续扩展，不作为本轮 MVP。

### 2.3 为正式功能增加 lit regression test

`experiment/jyhu` 中的 testcase 主要用于开发观察。

已经稳定且属于编译器正式行为的部分，需要增加正式测试。

* [x] 为简单 `if-else` 保留或补充 lit 测试。
* [x] 新增 `if-else if-else` 测试。
* [x] 新增 nested if 测试。
* [x] 测试重点检查：

  * condition 节点
  * select / isel / control node
  * branch 对应关系
  * CDFG edge
  * 不出现错误的多余节点
* [x] 测试名称能够反映功能，不使用个人名称。

## 预期成果

### 代码

根据现有实现位置修改对应控制流/CDFG generation 文件，尽量避免新建不必要的公共 API。

### 正式测试

建议形成类似：

```text
test/...
├── if_else.mlir
├── if_elseif_else.mlir
└── nested_if.mlir
```

具体位置应遵循仓库现有测试目录结构。

### 完成标准

* [x] `if`
* [x] `if-else`
* [x] `if-else if-else`
* [x] 两层 nested if

至少以上四类能够获得正确 path condition，并通过对应 regression test。

---

````markdown
# 任务三：对齐现有 CSTORE 后端契约并实现 `ADORA.cond_store`

## 目标

为条件写内存建立完整且与现有 Mapper / CGRA 后端兼容的编译链路：

```text
if / else 控制流
        ↓
计算正确的 path predicate
        ↓
ADORA.cond_store
        ↓
CDFG: CSTORE
        ↓
现有 Mapper / IOB
        ↓
conditional store
````

本任务**不自行重新设计一套 CSTORE 后端语义**。应先调查并确认仓库中已经存在的 CSTORE backend contract，再按照现有 Mapper / 硬件接口补齐 ADORA IR 和 CDFG 前端缺失部分。

当前代码已经表明 Mapper 对 `CSTORE` 有一定支持，包括：

* 将 `CSTORE` 视为 I/O node。
* 将 `CSTORE` 视为 store/output node。
* `CSTORE` 使用地址输入。
* `CSTORE` 可通过 `UseEn` 使用 enable 条件。

但仍需确认：

* 实际使用的 operation spec 是否包含 `CSTORE`。
* 实际 ADG / IOB 是否具有 `UseEn`。
* `CSTORE` 的 operand / port 约定。
* `CSTORE` 对应的 opcode、operand 数量等硬件配置。

因此必须先完成后端接口审计，再进入正式实现。

---

## 3.1 审计现有 CSTORE 后端支持

### 目标

只读取、分析现有代码，不修改 Mapper 或硬件接口，明确 `CSTORE` 从 CDFG 到硬件的完整 contract。

### 待办

* [x] 全局搜索以下关键字：

```text
CSTORE
CLOAD
UseEn
UseAddr
IsStore
DFGIONode
Operations::opCapable
Operations::OPC
```

* [x] 重点检查：

```text
mapper/include/dfg/
mapper/src/dfg/
mapper/src/ir/
mapper/src/mapper/
mapper/src/op/
test/spec/
lib/DFG/Documents/
```

* [x] 确认 Mapper 是否已经将 `CSTORE` 识别为合法 DFG operation。
* [x] 确认 `CSTORE` 是否作为 `DFGIONode` 处理。
* [x] 确认 `CSTORE` 是否被视为输出/store node。
* [x] 确认 IOB 配置中：

  * `UseAddr` 对 `CSTORE` 的处理。
  * `UseEn` 对 `CSTORE` 的处理。
  * `IsStore` 对 `CSTORE` 的处理。
* [x] 确认 CSTORE 的逻辑输入语义：

```text
data
address
enable / condition
```

* [x] **查明三个输入在 CDFG 中的确切 operand port 编号。**
* [x] 不根据 `STORE` 的端口布局自行推断 `CSTORE` 端口布局。
* [x] 检查实际使用的 operations JSON / operation spec 中是否存在 `CSTORE`：

  * operation name
  * OPC
  * numOperands
  * numRes
  * latency
* [x] 检查实际使用的 ADG / IOB 配置中是否存在 `UseEn`。
* [x] 搜索是否已有 CSTORE / CLOAD testcase、旧 benchmark 或历史实现可作为参考。
* [x] 确认 `ISEL` 的用途，但不要用 `ISEL` 替代 `CSTORE`，除非现有后端明确如此设计。

### 输出文档

新增开发记录：

```text
docs/development/cstore-backend-audit.md
```

至少回答：

```markdown
## CSTORE Backend Contract

### 1. DFG 表示
- operation name:
- node type:
- 是否属于 IO node:

### 2. Operand
- data port:
- address port:
- enable port:

### 3. Mapper
- CSTORE 是否已识别:
- UseAddr:
- UseEn:
- IsStore:

### 4. Operation Spec
- CSTORE 是否存在:
- OPC:
- numOperands:
- numRes:
- latency:

### 5. ADG / Hardware
- IOB 是否存在 UseEn:
- 是否能够真正禁止 store:

### 6. 当前缺口
- ADORA IR:
- CDFGgen:
- Mapper:
- operation spec:
- hardware:

### 7. 后续实现方案
```

### 阶段完成条件

只有能够明确回答下面的问题后，才能最终确定 `ADORA.cond_store` → `CSTORE` 的 lowering 接口：

* [x] CSTORE 的三个输入分别是什么。
* [x] 三个输入的 CDFG port 编号是什么。
* [x] 当前 Mapper 是否无需修改即可识别 CSTORE。
* [x] 当前使用的 operation spec 是否支持 CSTORE（审计结论：不支持）。
* [x] 当前 ADG / IOB 是否真正支持 enable（审计结论：不支持）。

---

## 3.2 明确 conditional store 的 lowering 规则

### 目标

不要简单地把“所有 if 中的 `memref.store`”全部替换成 `cond_store`。

应根据控制流语义决定使用：

```text
SELECT / ISEL + STORE
```

还是：

```text
CSTORE
```

### Case A：只有一个分支发生 store

输入：

```c
if (cond) {
    A[i] = x;
}
```

预期：

```text
CSTORE(
    data = x,
    addr = A[i],
    enable = cond
)
```

* [x] 应优先转换为 `cond_store / CSTORE`。
* [x] 不再为了保持原值额外生成：

```text
LOAD old_value
→ SELECT
→ STORE
```

---

### Case B：if 和 else 都写相同地址

输入：

```c
if (cond) {
    A[i] = x;
} else {
    A[i] = y;
}
```

优先保持：

```text
x ─────┐
       SELECT(cond)
y ─────┘
          ↓
       STORE A[i]
```

即：

```text
SELECT + 普通 STORE
```

* [x] 不强制生成两个 CSTORE。
* [x] 避免增加不必要的 I/O node 和条件 store。

---

### Case C：if / else 写不同地址

输入：

```c
if (cond) {
    A[i] = x;
} else {
    B[j] = y;
}
```

预期：

```text
CSTORE A[i], x,  cond
CSTORE B[j], y, !cond
```

* [x] 两个 store 分别保留自己的条件。
* [x] 不错误合并为普通 STORE。

---

### Case D：else-if 链

输入：

```c
if (a) {
    A[i] = x;
} else if (b) {
    B[j] = y;
} else {
    C[k] = z;
}
```

预期 path predicate：

```text
A store → a
B store → !a && b
C store → !a && !b
```

* [x] `cond_store` 必须使用完整 path predicate。
* [x] 不允许只使用当前局部 `if` 的 condition。

---

### Case E：nested if

输入：

```c
if (a) {
    if (b) {
        A[i] = x;
    }
}
```

预期：

```text
enable = a && b
```

* [x] 必须正确传播父级 predicate。

---

## 3.3 根据后端 contract 定义 `ADORA.cond_store`

### 前置条件

完成 3.1，已经明确 CSTORE backend contract。

### 待办

* [x] 在：

```text
include/ADORA/Dialect/ADORA/IR/ADORAOps.td
```

新增：

```text
ADORA.cond_store
```

* [x] 设计时参考 MLIR `memref.store`，但遵循 ADORA 现有 Dialect 风格。
* [x] 基本语义保持：

```text
value
memref
indices
condition
```

其中：

```text
condition = true
→ 执行 store

condition = false
→ 不产生 memory write
```

* [x] condition 优先采用 `i1`，除非现有 backend contract 明确要求其他类型。
* [x] `cond_store` 不产生普通数据 result。
* [x] 正确声明 memory write side effect。
* [x] 必要时实现：

  * builder
  * verifier
  * parser / printer
  * canonicalization
* [x] 在：

```text
lib/Dialect/ADORA/IR/ADORAOps.cpp
```

补充所需成员函数。

### 设计限制

* [x] 不把 `!ADORA.token` 当成 predicate。
* [x] 保持：

```text
token
= happens-before / 什么时候能够执行

condition
= predicate / 是否应该执行
```

* [x] 当前 `cond_store` 只针对 **kernel 内部的条件 memory store**。
* [x] Stage A 只支持静态 shape 的 rank-1 memref；动态 rank-1 和更高 rank
  必须 fail closed，避免生成不完整的 size/address metadata。
* [x] 暂不扩展：

  * `ADORA.BlockStore`
  * `ADORA.BlockLoad`
  * task-level conditional issue
* [x] `cond_load` 暂不实现，除非后续需求明确。

---

## 3.4 修改 if-conversion / CDFG generation

### 目标

让现有：

```text
lowerSCFIfToSelect
```

在处理 memory side effect 时能够根据 3.2 的规则选择：

```text
SELECT + STORE
```

或：

```text
COND_STORE / CSTORE
```

而不是始终通过：

```text
LOAD old value
→ SELECT
→ STORE
```

处理单侧条件 store。

### 待办

* [x] 梳理当前 `lowerSCFIfToSelect()` 的 store sinking 实现。
* [x] 保持已有纯计算 if-conversion 行为不变。
* [x] 对“双方写相同地址”的情况继续优先使用：

```text
select + store
```

* [x] 对单侧 store 生成 `ADORA.cond_store`。
* [x] 对双方写不同地址的情况生成两个不同 predicate 的 `ADORA.cond_store`。
* [x] 对 `else if` 和 nested if 使用任务二产生的完整 path predicate。
* [x] 不为 `cond=false` 的单侧 store 再创建无必要的旧值 `memref.load`。
* [x] 保证 transformation 后 MLIR verifier 通过。
* [x] 在任何 mutation（包括常量 `truncf` 规整）前完成整 kernel preflight；
  lowering 与 optimized/fallback 两次 CDFG generation 在临时 kernel 上执行，
  只有完整成功后才提交，失败时原 IR 保持不变且不输出成功 DOT。
* [x] branch-local load、copy/call、nested region/loop 等不满足
  memory-effect-free 且 speculatable 的操作必须报错；`scf.for` 下 store-bearing
  `scf.if` 或预先存在的 `ADORA.cond_store` 同样 fail closed，`affine.for` 保持支持。

### CDFG lowering

* [x] 在 CDFG generation 中增加：

```text
ADORA.cond_store
        ↓
CSTORE
```

* [x] CSTORE operation name 必须与现有 Mapper contract 完全一致。
* [x] data / address / enable 的 port 必须使用 3.1 审计得到的真实编号。
* [x] 不自行定义新的 Mapper operand convention。

---

## 3.5 与已有 memref byte-offset 功能兼容

当前个人分支已经实现：

```text
memref.load / memref.store

element index
    ↓
CONST sizeof(element)
    ↓
MUL
    ↓
byte offset
```

新增 `cond_store` 后必须复用该逻辑。

### 待办

* [x] 抽取或复用现有地址换算 helper，避免复制一套 `cond_store` 专用地址计算代码。
* [x] `cond_store` 地址仍需从 element index 转换成 byte offset。
* [x] i32：

```text
index
  ↓
CONST 4
  ↓
MUL
  ↓
CSTORE address input
```

* [x] i8 / 1-byte element 不生成无意义：

```text
CONST 1 + MUL
```

* [x] synthetic `CONST / MUL` 节点已有的空 `operation` 防护继续有效。
* [x] 普通 `memref.store` 的现有行为不得发生回归。

### 特别注意

普通 `memref.store` 当前的 address port 约定**不能直接推导 CSTORE 的 address port**。

必须：

```text
地址计算方式
→ 复用已有 memref byte-offset 实现

CDFG destination port
→ 使用 CSTORE backend contract
```

---

## 3.6 Mapper / operation spec 修改规则

### 原则

**只修改真正缺失的后端部分，不重复实现已经存在的 CSTORE Mapper 支持。**

### 情况 A：Mapper + operation spec + ADG 已完整支持 CSTORE

如果确认：

```text
Mapper          ✓
operations spec ✓
UseEn           ✓
```

则：

* [ ] 不修改 Mapper 核心代码。
* [ ] 只补齐 ADORA IR、CDFG generation 和测试。

---

### 情况 B：Mapper 支持，但 operation spec 缺少 CSTORE

如果：

```text
Mapper          ✓
operations spec ✗
```

则：

* [x] 记录缺失项。
* [x] 不自行猜测：

  * OPC
  * latency
  * operand 数量
  * bitwidth
* [ ] 获得正确硬件配置后再补 operation spec。

在信息不足前，此项视为阻塞，不通过“随便填一个 OPC”绕过。

---

### 情况 C：ADG / IOB 没有 `UseEn`

如果实际硬件：

```text
UseEn ✗
```

则：

* [x] 不伪造 CSTORE end-to-end 成功。
* [x] 不使用普通 `ISEL + STORE` 冒充真正 conditional store，除非项目明确要求这种 lowering。
* [x] 保留已经完成的：

  * ADORA IR
  * CDFG frontend
  * testcase
* [x] 将硬件支持列为明确阻塞项。

---

## 3.7 正式测试

### Dialect / IR test

* [x] `ADORA.cond_store` 能正确 parse / print。
* [x] verifier 能拒绝错误 condition type。
* [x] condition 为正确类型时 verifier 通过。

### Control-flow lowering test

至少覆盖：

* [x] 单侧：

```c
if (cond)
    A[i] = x;
```

预期：一个 CSTORE。

* [x] 同地址双侧：

```c
if (cond)
    A[i] = x;
else
    A[i] = y;
```

预期：`SELECT + STORE`。

* [x] 不同地址双侧：

```c
if (cond)
    A[i] = x;
else
    B[i] = y;
```

预期：两个 CSTORE。

* [x] `if / else if / else`。
* [x] nested if。

### Predicate test

检查：

```text
a
!a
a && b
!a && b
!a && !b
```

等 path predicate 是否正确传播。

### Memref 地址测试

* [x] i32 cond_store 生成正确 `×4` 地址。
* [x] 1-byte element 不生成 `×1`。
* [x] CSTORE address edge 连接到真实 backend address port。
* [x] enable edge 连接到真实 backend enable port。
* [x] nested `affine.apply` 地址先完整 compose 再展开；每个 CSTORE 在写 DOT
  前必须通过 data/address/enable 端口与 memref metadata 完整性检查。
* [x] CSTORE 所在 execution region 的 mapped memory operation 按源码顺序保守
  串联，不能越过中间对其他 memref 的普通 memory operation。

### Regression

* [x] 原普通 `memref.load` 测试继续通过。
* [x] 原普通 `memref.store` 测试继续通过。
* [x] 原简单 if-conversion 测试继续通过。
* [x] 完整 `check-adora` 不产生新的 regression failure。

---

## 3.8 端到端验证

如果 3.1 确认当前 hardware spec 已完整支持 CSTORE，则至少运行一个：

```c
if (cond) {
    A[i] = x;
}
```

完整流程：

```text
C / MLIR
   ↓
if-conversion
   ↓
ADORA.cond_store
   ↓
CDFG CSTORE
   ↓
cgra-mapper
   ↓
configuration / emit
```

检查：

* [x] CDFG 中出现 `CSTORE`。
* [x] CSTORE 有正确 data input。
* [x] CSTORE 有正确 byte-address input。
* [x] CSTORE 有正确 enable input。
* [ ] Mapper 不报 `CSTORE is not supported`。
* [ ] Mapper 将 CSTORE 识别为 I/O / output node。
* [ ] IOB 配置正确启用：

  * `UseAddr`
  * `UseEn`
  * `IsStore`
* [ ] 如果已有仿真环境可运行，验证：

  * condition=true：内存发生写入。
  * condition=false：内存保持原值，且不是“重新写回旧值”。

---

## 预期成果

### 开发文档

```text
docs/development/cstore-backend-audit.md
```

明确记录现有 CSTORE backend contract 和缺口。

### ADORA Dialect

```text
include/ADORA/Dialect/ADORA/IR/ADORAOps.td
lib/Dialect/ADORA/IR/ADORAOps.cpp
```

新增稳定的：

```text
ADORA.cond_store
```

### CDFG

形成：

```text
ADORA.cond_store
→ CSTORE
```

完整 lowering。

### 控制流优化

现有 if-conversion 从：

```text
所有条件 store
→ LOAD old + SELECT + STORE
```

扩展为：

```text
纯计算
→ SELECT / ISEL

双方同地址 store
→ SELECT + STORE

真正条件 store
→ CSTORE
```

### 正式测试

新增 lit regression tests，覆盖：

```text
if / else-if / nested-if
+
cond_store
+
path predicate
+
memref byte offset
+
CSTORE
```

---

## 完成标准

任务三只有满足下面条件后才算完成：

* [x] 已完成 CSTORE backend audit。
* [x] 已明确 CSTORE data/address/enable 的真实 port contract。
* [x] 已明确当前 operation spec 是否支持 CSTORE（当前实际 spec 不支持）。
* [x] 已明确当前 ADG / IOB 是否支持 `UseEn`（当前实际 ADG / IOB 不支持）。
* [x] `ADORA.cond_store` 定义与现有 backend contract 对齐。
* [x] 单侧条件 store 能正确转换成 CSTORE。
* [x] 双侧相同地址 store 能继续使用 `SELECT + STORE`。
* [x] 双侧不同地址 store 能正确生成两个条件 CSTORE。
* [x] else-if / nested-if 使用完整 path predicate。
* [x] cond_store 正确复用已有 memref byte-offset 地址换算。
* [x] 正式 lit tests 全部通过。
* [x] 完整 `check-adora` 无新增失败。

如果当前硬件 spec 暂时缺少 CSTORE / UseEn，则允许将任务三拆分为：

```text
阶段 A（必须完成）
backend audit
+ ADORA.cond_store
+ control-flow lowering
+ CDFG CSTORE
+ regression tests

阶段 B（等待正确硬件接口后完成）
operation spec
+ Mapper end-to-end
+ simulation
```

**不得为了完成阶段 B 而自行猜测硬件 OPC、port 或 enable 语义。**

```
```


---

## 任务三当前状态

**阶段 A 已完成，任务四可以开始；阶段 B 仍然阻塞，最终上游 PR 必须等待真实硬件 contract。**

仓库中的实际 operation specs 均缺少 `CSTORE`；实际 ADG / IOB 仅提供两个
operand，且没有 `UseEn`。因此本轮只完成并验证了 backend audit、
`ADORA.cond_store`、控制流 lowering、规范化 CDFG ports（data/address/enable =
0/1/2）、完整 I/O metadata、静态 rank-1 边界、transactional fail-closed
lowering、memory source ordering、byte offset 和正式回归测试。当前 fp32 Mapper 尝试在这一硬件缺口处
失败，不能视为 Mapper 配置、emit 或 conditional-store suppression 仿真通过；
待获得真实的 CSTORE OPC/operation spec、三输入 IOB 和 `UseEn` 配置后再完成阶段 B。

---

# 任务四：整理 `experiment/jyhu`、完成全量验证并准备统一 PR

## 目标

清理长期开发产生的临时文件，将 `experiment/jyhu` 整理为便于继续实验但不会污染主仓库的个人实验区；随后完成全部功能的统一回归验证。

**只有任务一至任务三全部达到完成标准后，才进入 PR 阶段。**

---

## 4.1 整理 `experiment/jyhu`

原则：

> 分类整理，但不要大规模重构到无法辨认原来的实验内容。

### 第一步：先盘点，不直接删除

* [ ] 输出当前目录树：

```bash
find experiment/jyhu -maxdepth 3 -type f | sort
```

* [ ] 按下面几类标记现有文件：

  * 原始 `.c`
  * 手工编写 `.mlir`
  * 可复现实验脚本
  * CDFG `.dot`
  * mapper 输出
  * 临时日志
  * 自动生成中间 IR
  * 已废弃 testcase
  * 重复文件
  * 不确定用途文件

### 第二步：进行最小整理

建议结构：

```text
experiment/jyhu/
├── simple/
│   ├── simple.c
│   ├── simple.mlir
│   └── README.md
│
├── memref/
│   ├── ...
│   └── README.md
│
├── control-flow/
│   ├── if_simple/
│   ├── if_else/
│   ├── if_elseif_else/
│   ├── nested_if/
│   ├── if_load/
│   └── if_store/
│
├── scripts/
│   └── ...
│
└── archive/
    └── 仅保留仍有参考价值但不再使用的旧 testcase
```

不要求完全采用该结构；应以“少改名、易理解、能复现”为原则。

### 第三步：删除明确无价值的生成物

* [ ] 删除可以通过 pipeline 随时重新生成的旧：

  * `.dot`
  * mapper 临时结果
  * pipeline log
  * duplicated MLIR snapshots
  * 旧 `/tmp` 路径记录
  * 无价值 build artifacts
* [ ] 删除明显重复且已被新 testcase 替代的文件。
* [ ] 不确定用途的文件先移入 `archive/`，不要直接删除。

### 第四步：补最小说明

主要实验目录可以增加简短 `README.md`，只说明：

```text
这个 testcase 测什么
入口文件是什么
推荐运行命令是什么
期望观察什么
```

不要为个人实验目录写复杂文档。

### 提交规则

* [ ] `experiment/jyhu` 默认作为个人实验内容处理。
* [ ] **不要将大规模 `experiment/jyhu` 整理结果放入最终主仓库 PR。**
* [ ] 正式、稳定、需要长期回归的 testcase 应转成：

  * `test/` 下的 lit test
  * 或仓库已有正式 benchmark 目录
* [ ] `experiment/jyhu` 仅作为开发验证材料保留。

---

## 4.2 全量回归验证

在所有代码完成后统一执行。

### 专项测试

* [ ] memref load/store byte-offset test。
* [ ] simple if。
* [ ] if-else。
* [ ] if-else-if-else。
* [ ] nested if。
* [ ] cond_store。
* [ ] cond_store + i32 byte offset。
* [ ] cond_store + nested predicate。

### 完整 regression

* [ ] 运行：

```bash
cmake --build <build-dir> --target check-adora
```

* [ ] 将结果与原 `scheduletasks` baseline 对比。
* [ ] 已知 baseline failure 可以继续存在。
* [ ] **不得引入新的 regression failure。**

当前已知个人 memref 提交的基线结果：

```text
32 passed
8 unsupported
3 failed
```

其中 3 个失败与未修改 `scheduletasks` 基线一致。

最终验证时重新记录实际结果，不直接假定数字保持不变。

---

## 4.3 端到端验证

至少选择：

```text
simple memref case
+
simple conditional store case
+
if-else-if / nested if case
```

执行：

```text
C / MLIR
→ adoracc
→ task schedule
→ CDFG
→ mapper
```

在后端支持允许时继续：

```text
→ emit
→ simulator / runtime
```

检查：

* [ ] pipeline 无 crash。
* [ ] MLIR verifier 通过。
* [ ] CDFG 无 undefined node。
* [ ] memref byte offset 正确。
* [ ] condition edge/predicate 正确。
* [ ] cond_store operand 顺序正确。
* [ ] 普通 store 未被错误转成 cond_store。
* [ ] `if-else-if` 的 path predicate 正确。
* [ ] token dependency 与 predicate 没有相互混淆。

---

## 4.4 代码与提交整理

在准备 PR 前：

* [ ] `git status` 干净。
* [ ] 检查最终 diff，不包含：

  * `/tmp` 文件
  * build 产物
  * 大量个人实验结果
  * 无关 debug 输出
  * 无关格式化
  * unrelated benchmark changes
* [ ] 确认之前的 memref 功能提交仍然完整。
* [ ] 必要时将后续功能拆为逻辑清楚的小 commit，例如：

```text
1. add memref byte-offset support
2. extend control-flow path analysis
3. add ADORA cond_store IR and CDFG support
4. add cond_store mapper/backend support
5. add regression tests
```

不要为了提交历史好看而破坏已经验证过的提交；只在确有必要时整理历史。

---

# 最终 PR 条件

**不要在中途为当前 memref 功能单独创建 PR。**

只有以下条件全部满足后，统一创建一次 PR：

* [ ] 原 memref load/store byte-offset 功能仍通过。
* [ ] 简单 `if-else` 行为不回归。
* [ ] `if-else-if-else` 已支持并测试。
* [ ] nested if 最小场景已支持并测试。
* [ ] `ADORA.cond_store` IR 已稳定。
* [ ] `cond_store` CDFG generation 已稳定。
* [ ] Mapper/硬件相关语义已经明确并按真实接口实现。
* [ ] `cond_store` 地址换算与已有 memref byte-offset 逻辑兼容。
* [ ] 新增正式 lit regression tests。
* [ ] 完整 `check-adora` 无新增失败。
* [ ] 至少一个 conditional-store testcase 完成端到端验证。
* [ ] `experiment/jyhu` 已完成本地整理，但个人实验内容不进入主 PR。
* [ ] 最终 diff 中不存在无关修改。

最终 PR 应作为一个完整功能组提交，主题可以围绕：

```text
control-flow and conditional memref support
```

而不是单独只描述之前的 memref 地址换算。

---

# 推荐执行顺序

```text
任务一
控制流 testcase + baseline 分析
        ↓
任务二
if / else-if / nested-if 路径条件
        ↓
任务三
cond_store IR + CDFG
        ↓
等待/补齐后端语义
        ↓
cond_store Mapper / backend
        ↓
任务四
整理实验目录 + 全量回归
        ↓
统一 PR
```

当前如果后端语义仍未明确，优先推进：

```text
1. control-flow testcase
2. baseline 调研
3. if-else-if / nested-if path predicate
4. cond_store IR 前端定义与测试框架
5. experiment/jyhu 整理
```

暂缓：

```text
1. 猜测 LSU predicate port
2. 猜测 Mapper control-edge 语义
3. 自行定义最终硬件 opcode
4. conditional BlockLoad / BlockStore
5. 大规模 break / continue / switch 支持
```
