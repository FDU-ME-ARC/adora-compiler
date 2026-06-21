# 事件级仿真器 — event_build 严谨重写设计（REDESIGN）

> 起因：现有 `extract/event_build.py` 用 `find_outer`（只取第一个 affine.for）+ 扁平
> template，**嵌套 affine.for 直接算错**。这是正确性问题，不是优化问题。
> 决定：从基底重写 MLIR → Event 的解析，**严格按 affine 嵌套递归展开**，事件线性增加
> （嵌套 = trip 乘积，本就该这么多），不背历史包袱。run 久的问题（大 trip）暂不管。

---

## 0. 核心原则

把 event_build 写成一个**对 func body 的解释器（interpreter）**：按程序序递归遍历，
遇到 `affine.for` 就按 trip 循环展开其 body，遇到 ADORA op 就 emit 一个 Event。
维护一个 **SSA 环境** + 用编译器算好的 **内存依赖摘要**，两条腿建依赖。

---

## 1. 依赖来源：两类，分别用两个权威源（查证过）

### (a) SSA def-use（token + memref 结果）→ 用环境 env
`kernel async [%tok...]`、`BlockLoad async [%tok]`、kernel region 读 `%result`、
store 读 kernel 输出 `%1`：这些是 **SSA 值的 def-use**，在**同一动态实例**内连边。
解析方式：解释器维护 `env: 静态SSA Value -> 最近产生它的 event eid`，
- emit 一个 op 时，把它 result 的 SSA Value 写进 env（覆盖）。
- 消费 op 时，对每个 operand 用 `OpResult.isinstance(v)` + `v.owner` 找 def，
  再查 env 拿到**当前执行路径**上的 producer eid → 连边。
- 跨迭代：env 不在迭代间清空，所以 iter i+1 读到 iter i 写的值时自然连成 loop-carried SSA 边。

### (b) memref 内存依赖（RAW/WAR/WAW/RAR）→ 用编译器的 `adora.dep_summary`
查证：scheduled MLIR 的 FuncOp 上挂着
```
adora.dep_summary = [{block_idx=0, edges=[{src,dst,kind:"RAW"|..,overlap}...]}]
adora.lc_dep_summary = [{edges=[{kind:"LC-RAR",step,exact}...], loop_idx, loop_op}]
```
- `dep_summary.edges` 的 src/dst 是**块内 ADORA-op 的序号**（按 block 内出现顺序，
  只数 BlockLoad/LocalMemAlloc/kernel/BlockStore 这些被建进 task graph 的 op）。
  这是编译器 `DependencyAnalysis` 用 mayOverlapBoxesByQuadruple 算好的**内存别名依赖**，
  **不要自己再推 aliasing**——直接消费。
- `lc_dep_summary` 是 loop-carried（跨迭代）内存依赖，step=迭代距离。
- 映射：把每个 ADORA op 实例按"块内序号"对齐 dep_summary 的 src/dst；
  同迭代内连 (src→dst)；lc 边连 (iter it 的 src → iter it+step 的 dst)。

> ⚠️ 注意：gesummv 里 token 已经把 load→kernel→store 显式连了；dep_summary 的 RAW
> 大多与 token 重合。但**通用程序里 token 不一定全**（标量 glue、跨 kernel WAR/WAW），
> 所以两条腿都要建，取并集，去重。

---

## 2. 解释器算法（递归，严格按 affine 嵌套）

### 三个例子揭示的真实结构（查证 gesummv / tri / attn）
- **gesummv**：外层 `affine.for 0..64` **包住** ADORA ops → 事件展开 ×64；kernel 内层 for → cost。
- **tri**：ADORA ops 在 func 顶层执行**一次**；affine.for 全在 **kernel 体内**（compute loop）→ 只算 cost，不展开成事件。
- **attn**：外层 `affine.for 0..8 iter_args(%t1..%t4)` 携带 **4 个 async token 跨迭代**；
  `affine.yield %9,%9,%12,%12` 把本迭代的 store token 传给下迭代；
  `BlockStore async [%8,%arg5,%arg7]` 依赖**上一迭代**的 store（loop-carried token）。
  还有 `ADORA.event.create -> !ADORA.token` 在循环前造初始 token。

→ 关键区分：**func 层的 affine.for = 事件展开循环**；**kernel 体内的 affine.for = compute，
  只进 cost**。解释器**遇到 ADORA.kernel 就停**（不递归进它的 region 展开事件）。

### 解释器（SSA 环境 + iter_args/yield）
```
env: dict[SSA Value -> 产生它的 eid]      # 标量/非事件 producer -> 不连边(视为 ready)

walk(region, env, outer_it):
  for op in region 程序序:
    if op is affine.for:
      trip = _for_trip(op)
      carried = [producer(init) for init in op.iter_args_init]   # 初始携带值
      for it in range(trip):
        local = env.copy()
        bind op.region block_args[iter_arg] -> carried[i]        # 上一迭代 yield
        walk(op.body, local, outer_it_for_this_loop(it))
        carried = [producer_via local(y) for y in affine.yield 的操作数]  # 传给下迭代
      bind op.results -> carried[i]                              # 循环结果 = 末次 carried
    elif op is ADORA.kernel:
      e = emit KERNEL; cost = II * Π(kernel 体内 affine.for trips) + drain
      deps: 对 async operand 查 env -> link; do NOT 递归 body
      env[op.result] = e.eid
    elif op is ADORA.BlockLoad/LocalMemAlloc/BlockStore:
      e = emit LOAD/ALLOC/STORE; bytes 从 memref 类型
      deps: 对 async/memref operand 查 env -> link
      env[op.results] = e.eid
    elif op is ADORA.event.create:
      e = emit 零代价 marker(NONE); env[op.result] = e.eid     # 初始 token
    elif op is affine.yield / ADORA.terminator:
      pass                                                      # 父循环处理
    else:  # 标量 glue (arith / memref.alloca / affine.load/store)
      env[op.results] = SENTINEL_READY                          # 非事件，依赖时跳过
```
- producer(v)：v 是 OpResult 且 owner 是事件 op → env[v]；否则（标量/块外）→ 视为 ready，不连边。
- **事件数 = func 层 affine.for 嵌套下 ADORA op 的动态实例数 = trip 乘积**。线性，正确。
- **loop-carried 依赖自动正确**：iter_args 把 block arg 绑到上一迭代 yield 的 producer，
  跨迭代 token 边（attn 的 prologue/steady/epilogue）天然连出。

### bank slot 索引
每个 buffer 流（KernelName.Id）维护一个发射序号 `seq`（第几次被 load/alloc），
`slot = seq % depth`；slot-recycle 边 reader(seq-depth)→load(seq) 不变。
Event.it（标签用）= 最外层展开循环的迭代号。

### dep_summary（内存依赖）：v1 先不连，标注 TODO
查证：gesummv/tri/attn 的 token 链（含 iter_args 携带）已**完整表达调度意图的时序**，
v1 仅用 SSA-env 即可正确。`adora.dep_summary`/`lc_dep_summary` 的 WAR/WAW 多与 token 重合，
留作 v2 augmentation（需先核 ScheduleAdoraTasks 的 op 序号口径，见 §5）。

## 3. 资源 / bank / cost：复用已写好的

- Event/Opcode/ResKind/Timeline：`core/event.py`（不动）。
- BankAllocator cur/old/older + SRAMAccess：`core/sram_track.py`（不动）。
- 离散事件 list-scheduler + hang 检测：`core/event_sim.py`（不动）。
- spec 参数：`arch/spec.py`（不动）。
- slot-recycle 跨迭代边：保留（reader(it-depth)→load(it)），与 §2 的依赖图叠加。
- **只重写 `extract/event_build.py`**：从 template/find_outer 换成 §2 的解释器。

---

## 4. 验证（重写后必须全过）

1. gesummv（单层 trip=64）：events=768、跨 kernel E8←E2、makespan>4480、overlap>1、0 冲突
   —— 与当前结果**逐一对齐**（回归不退化）。
2. 嵌套用例：找一个 `for i: for j: kernel` 的 scheduled MLIR（如 gemm/2mm 多层），
   断言 events = trip_i × trip_j × op 数，且依赖跨双层正确。
3. dep_summary 消费：构造一个 token 不全但有 WAR 的例子，验证内存边补上了。

---

## 5. 待解析细节（重写时确认）

- dep_summary 的 src/dst 序号到底数哪些 op（只 task-graph op 还是全部）？
  对照 C++ `ScheduleAdoraTasks.cpp::generateTaskGraphFromBlock` 的建图顺序确认。
- 多层 affine.for 时 Event.it 用 flatten 全局号还是 tuple？建议 flatten（= 当前外层迭代序），
  bank slot 轮转按 flatten 号取模即可。
- env 的 key：MLIR-py Value 可 hash/eq（已验证 `==` 是指针相等）；用 Value 直接做 dict key。
- kernel inner_trip / II / drain：当前 II=1/drain=4 占位；可后续接 ii_model+dot。

---

## 6. 一句话

把"猜一个外层 + 扁平模板"换成"像解释器一样严格按 affine 嵌套递归展开 + env 连 SSA 边 +
dep_summary 连内存边"。事件线性增长是对的（嵌套本就是乘积），正确性优先，性能以后再说。
