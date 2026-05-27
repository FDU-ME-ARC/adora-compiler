# adoracc.py 重构计划

> 文档路径：`docs/adoracc_refactor_plan.md`  
> 涉及文件：`tools/adoracc/adoracc.py`（主源），`build/bin/adoracc.py`（同步目标）  
> 状态：📋 待实施

---

## 一、背景与目标

### 现存问题

| # | 问题描述 |
|---|---|
| 1 | 中间产物散落在 `tempfiles/` 隐藏目录，调试时难以快速定位各阶段输出 |
| 2 | `pipeline.log` 埋在 `tempfiles/` 里，报错时提示的路径不直观 |
| 3 | normalize 产物（`*_normalized.mlir`）与 DFG-gen 产出（`*_CDFG.dot`）混在同一个 `tempfiles/DFGs/` 下 |
| 4 | schedule-tasks 的 pre/post 两份 IR 都扔进 `1_kernels_opt/`，命名靠后缀 `_opt` / `_sched` 区分，不直观 |
| 5 | cgeist 输出（`.mlir` frontend 结果）直接放在 `ir_dir` 根级，没有阶段编号 |
| 6 | `tempfiles/DesignSpace`（auto-unroll 产物）位置随 cwd 隐式决定，缺乏显式管理 |
| 7 | 编译失败时 log 路径提示指向 `tempfiles/`，已被删除时无法找到 |

### 重构目标

1. **目录结构清晰化**：所有阶段输出按编号子目录存放，一眼看出流水线顺序。
2. **消灭 `tempfiles/`**：所有中间产物提升到有意义的命名目录；`DesignSpace` 显式放在 `adora-cc-ir/` 根下。
3. **`pipeline.log` 提升到根**：`adora-cc-ir/pipeline.log`，报错提示直接指向此处。
4. **schedule-tasks 产物分离**：`5_task-schedule/` 单独存放 `.pre.mlir` / `.post.mlir`，失败时保留 `.post.failed.mlir`。
5. **终端输出优化**：成功时打印各阶段路径摘要；失败时同时输出 log 路径与失败产物路径。

---

## 二、目录结构对比

### 当前结构

```
adora-cc-ir/
├── 0_kernels/
│   └── {name}_kernel.mlir          # kernel-extract 输出
├── 1_kernels_opt/
│   ├── {name}_opt.mlir             # kernel-opt 输出（schedule 前）
│   └── {name}_sched.mlir           # schedule-tasks 输出
├── 2_dfgs/
│   └── {name}_CDFG.dot             # DFG 最终产物
├── tempfiles/
│   ├── pipeline.log                # 所有子命令日志
│   └── DFGs/
│       └── {name}_normalized.mlir  # normalize 输出
│       └── {name}_CDFG.dot         # DFG-gen 原始输出（再 copy 到 2_dfgs/）
└── {name}.mlir                     # cgeist 输出（.C 输入时）
```

### 目标结构

```
adora-cc-ir/
├── pipeline.log                    # ★ 提升到根级
├── 1_frontend/
│   └── {name}.mlir                 # cgeist 输出（.C 输入时）
├── 2_normalize/
│   └── {name}_normalized.mlir      # normalize 输出
├── 3_kernel-extract/
│   └── {name}_kernel.mlir          # kernel-extract 输出
├── 4_kernel-opt/
│   └── {name}_opt.mlir             # kernel-opt 输出
├── 5_task-schedule/
│   ├── {name}.pre.mlir             # schedule 前（kernel-opt 的拷贝）
│   └── {name}.post.mlir            # schedule 成功输出
│   （或）
│   └── {name}.post.failed.mlir     # schedule 失败时的原样拷贝 + 标记
├── 6_dfg/
│   └── {name}_CDFG.dot             # DFG 最终产物
└── DesignSpace/                    # auto-unroll 产物（cwd 显式指向此处）
```

---

## 三、函数级改动详述

### 3.1 `prepare_ir_dirs(root)`

**当前返回 key 与目录：**

| key | 当前目录 |
|---|---|
| `"ir"` | `adora-cc-ir/` |
| `"kernels"` | `adora-cc-ir/0_kernels/` |
| `"kernels_opt"` | `adora-cc-ir/1_kernels_opt/` |
| `"dfgs"` | `adora-cc-ir/2_dfgs/` |
| `"tempfiles"` | `adora-cc-ir/tempfiles/` |
| `"temp_dfg"` | `adora-cc-ir/tempfiles/DFGs/` |

**重构后返回 key 与目录：**

| key | 新目录 | 说明 |
|---|---|---|
| `"ir"` | `adora-cc-ir/` | 不变；`pipeline.log` 写入此处 |
| `"frontend"` | `adora-cc-ir/1_frontend/` | 新增 |
| `"normalize"` | `adora-cc-ir/2_normalize/` | 原 `temp_dfg` |
| `"kernels"` | `adora-cc-ir/3_kernel-extract/` | 原 `0_kernels` |
| `"kernels_opt"` | `adora-cc-ir/4_kernel-opt/` | 原 `1_kernels_opt` |
| `"schedule"` | `adora-cc-ir/5_task-schedule/` | 新增 |
| `"dfgs"` | `adora-cc-ir/6_dfg/` | 原 `2_dfgs` |

**删除 key：** `"tempfiles"`, `"temp_dfg"`

**代码改动（伪 diff）：**

```python
# 旧
kernels_dir     = ir_dir / "0_kernels"
kernels_opt_dir = ir_dir / "1_kernels_opt"
dfgs_dir        = ir_dir / "2_dfgs"
tempfiles_dir   = ir_dir / "tempfiles"
temp_dfg_dir    = tempfiles_dir / "DFGs"

for directory in (kernels_dir, kernels_opt_dir, dfgs_dir, temp_dfg_dir):
    directory.mkdir(parents=True, exist_ok=True)

return {
    "ir": ir_dir, "kernels": kernels_dir,
    "kernels_opt": kernels_opt_dir, "dfgs": dfgs_dir,
    "tempfiles": tempfiles_dir, "temp_dfg": temp_dfg_dir,
}

# 新
frontend_dir    = ir_dir / "1_frontend"
normalize_dir   = ir_dir / "2_normalize"
kernels_dir     = ir_dir / "3_kernel-extract"
kernels_opt_dir = ir_dir / "4_kernel-opt"
schedule_dir    = ir_dir / "5_task-schedule"
dfgs_dir        = ir_dir / "6_dfg"

for directory in (frontend_dir, normalize_dir, kernels_dir,
                  kernels_opt_dir, schedule_dir, dfgs_dir):
    directory.mkdir(parents=True, exist_ok=True)

return {
    "ir": ir_dir, "frontend": frontend_dir,
    "normalize": normalize_dir, "kernels": kernels_dir,
    "kernels_opt": kernels_opt_dir, "schedule": schedule_dir,
    "dfgs": dfgs_dir,
}
```

---

### 3.2 `build_pipeline(...)` — `log_dir` 绑定

**当前：**
```python
log_dir = dirs["tempfiles"]
```

**新：**
```python
log_dir = dirs["ir"]   # pipeline.log 写到 adora-cc-ir/ 根
```

影响所有 `run_command(..., log_dir=log_dir)` 调用，无需逐一修改。

---

### 3.3 `build_pipeline(...)` — cgeist 段（Step 0 → Step 1）

**当前：** 输出直接写到 `dirs["ir"] / f"{base_name}.mlir"`

**新：** 输出写到 `dirs["frontend"] / f"{base_name}.mlir"`

```python
# 旧
cgeist_output = dirs["ir"] / f"{base_name}.mlir"

# 新
cgeist_output = dirs["frontend"] / f"{base_name}.mlir"
```

`strip_module_attrs` 的原地覆写路径随之跟随，无需单独修改。

---

### 3.4 `build_pipeline(...)` — normalize 段（Step 1 → Step 2）

**当前：** `normalized = dirs["temp_dfg"] / f"{base_name}_normalized.mlir"`

**新：** `normalized = dirs["normalize"] / f"{base_name}_normalized.mlir"`

```python
# 旧
normalized = dirs["temp_dfg"] / f"{base_name}_normalized.mlir"

# 新
normalized = dirs["normalize"] / f"{base_name}_normalized.mlir"
```

---

### 3.5 `build_pipeline(...)` — kernel-extract 段（Step 2 → Step 3）

目录 key 已从 `"kernels"` 映射到新目录，代码无需改动（key 名不变）。

```python
kernel_mlir = dirs["kernels"] / f"{base_name}_kernel.mlir"  # 不变，路径自动变
```

---

### 3.6 `build_pipeline(...)` — kernel-opt 段（Step 3 → Step 4）

目录 key 已从 `"kernels_opt"` 映射到新目录，代码无需改动。

**但 auto-unroll 的 cwd 需要更新：**

```python
# 旧
run_command(
    kernel_opt_cmd,
    cwd=dirs["tempfiles"] if enable_unroll else None,
    ...
)

# 新：DesignSpace 落在 adora-cc-ir/ 根下
run_command(
    kernel_opt_cmd,
    cwd=dirs["ir"] if enable_unroll else None,
    ...
)
```

---

### 3.7 `build_pipeline(...)` — schedule-tasks 段（Step 4 → Step 5）⭐ 最大改动

#### 当前行为
- pre 产物：`4_kernel-opt/{name}_opt.mlir`（就地不复制）
- post 产物：`1_kernels_opt/{name}_sched.mlir`
- 失败：`kernel_sched = kernel_opt`（静默回退，无标记）

#### 新行为
- pre 产物：`5_task-schedule/{name}.pre.mlir`（从 kernel-opt 拷贝）
- post 产物：`5_task-schedule/{name}.post.mlir`（schedule 成功输出）
- 失败产物：`5_task-schedule/{name}.post.failed.mlir`（内容同 pre，名字标记失败）
- 失败时 `kernel_sched` 仍回退到 kernel-opt 输出（语义不变）

**伪代码：**

```python
kernel_sched = kernel_opt  # fallback
if schedule_tasks:
    pre_mlir  = dirs["schedule"] / f"{base_name}.pre.mlir"
    post_mlir = dirs["schedule"] / f"{base_name}.post.mlir"

    # 拷贝 pre
    shutil.copy(kernel_opt, pre_mlir)

    sched_cmd = [
        tools["cgra-opt"],
        "--adora-schedule-tasks",
        str(pre_mlir),
        "-o", str(post_mlir),
    ]
    result = subprocess.run(sched_cmd, capture_output=True, text=True)

    # 写 schedule 命令到 log（保持与 run_command 一致的格式）
    _append_to_log(dirs["ir"] / PIPELINE_LOG_NAME, sched_cmd, result)

    if result.returncode != 0:
        failed_mlir = dirs["schedule"] / f"{base_name}.post.failed.mlir"
        shutil.copy(pre_mlir, failed_mlir)
        print(
            f"[adoracc] Warning: adora-schedule-tasks failed on {kernel_opt.name};\n"
            f"  failed IR saved to: {failed_mlir}\n"
            f"  See log: {dirs['ir'] / PIPELINE_LOG_NAME}\n"
            + result.stderr,
            file=sys.stderr,
        )
        kernel_sched = kernel_opt  # fallback
    else:
        kernel_sched = post_mlir
```

> `_append_to_log` 是一个私有小函数，把 schedule 命令的 stdout/stderr 追加写入 `pipeline.log`，格式与 `run_command` 保持一致。

---

### 3.8 `build_pipeline(...)` — DFG-gen 段（Step 5 → Step 6）

DFG-gen 用 `dirs["normalize"]` 替换原 `dirs["temp_dfg"]`（glob 来源改变），
`dirs["dfgs"]` key 不变但现在映射到 `6_dfg/`：

```python
# 旧
for dot_file in dirs["temp_dfg"].glob("*_CDFG.dot"):
    shutil.copy(dot_file, dirs["dfgs"] / dot_file.name)

# 新
for dot_file in dirs["normalize"].glob("*_CDFG.dot"):
    shutil.copy(dot_file, dirs["dfgs"] / dot_file.name)
```

> DFG-gen 的 cwd 已在上一轮修复为 `adora_compiler_root`，此处不变。

---

### 3.9 `main()` — 错误处理与终端输出

#### 错误时 log 路径提示

```python
# 旧
print(f"See subprocess log: {dirs['tempfiles'] / PIPELINE_LOG_NAME}", file=sys.stderr)

# 新
print(f"See pipeline log:   {dirs['ir'] / PIPELINE_LOG_NAME}", file=sys.stderr)
```

#### 成功时输出摘要

```python
# 旧
print(f"Final optimal mlir file: {dirs['kernels_opt']}", file=sys.stderr)
print(f"CDFG output directory: {dirs['dfgs']}", file=sys.stderr)

# 新（打印各阶段路径，并标注 schedule 是否生效）
sched_status = "enabled" if args.schedule_tasks else "disabled"
print("", file=sys.stderr)
print("[adoracc] Pipeline completed successfully.", file=sys.stderr)
print(f"  frontend IR  : {dirs['frontend']}", file=sys.stderr)
print(f"  normalize    : {dirs['normalize']}", file=sys.stderr)
print(f"  kernel-extract: {dirs['kernels']}", file=sys.stderr)
print(f"  kernel-opt   : {dirs['kernels_opt']}", file=sys.stderr)
print(f"  task-schedule: {dirs['schedule']}  [{sched_status}]", file=sys.stderr)
print(f"  dfg          : {dirs['dfgs']}", file=sys.stderr)
print(f"  pipeline log : {dirs['ir'] / PIPELINE_LOG_NAME}", file=sys.stderr)
```

---

## 四、新增辅助函数

### `_append_to_log(log_path, cmd, result)`

将 schedule-tasks 的执行记录追加到 `pipeline.log`，格式与 `run_command` 一致：

```python
def _append_to_log(
    log_path: Path,
    cmd: list[str],
    result: subprocess.CompletedProcess,
) -> None:
    with open(log_path, "a", encoding="utf-8") as logf:
        logf.write("\n" + "=" * 72 + "\n")
        logf.write(datetime.now().isoformat(timespec="seconds") + "\n")
        logf.write("+ " + " ".join(cmd) + "\n")
        logf.write("-" * 72 + "\n")
        if result.stdout:
            logf.write(result.stdout)
        if result.stderr:
            logf.write(result.stderr)
```

---

## 五、改动范围汇总

| 函数 | 改动类型 | 预估行数 |
|---|---|---|
| `prepare_ir_dirs()` | 重写目录定义与返回值 | ~15 行 |
| `build_pipeline()` — log_dir | 1 行替换 | 1 行 |
| `build_pipeline()` — cgeist 段 | 1 行路径替换 | 1 行 |
| `build_pipeline()` — normalize 段 | 1 行路径替换 | 1 行 |
| `build_pipeline()` — kernel-opt cwd | 1 行替换 | 1 行 |
| `build_pipeline()` — schedule-tasks 段 | 重写整段 | ~25 行 |
| `build_pipeline()` — DFG glob 来源 | 1 行替换 | 1 行 |
| `main()` — log 路径提示 | 1 行替换 | 1 行 |
| `main()` — 成功摘要输出 | 扩展输出 | ~10 行 |
| `_append_to_log()` | 新增函数 | ~12 行 |
| **合计** | | **~68 行** |

---

## 六、不在本次改动范围内

| 项目 | 说明 |
|---|---|
| `parse_args()` | 无需改动 |
| `strip_module_attrs()` | 无需改动 |
| `has_adora_kernel()` | 无需改动 |
| `require_tool*()` | 无需改动 |
| `run_command()` | 无需改动 |
| DFG-gen cwd 修复 | 已完成 |
| `build/bin/` 同步 | 改完后执行 `ninja adoracc.py` 或手动 `cp` |

---

## 七、同步到 build 目录

`tools/adoracc/adoracc.py` 由 CMake 安装到 `build/bin/adoracc.py`。
改完后执行：

```bash
cd /data00/home/loujiahang/adora/adora-compiler/build
ninja adoracc.py
# 或手动同步：
cp tools/adoracc/adoracc.py build/bin/adoracc.py
```

> `build/bin/adoracc.py` 需与 `tools/adoracc/adoracc.py` 始终保持一致。

---

## 八、验证清单

改完后用以下命令验证：

```bash
# 1. 用一个简单 .mlir 文件跑完整流水线
adoracc.py test.mlir --work-dir /tmp/test_adoracc

# 2. 检查目录结构
ls /tmp/test_adoracc/adora-cc-ir/

# 3. 确认各阶段文件存在
ls /tmp/test_adoracc/adora-cc-ir/1_frontend/
ls /tmp/test_adoracc/adora-cc-ir/2_normalize/
ls /tmp/test_adoracc/adora-cc-ir/3_kernel-extract/
ls /tmp/test_adoracc/adora-cc-ir/4_kernel-opt/
ls /tmp/test_adoracc/adora-cc-ir/5_task-schedule/
ls /tmp/test_adoracc/adora-cc-ir/6_dfg/
cat /tmp/test_adoracc/adora-cc-ir/pipeline.log | head -30

# 4. 验证 tempfiles/ 不再存在
ls /tmp/test_adoracc/adora-cc-ir/tempfiles/ 2>&1  # 应报错 No such file

# 5. 测试 schedule-tasks 失败时的降级行为（构造一个会失败的 kernel）
adoracc.py bad_kernel.mlir --work-dir /tmp/test_adoracc2
ls /tmp/test_adoracc2/adora-cc-ir/5_task-schedule/  # 应有 .post.failed.mlir
```

---

*最后更新：自动生成*
