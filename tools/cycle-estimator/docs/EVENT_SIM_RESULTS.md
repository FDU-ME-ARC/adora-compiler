# 事件级仿真器 — 全部 8 个正式实验例子结果
# 生成命令: run.py --event-sim --mlir <x.final.mlir> --spec vitra_spec.json --viz g.png --viz-sram s.png
# spec=cgra_bf16, DMA_BPC=16 B/cyc, 4 SPAD bank x16KB, PE=16/tile

| 例子 | trip | events | makespan | overlap | conflicts | 备注 |
|---|---|---|---|---|---|---|
| atax     | 1  | 13   | 592   | 1.12x | 0 | ADORA op 跑一次, kernel 内 16x16 nest |
| attn     | 8  | 132  | 2248  | 1.26x | 0 | iter_args 携带 4 token (loop-carried) |
| cholesky | 16 | 195  | 4620  | 1.05x | 0 | 外层 for x16 |
| ffn      | 1  | 7    | 33412 | 1.01x | 0 | 单大 kernel (inner~32768), compute-bound |
| gesummv  | 64 | 768  | 8737  | 1.49x | 0 | 外层 for x64, 跨 kernel RAR |
| jacobi1d | 8  | 100  | 992   | 1.00x | 0 | 严格 load->kernel->store->load 链, 真串行(非bug) |
| sobel    | 62 | 1126 | 18732 | 1.54x | 0 | 最大例子, 12 load/iter, 强 overlap |
| viterbi  | 1  | 10   | 88    | 1.06x | 0 | 小例子 |

图: 每个例子 <name>_gantt.png (图A 资源甘特) + <name>_sram.png (图B SPAD 占用)
全部 0 SPAD 冲突. jacobi1d overlap=1.00 是因为 token 链强制串行(load 等前一个 store), 仿真忠实复现.
