/*
 * viterbi.c — Viterbi Decoding (T=64 时间步，S=8 状态，整数对数概率)
 *
 * 任务调度特征：循环携带依赖（loop-carried dependency）
 *
 *   for t = 0..T-1:
 *     trellis[t][s] = max_over_prev { trellis[t-1][s'] + trans[s'][s] } + emit_score[t][s]
 *
 *   每个时间步 t 完全依赖 t-1 的结果（无法跨步并行）。
 *   但同一时间步内的 S 个状态更新可以完全并行。
 *
 * CGRA 映射设计：
 *   - emit_score[T_LEN][N_STATE] 由调用方预计算（CPU 侧查 obs 动态下标）
 *   - 消除了 emit[s][obs[t]] 中的动态下标，全部访问变为仿射（静态）模式
 *   - cgra-mapper 只看到纯 affine memref 访问，无 index_cast
 *
 * 调度依赖图（时间轴方向的链式依赖）：
 *   t=0 ─(RAW)─► t=1 ─(RAW)─► t=2 ─...─► t=63
 *   每步内 S=8 个状态并行
 */

#define T_LEN   64  /* 时间步 */
#define N_STATE  8  /* 状态数 */
#define LOG_ZERO (-2147483648)  /* 对数空间最小值 */

/*
 * viterbi():
 *   emit_score[T_LEN][N_STATE] — 预计算的发射分数（CPU 侧已完成 obs 查表）
 *   path[T_LEN]                — 各时刻最优路径分数输出
 *   trans[N_STATE][N_STATE]    — 转移分数 trans[from][to]
 */
void viterbi(int emit_score[T_LEN][N_STATE],
             int path[T_LEN],
             int trans[N_STATE][N_STATE])
{
    int trellis[T_LEN][N_STATE];

#pragma scop
    /* t=0: 初始化 */
    for (int s = 0; s < N_STATE; s++)
        trellis[0][s] = emit_score[0][s];

    /* t=1..T-1: 前向递推（loop-carried on t） */
    for (int t = 1; t < T_LEN; t++) {
        for (int s = 0; s < N_STATE; s++) {
            int best = LOG_ZERO;
            for (int sp = 0; sp < N_STATE; sp++) {
                int score = trellis[t-1][sp] + trans[sp][s];
                best = (score > best) ? score : best;
            }
            trellis[t][s] = best + emit_score[t][s];
        }
    }

    /* 读出最终时刻各状态分数（纯仿射访问，CGRA 映射目标） */
    for (int s = 0; s < N_STATE; s++)
        path[s] = trellis[T_LEN-1][s];
#pragma endscop
}
