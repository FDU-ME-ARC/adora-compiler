/*
 * gesummv.c — GESUMMV: y = alpha*(A*x) + beta*(B*x)
 *
 * 任务调度示例（菱形 fork-join）：
 *   Stage 1: tmp = A * x        ─┐  Stage1 ∥ Stage2 无依赖（读不同矩阵、写不同向量）
 *   Stage 2: y2  = B * x        ─┤  → 可分到不同 tile 并行交叠
 *   Stage 3: y = alpha*tmp + beta*y2 ─┘  AND-join：依赖 Stage1 AND Stage2
 *
 * ⚠️ 关键：必须写成【单个函数】。若拆成多个 C 函数，adoracc 会编成多个独立
 *    func.func，跨函数的 fork-join 依赖（merge 读 tmp/y2）在 schedule pass 的
 *    per-func dep 分析里看不到 → dep_type 标不出来（这是 fft/旧版 gesummv 的病根）。
 *    单函数内多 kernel（中间结果走局部数组）才能让 schedule 看到 RAW 依赖。
 *    对照：03_3mm 是单 func 多 kernel，dep 标得对。
 *
 * N=64，整型，自包含，可走 adoracc。
 */

#define N      64
#define ALPHA  3
#define BETA   2

void gesummv(int A[N][N], int B[N][N], int x[N], int y[N])
{
    int tmp[N];   /* Stage1 输出，Stage3 读 — 局部数组，单 func 内可见依赖 */
    int y2[N];    /* Stage2 输出，Stage3 读 */

#pragma scop
    /* Stage 1: tmp = A * x */
    for (int i = 0; i < N; i++) {
        int acc = 0;
        for (int j = 0; j < N; j++)
            acc += A[i][j] * x[j];
        tmp[i] = acc;
    }
    /* Stage 2: y2 = B * x  (independent of Stage 1) */
    for (int i = 0; i < N; i++) {
        int acc = 0;
        for (int j = 0; j < N; j++)
            acc += B[i][j] * x[j];
        y2[i] = acc;
    }
    /* Stage 3: y = alpha*tmp + beta*y2  (AND-join, reads tmp & y2) */
    for (int i = 0; i < N; i++)
        y[i] = ALPHA * tmp[i] + BETA * y2[i];
#pragma endscop
}
