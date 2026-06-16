/*
 * atax.c — ATAX kernel (C reference, self-contained, no polybench headers)
 *
 * 任务调度示例：y = A^T * (A * x)
 *   Stage 1: tmp = A * x        (MxN · N → M)
 *   Stage 2: y   = A^T * tmp    (NxM · M → N)
 *
 * Stage1 → Stage2 是 RAW 依赖（Stage2 读 Stage1 写的 tmp）。
 * 演示线性链（2 节点）调度：是否在两级间插 barrier。
 *
 * 维度写死，整型，自包含（对齐 complex/ 其它 kernel 的写法，可直接走 adoracc）。
 */

#define M 16
#define N 16

/* Stage 1: tmp[i] = sum_j A[i][j] * x[j] */
static void atax_mv(int A[M][N], int x[N], int tmp[M])
{
#pragma scop
    for (int i = 0; i < M; i++) {
        int acc = 0;
        for (int j = 0; j < N; j++)
            acc += A[i][j] * x[j];
        tmp[i] = acc;
    }
#pragma endscop
}

/* Stage 2: y[j] = sum_i A[i][j] * tmp[i]   (A^T * tmp) */
static void atax_mtv(int A[M][N], int tmp[M], int y[N])
{
#pragma scop
    for (int j = 0; j < N; j++) {
        int acc = 0;
        for (int i = 0; i < M; i++)
            acc += A[i][j] * tmp[i];
        y[j] = acc;
    }
#pragma endscop
}

/* Top-level — 2-stage serial chain */
void atax(int A[M][N], int x[N], int y[N])
{
    int tmp[M];
    atax_mv(A, x, tmp);    /* Stage 1: tmp = A * x   */
    atax_mtv(A, tmp, y);   /* Stage 2: y   = A^T*tmp  */
}
