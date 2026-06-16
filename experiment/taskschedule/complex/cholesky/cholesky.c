/*
 * cholesky.c — Cholesky-style triangular update (RECTANGULAR-loop variant)
 *
 * 任务调度示例：A = L·Lᵀ 风格的两阶段累减分解。
 *   Stage 1 (off-diag): A[i][j] -= sum_k A[i][k]*A[j][k]
 *   Stage 2 (diag)    : A[i][i] -= sum_k A[i][k]^2
 *
 * 调度特征（这才是可视化关心的）：
 *   - 2 个 stage 串行（Stage1 写 A，Stage2 读 A → RAW 依赖）
 *   - 每个元素是一个 inner-product 归约（累加依赖链）
 *
 * ⚠️ 与教科书 Cholesky 的区别：原算法内层是三角边界（j<i, k<j），但
 *    `--adora-kernel-dfg-gen` 对非矩形迭代空间会 segfault（见 SWEEP_RESULTS B4）。
 *    这里改成全矩形循环（0..N），语义上多累加了三角形外的项、且省去 sqrt/除法，
 *    不再是数值精确的 Cholesky；但保留了"2-stage + 归约依赖"的调度结构，
 *    作为强依赖调度样本的可视化代表足够。需要数值正确版时另走非 CGRA 路径。
 *
 * 维度写死，整型，自包含，可走 adoracc。
 */

#define N 16

/* Stage 1: off-diagonal-style accumulate-subtract (rectangular) */
static void chol_offdiag(int A[N][N])
{
#pragma scop
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            int acc = A[i][j];
            for (int k = 0; k < N; k++)
                acc -= A[i][k] * A[j][k];
            A[i][j] = acc;
        }
#pragma endscop
}

/* Stage 2: diagonal-style accumulate-subtract (rectangular, no sqrt) */
static void chol_diag(int A[N][N])
{
#pragma scop
    for (int i = 0; i < N; i++) {
        int acc = A[i][i];
        for (int k = 0; k < N; k++)
            acc -= A[i][k] * A[i][k];
        A[i][i] = acc;
    }
#pragma endscop
}

/* Top-level — off-diagonal then diagonal (Stage1 → Stage2 RAW) */
void cholesky(int A[N][N])
{
    chol_offdiag(A);   /* Stage 1 */
    chol_diag(A);      /* Stage 2 */
}
