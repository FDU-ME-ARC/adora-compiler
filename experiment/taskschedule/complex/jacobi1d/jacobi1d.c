/*
 * jacobi1d.c — Jacobi 1-D stencil (C reference, self-contained)
 *
 * 任务调度示例：T 步时间迭代，每步两遍 3-point stencil
 *   half-step a: B[i] = (A[i-1]+A[i]+A[i+1]) * c
 *   half-step b: A[i] = (B[i-1]+B[i]+B[i+1]) * c
 *
 * 迭代间是 loop-carried 依赖（第 t 步读 t-1 步的结果），
 * 演示循环携带依赖的调度（能否跨迭代流水重叠 BlockLoad / compute）。
 *
 * 维度写死，整型（用定点近似 1/3），自包含，可直接走 adoracc。
 */

#define N      32
#define TSTEPS 8

/* one half-step: out[i] = (in[i-1] + in[i] + in[i+1]) / 3   (integer approx) */
static void stencil(int in[N], int out[N])
{
#pragma scop
    for (int i = 1; i < N - 1; i++)
        out[i] = (in[i - 1] + in[i] + in[i + 1]) / 3;
#pragma endscop
}

/* Top-level — TSTEPS 步，每步 A->B 再 B->A（loop-carried） */
void jacobi1d(int A[N], int B[N])
{
    for (int t = 0; t < TSTEPS; t++) {
        stencil(A, B);   /* B = stencil(A) */
        stencil(B, A);   /* A = stencil(B) */
    }
}
