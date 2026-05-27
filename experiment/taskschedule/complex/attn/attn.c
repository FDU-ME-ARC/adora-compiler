/*
 * attn.c — Scaled Dot-Product Attention kernel (C reference)
 *
 * 任务调度示例：展示 attention 的三阶段串行依赖关系
 *   Stage 1: score = Q * K^T          (GEMM: NxD × DxN → NxN)
 *   Stage 2: score = score / sqrt(D)   (element-wise scale)
 *   Stage 3: out   = softmax(score) * V (softmax + GEMM: NxN × NxD → NxD)
 *
 * 参数：
 *   N = SEQ_LEN = 8    (sequence length)
 *   D = HEAD_DIM = 16  (head dimension)
 *
 * Stage1 → Stage2 → Stage3 是严格顺序依赖，适合演示
 * CGRA 任务调度器如何在流水级边界插入同步屏障。
 */

#define SEQ_LEN  8
#define HEAD_DIM 16
/*
 * Stage 1: score[N][N] = Q[N][D] * K^T[D][N]
 *   score[i][j] = sum_k Q[i][k] * K[j][k]
 */
static void qk_matmul(
    int Q[SEQ_LEN][HEAD_DIM],
    int K[SEQ_LEN][HEAD_DIM],
    int score[SEQ_LEN][SEQ_LEN])
{
#pragma scop
    for (int i = 0; i < SEQ_LEN; i++)
        for (int j = 0; j < SEQ_LEN; j++) {
            int acc = 0;
            for (int k = 0; k < HEAD_DIM; k++)
                acc += Q[i][k] * K[j][k];
            score[i][j] = acc;
        }
#pragma endscop
}

/*
 * Stage 2: score[i][j] >>= SCALE_SHIFT  (integer approx of /sqrt(D))
 *   sqrt(16) = 4 → right-shift by 2
 */
#define SCALE_SHIFT 2

static void scale(int score[SEQ_LEN][SEQ_LEN])
{
#pragma scop
    for (int i = 0; i < SEQ_LEN; i++)
        for (int j = 0; j < SEQ_LEN; j++)
            score[i][j] >>= SCALE_SHIFT;
#pragma endscop
}

/*
 * Stage 3a: softmax approximation (integer, row-wise ReLU-normalize)
 *   For CGRA: use ReLU(x) / row_sum to avoid exp().
 *   weight[i][j] = max(score[i][j], 0) / row_sum_i
 *
 * Stage 3b: out[N][D] = weight[N][N] * V[N][D]
 *   out[i][d] = sum_j weight[i][j] * V[j][d]
 *
 * For integer-only CGRA: fuse weight_numerator directly:
 *   out[i][d] = sum_j relu(score[i][j]) * V[j][d]
 *   (row normalization deferred to host post-processing)
 */
static void attn_out(
    int score[SEQ_LEN][SEQ_LEN],
    int V[SEQ_LEN][HEAD_DIM],
    int out[SEQ_LEN][HEAD_DIM])
{
#pragma scop
    for (int i = 0; i < SEQ_LEN; i++)
        for (int d = 0; d < HEAD_DIM; d++) {
            int acc = 0;
            for (int j = 0; j < SEQ_LEN; j++) {
                int w = score[i][j] > 0 ? score[i][j] : 0;  /* ReLU */
                acc += w * V[j][d];
            }
            out[i][d] = acc;
        }
#pragma endscop
}

/*
 * Top-level kernel — 三阶段串行流水
 */
void attention(
    int Q[SEQ_LEN][HEAD_DIM],
    int K[SEQ_LEN][HEAD_DIM],
    int V[SEQ_LEN][HEAD_DIM],
    int out[SEQ_LEN][HEAD_DIM])
{
    int score[SEQ_LEN][SEQ_LEN];

    qk_matmul(Q, K, score);   /* Stage 1: Q*K^T        */
    scale(score);              /* Stage 2: /sqrt(D)      */
    attn_out(score, V, out);  /* Stage 3: softmax * V   */
}