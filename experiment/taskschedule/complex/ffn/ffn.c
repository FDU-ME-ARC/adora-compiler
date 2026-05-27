/*
 * ffn.c — Two-layer Feed-Forward Network (Transformer FFN block)
 *
 * Architecture:
 *   hidden[B][H] = ReLU( sum_i  input[B][I] * W1[I][H] )   Stage 1: FC1 + ReLU
 *   out[B][O]    =       sum_h  hidden[B][H] * W2[H][O]     Stage 2: FC2
 *
 * Dimensions (matching polybench SMALL-class):
 *   B = BATCH    = 16   (sequence positions / batch)
 *   I = INPUT_DIM = 32  (input features)
 *   H = HIDDEN_DIM = 64 (FFN hidden dim, typically 4×input)
 *   O = OUTPUT_DIM = 32 (= INPUT_DIM for residual-compatible output)
 *
 * Data type: int32 (fixed-point, consistent with bicg/gemm sweep)
 *
 * Task scheduling dependency:
 *   Stage1 (FC1+ReLU) writes hidden[16×64]
 *   Stage2 (FC2)      reads  hidden[16×64]   ← RAW: Stage1 must finish first
 *
 * All arrays are function parameters (no local intermediate alloca),
 * enabling adoracc/cgeist to produce clean affine.load/affine.store dialect.
 */

#define BATCH       16
#define INPUT_DIM   32
#define HIDDEN_DIM  64
#define OUTPUT_DIM  32

/*
 * Stage 1: hidden[b][h] = ReLU( sum_i input[b][i] * W1[i][h] )
 */
void ffn_fc1(
    int input [BATCH][INPUT_DIM],
    int W1    [INPUT_DIM][HIDDEN_DIM],
    int hidden[BATCH][HIDDEN_DIM])
{
#pragma scop
    for (int b = 0; b < BATCH; b++)
        for (int h = 0; h < HIDDEN_DIM; h++) {
            int acc = 0;
            for (int i = 0; i < INPUT_DIM; i++)
                acc += input[b][i] * W1[i][h];
            hidden[b][h] = acc > 0 ? acc : 0;   /* ReLU */
        }
#pragma endscop
}

/*
 * Stage 2: out[b][o] = sum_h hidden[b][h] * W2[h][o]
 * RAW dependency: reads hidden written in Stage 1
 */
void ffn_fc2(
    int hidden[BATCH][HIDDEN_DIM],
    int W2    [HIDDEN_DIM][OUTPUT_DIM],
    int out   [BATCH][OUTPUT_DIM])
{
#pragma scop
    for (int b = 0; b < BATCH; b++)
        for (int o = 0; o < OUTPUT_DIM; o++) {
            int acc = 0;
            for (int h = 0; h < HIDDEN_DIM; h++)
                acc += hidden[b][h] * W2[h][o];
            out[b][o] = acc;
        }
#pragma endscop
}

/*
 * Top-level: two-stage sequential FFN
 * hidden[] is passed explicitly so adoracc sees it as a memref arg,
 * producing affine.load/affine.store (not memref.alloca).
 */
void ffn(
    int input [BATCH][INPUT_DIM],
    int W1    [INPUT_DIM][HIDDEN_DIM],
    int W2    [HIDDEN_DIM][OUTPUT_DIM],
    int hidden[BATCH][HIDDEN_DIM],   /* caller-allocated intermediate */
    int out   [BATCH][OUTPUT_DIM])
{
    ffn_fc1(input, W1, hidden);   /* Stage 1: FC1 + ReLU → hidden */
    ffn_fc2(hidden, W2, out);     /* Stage 2: FC2         → out   */
}
