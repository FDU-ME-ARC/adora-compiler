/*
 * fft.c — Cooley-Tukey Radix-2 FFT (integer butterfly, N=16)
 *
 * 任务调度特征：log2(N)=4 层 butterfly，每层内部全并行，层间严格 RAW 依赖。
 *
 *   Stage 0 ──(RAW)──► Stage 1 ──(RAW)──► Stage 2 ──(RAW)──► Stage 3
 *
 * 每层展开为独立函数（常量 bound），#pragma scop 使 cgeist 提升为 affine。
 * 蝶形因子预计算为常量数组，避免非仿射除法。
 */

#define FFT_N 16

/* Q8 twiddle factors: W[k] = cos(2π·k/N) + j·sin(2π·k/N), scaled by 256 */
static const int W_re[8] = { 256, 237, 181,  98,   0, -98,-181,-237 };
static const int W_im[8] = {   0, -98,-181,-237,-256,-237,-181, -98 };

/* -----------------------------------------------------------------------
 * Stage 0: half=8, step=16, 1 group, 8 butterflies per group
 * tw_stride = 1  (tw_k = k * 1)
 * ----------------------------------------------------------------------- */
void fft_stage0(int re[FFT_N], int im[FFT_N])
{
#pragma scop
    for (int k = 0; k < 8; k++) {
        int a = k, b = k + 8;
        int wr = W_re[k], wi = W_im[k];
        int tr = (wr * re[b] - wi * im[b]) >> 8;
        int ti = (wr * im[b] + wi * re[b]) >> 8;
        re[b] = re[a] - tr;  im[b] = im[a] - ti;
        re[a] = re[a] + tr;  im[a] = im[a] + ti;
    }
#pragma endscop
}

/* -----------------------------------------------------------------------
 * Stage 1: half=4, step=8, 2 groups, 4 butterflies per group
 * tw_stride = 2  (tw_k = k * 2)
 * ----------------------------------------------------------------------- */
void fft_stage1(int re[FFT_N], int im[FFT_N])
{
#pragma scop
    for (int g = 0; g < 2; g++) {
        for (int k = 0; k < 4; k++) {
            int a = g * 8 + k, b = a + 4;
            int wr = W_re[k * 2], wi = W_im[k * 2];
            int tr = (wr * re[b] - wi * im[b]) >> 8;
            int ti = (wr * im[b] + wi * re[b]) >> 8;
            re[b] = re[a] - tr;  im[b] = im[a] - ti;
            re[a] = re[a] + tr;  im[a] = im[a] + ti;
        }
    }
#pragma endscop
}

/* -----------------------------------------------------------------------
 * Stage 2: half=2, step=4, 4 groups, 2 butterflies per group
 * tw_stride = 4  (tw_k = k * 4)
 * ----------------------------------------------------------------------- */
void fft_stage2(int re[FFT_N], int im[FFT_N])
{
#pragma scop
    for (int g = 0; g < 4; g++) {
        for (int k = 0; k < 2; k++) {
            int a = g * 4 + k, b = a + 2;
            int wr = W_re[k * 4], wi = W_im[k * 4];
            int tr = (wr * re[b] - wi * im[b]) >> 8;
            int ti = (wr * im[b] + wi * re[b]) >> 8;
            re[b] = re[a] - tr;  im[b] = im[a] - ti;
            re[a] = re[a] + tr;  im[a] = im[a] + ti;
        }
    }
#pragma endscop
}

/* -----------------------------------------------------------------------
 * Stage 3: half=1, step=2, 8 groups, 1 butterfly per group
 * tw_stride = 8  (tw_k = 0 always → W_re[0]=256, W_im[0]=0 → no rotation)
 * ----------------------------------------------------------------------- */
void fft_stage3(int re[FFT_N], int im[FFT_N])
{
#pragma scop
    for (int g = 0; g < 8; g++) {
        int a = g * 2, b = a + 1;
        int tr = re[b], ti = im[b];
        re[b] = re[a] - tr;  im[b] = im[a] - ti;
        re[a] = re[a] + tr;  im[a] = im[a] + ti;
    }
#pragma endscop
}

void fft(int re[FFT_N], int im[FFT_N])
{
    fft_stage0(re, im);
    fft_stage1(re, im);
    fft_stage2(re, im);
    fft_stage3(re, im);
}
