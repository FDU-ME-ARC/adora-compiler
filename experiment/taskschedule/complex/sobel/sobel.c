/*
 * sobel.c — Sobel Edge Detection (64×64 grayscale, i32)
 *           Flattened single-function version for cgeist compatibility.
 *
 * 任务调度特征：菱形依赖（AND-join）
 *
 *   Stage 1a: Gx[i][j] = Kx ⊛ img   (水平梯度，可与 Gy 并行)
 *   Stage 1b: Gy[i][j] = Ky ⊛ img   (垂直梯度，可与 Gx 并行)
 *   Stage 2:  G[i][j]  = |Gx| + |Gy| (AND-join: 等 Gx AND Gy 都完成)
 *
 * 调度依赖图（菱形）：
 *        img
 *       ↙   ↘
 *     Gx     Gy    ← 并行
 *       ↘   ↙
 *        G         ← AND-join
 */

#define H 64
#define W 64

/*
 * sobel(): 单函数实现，三个逻辑 stage 顺序执行。
 * 调度器可识别 Gx/Gy 的依赖图并决策是否并行化 Stage1a 和 Stage1b。
 */
void sobel(int img[H][W], int G[H][W])
{
    int Gx[H][W];
    int Gy[H][W];

#pragma scop
    /* 初始化 */
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++) {
            Gx[i][j] = 0;
            Gy[i][j] = 0;
        }

    /* Stage 1a: Gx — horizontal Sobel  Kx=[[-1,0,1],[-2,0,2],[-1,0,1]] */
    for (int i = 1; i < H-1; i++)
        for (int j = 1; j < W-1; j++)
            Gx[i][j] =
                -1*img[i-1][j-1] + 1*img[i-1][j+1]
                -2*img[i  ][j-1] + 2*img[i  ][j+1]
                -1*img[i+1][j-1] + 1*img[i+1][j+1];

    /* Stage 1b: Gy — vertical Sobel  Ky=[[-1,-2,-1],[0,0,0],[1,2,1]] */
    for (int i = 1; i < H-1; i++)
        for (int j = 1; j < W-1; j++)
            Gy[i][j] =
                -1*img[i-1][j-1] - 2*img[i-1][j] - 1*img[i-1][j+1]
                +1*img[i+1][j-1] + 2*img[i+1][j] + 1*img[i+1][j+1];

    /* Stage 2: magnitude = |Gx| + |Gy|  (AND-join) */
    for (int i = 1; i < H-1; i++)
        for (int j = 1; j < W-1; j++) {
            int gx = Gx[i][j]; if (gx < 0) gx = -gx;
            int gy = Gy[i][j]; if (gy < 0) gy = -gy;
            G[i][j] = gx + gy;
        }
#pragma endscop
}
