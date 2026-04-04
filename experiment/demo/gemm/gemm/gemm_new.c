/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* gemm.c: this file is part of PolyBench/C */

// #include <stdio.h>
// #include <unistd.h>
// #include <string.h>
// #include <math.h>

/* Include polybench common header. */
// #include "rocket_polybench.h"

/* Include benchmark-specific header. */
#include "gemm.h"
#include "adora_test.h"

/* Include benchmark-specific header. */
// #define MINI_DATASET
// #define SMALL_DATASET
// #define MEDIUM_DATASET
#define LARGE_DATASET
#include "gemm.h"



void gemm(
		//  DATA_TYPE alpha,
		//  DATA_TYPE beta,
		 DATA_TYPE C[NI][NJ], /// 1000 1100
		 DATA_TYPE A[NI][NK], /// 1000 1200
		 DATA_TYPE B[NK][NJ]) /// 1200 1100
{
  int i, j, k;
  DATA_TYPE alpha = 1.5;
  DATA_TYPE beta = 1.2;
//BLAS PARAMS
//TRANSA = 'N'
//TRANSB = 'N'
// => Form C := alpha*A*B + beta*C,
//A is NIxNK
//B is NKxNJ
//C is NIxNJ
#pragma scop
  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j++){
	    C[i][j] *= beta;
    }
    for (k = 0; k < NK; k++) {
      for (j = 0; j < NJ; j++){
	      C[i][j] += alpha * A[i][k] * B[k][j];
      }
    }
  }
#pragma endscop

}