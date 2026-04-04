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
#include <math.h>

/* Include polybench common header. */
#include "rocket_polybench.h"

/* Include benchmark-specific header. */
#define MINI_DATASET
// #define SMALL_DATASET
// #define MEDIUM_DATASET
// #define LARGE_DATASET
#ifndef ROCKET_TARGET
#define ROCKET_TARGET
#define __riscv
#define __GNUC__
#endif
# define POLYBENCH_USE_SCALAR_LB
# define POLYBENCH_USE_C99_PROTO
# define DATA_TYPE_IS_FLOAT

#include "adora_test.h"
#include "include/encoding.h"
#include "gemm.h"


/* Array initialization. */
static
void init_array(int ni, int nj, int nk,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		// DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
		// DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		// DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj))
  	DATA_TYPE C[NI][NJ],
		DATA_TYPE A[NI][NK],
		DATA_TYPE B[NK][NJ])
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
      C[i][j] = (DATA_TYPE) ((i*j+1) % ni) / ni;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i][j] = (DATA_TYPE) (i*(j+1) % nk) / nk;
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (DATA_TYPE) (i*(j+2) % nj) / nj;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
#define FpToHex(x) *((unsigned int*)&x)
static
void print_array(int ni, int nj,
		 DATA_TYPE C1[NI][NJ], DATA_TYPE C2[NI][NJ])
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
	// if ((i * ni + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	// fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, C[i][j]);
      if((int)(10000 * C1[i][j]) != (int)(10000 * C2[i][j])){
        printf("[%d, %d]%ld-%ld," , i, j, (int)(10000 * C1[i][j]), (int)(10000 * C2[i][j]));
        printf("%x-%x\t", FpToHex(C1[i][j]), FpToHex(C2[i][j]));
      }
    }
  POLYBENCH_DUMP_END("C");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_gemm_cpu(int ni, int nj, int nk,
		 DATA_TYPE alpha,
		 DATA_TYPE beta,
		 DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj),
		 DATA_TYPE POLYBENCH_2D(A,NI,NK,ni,nk),
		 DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj))
{
  int i, j, k;

//BLAS PARAMS
//TRANSA = 'N'
//TRANSB = 'N'
// => Form C := alpha*A*B + beta*C,
//A is NIxNK
//B is NKxNJ
//C is NIxNJ
#pragma scop
  for (i = 0; i < _PB_NI; i++) {
    for (j = 0; j < _PB_NJ; j++)
	C[i][j] *= beta;
    for (k = 0; k < _PB_NK; k++) {
       for (j = 0; j < _PB_NJ; j++)
	  C[i][j] += alpha * A[i][k] * B[k][j];
    }
  }
#pragma endscop

}

void gemm(
		//  DATA_TYPE alpha,
		//  DATA_TYPE beta,
		 DATA_TYPE C[NI][NJ], /// 1000 1100
		 DATA_TYPE A[NI][NK], /// 1000 1200
		 DATA_TYPE B[NK][NJ]); /// 1200 1100

void gemm_opt(
		//  DATA_TYPE alpha,
		//  DATA_TYPE beta,
		 DATA_TYPE C[NI][NJ], /// 1000 1100
		 DATA_TYPE A[NI][NK], /// 1000 1200
		 DATA_TYPE B[NK][NJ]); /// 1200 1100

int main(int argc, char** argv)
{
  printf("CGRA start gemm mini!\n");
  printf("NI: %d!\n", NI);
  printf("NJ: %d!\n", NJ);
  printf("NK: %d!\n", NK);
  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;
  int nk = NK;

  long long unsigned start;
  long long unsigned end;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  // POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,NI,NJ,ni,nj);
  // POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NK,ni,nk);
  // POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NK,NJ,nk,nj);
  DATA_TYPE C1[NI][NJ], C2[NI][NJ],  C3[NI][NJ];
  DATA_TYPE A[NI][NK];
  DATA_TYPE B[NK][NJ];

  /* Initialize array(s). */


  /* Start timer. */
  // polybench_start_instruments;

  init_array (ni, nj, nk, &alpha, &beta,
	      C1, A, B);
  printf("\nstart cpu kernel\n");
  start = rdcycle();
  kernel_gemm_cpu (ni, nj, nk,
	       alpha, beta,
	       C1,
	       A,
	       B);
  end = rdcycle();
  printf("It takes %llu cycles for CPU to finish the task.\n", end - start);

  /* Run kernel. */
  init_array (ni, nj, nk, &alpha, &beta,
	      C2, A, B);
  printf("\nstart gemm\n");
  start = rdcycle();
  gemm (C2,A,B);
  end = rdcycle();
  printf("It takes %llu cycles for CGRA to finish the gemm.\n", end - start);


  /* Stop and print timer. */
  // polybench_stop_instruments;
  // polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
     print_array(NI, NJ, C1, C2);
  // polybench_prevent_dce(print_array(ni, nj,  POLYBENCH_ARRAY(C)));


  /* Run kernel. */
  init_array (ni, nj, nk, &alpha, &beta,
	      C3, A, B);
  printf("\nstart gemm opt\n");
  start = rdcycle();
  gemm_opt (C3,A,B);
  end = rdcycle();
  printf("It takes %llu cycles for CGRA to finish the gemm_opt.\n", end - start);

     print_array(NI, NJ, C1, C3);
  /* Be clean. */
  // POLYBENCH_FREE_ARRAY(C);
  // POLYBENCH_FREE_ARRAY(A);
  // POLYBENCH_FREE_ARRAY(B);

  return 0;
}
