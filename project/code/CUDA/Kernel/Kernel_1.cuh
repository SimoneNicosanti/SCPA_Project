#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <stdio.h>

#include "Matrix.h"

// Implem_1

template <const int MB, const int KB, const int NB>
__global__ void gpuProduct_1(Matrix A, Matrix B, Matrix C, int m, int k , int n, int pitchA, int pitchB, int pitchC) {

    __shared__ MatrixElemType subA[BLOCK_SIZE][K_BLOCK_LEN] ;
    __shared__ MatrixElemType subB[K_BLOCK_LEN][BLOCK_SIZE] ;

    int thrY = threadIdx.x / BLOCK_SIZE ;
    int thrX = threadIdx.x % BLOCK_SIZE ;

    int row = thrY + BLOCK_SIZE * blockIdx.y ;
    int col = thrX + BLOCK_SIZE * blockIdx.x ;

    MatrixElemType cAcc = 0.0 ;

    for (int kDispl = 0 ; kDispl < k ; kDispl += K_BLOCK_LEN) {
        int currKLen = min(K_BLOCK_LEN, k - kDispl) ;

        int kIdxA = threadIdx.x % K_BLOCK_LEN ;
        int kIdxB = threadIdx.x / K_BLOCK_LEN ;
        
        if (kIdxA < currKLen && row < m) {
            subA[thrY][kIdxA] = A[INDEX(row, kDispl + kIdxA, pitchA)] ;
        }
        if (kIdxB < currKLen && col < n) {
            subB[kIdxB][thrX] = B[INDEX(kDispl + kIdxB, col, pitchB)] ;
        }
        __syncthreads() ;

        if (row < m && col < n) {
            for (int kIdx = 0 ; kIdx < currKLen ; kIdx++) {
                cAcc += subA[thrY][kIdx] * subB[kIdx][thrX] ;
            }
        }
        __syncthreads() ;
    }
    if (row < m && col < n) {
        C[INDEX(row, col, pitchC)] += cAcc ;
    }
}
