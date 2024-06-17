#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <stdio.h>

#include "Matrix.h"

// Implem_1

/*
    Implementazione con:
    * Operazione eseguite direttamente in memoria globale
    * Senza caching in local memory
    * Usando Coalesce della global memory
    * Usando registro accumulatore
*/

template <const int BLOCK_SIZE>
__global__ void gpuProduct_1(Matrix A, Matrix B, Matrix C, int m, int k , int n, int pitchA, int pitchB, int pitchC) {

    int rowIdx = (threadIdx.x / BLOCK_SIZE) + BLOCK_SIZE * blockIdx.y ;
    int colIdx = (threadIdx.x % BLOCK_SIZE) + BLOCK_SIZE * blockIdx.x ;

    MatrixElemType cAcc = 0.0 ;
    if (rowIdx < m && colIdx < n) {
        for (int kIdx = 0 ; kIdx < k ; kIdx++) {
            cAcc += A[INDEX(rowIdx, kIdx, pitchA)] * B[INDEX(kIdx, colIdx, pitchB)] ;
        }

        C[INDEX(rowIdx, colIdx, pitchC)] += cAcc ;
    }
    
}
