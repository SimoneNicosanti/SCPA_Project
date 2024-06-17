#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <stdio.h>

#include "Matrix.h"

// Implem_0
/*
    Implementazione Naive del Kernel.
    - Operazioni eseguite direttamente nella global memory 
    - Senza caching nella memoria locale
*/


__global__ void gpuProduct_0(Matrix A, Matrix B, Matrix C, int m, int k , int n, int pitchA, int pitchB, int pitchC) {

    int rowIdx = threadIdx.x + blockIdx.y * blockDim.y ;
    int colIdx = threadIdx.x + blockIdx.x * blockDim.x ;

    if (colIdx < n && rowIdx < m) {
        for (int kIdx = 0 ; kIdx < k ; kIdx++) {
            C[INDEX(rowIdx, colIdx, pitchC)] += A[INDEX(rowIdx, kIdx, pitchA)] * B[INDEX(kIdx, colIdx, pitchB)] ;
        }
    }
}
