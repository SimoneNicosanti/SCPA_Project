#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <stdio.h>

#include "Matrix.h"

/*
    In this case each thread:
    - Loads multiple elements from GMEM to SMEM
    - Computes a sub square of elements in Matrix C
    - The subA is loaded transposed: this allow us to load from subA contiguous elements
*/

template <const int MB, const int KB, const int NB>
__device__ void loadSubMatrices_5(
    Matrix A, Matrix B, 
    int m, int k, int n, 
    int pitchA, int pitchB, 
    int kDispl, 
    Matrix subA, Matrix subB
) {
    int startLoadSubRowA = threadIdx.x / KB ;
    int startLoadRowA = MB * blockIdx.y ;
    int kSubA = threadIdx.x % KB ;
    int rowsPerBlock = min(MB, m - MB * blockIdx.y) ;

    int loadingIncr = blockDim.x / KB ;
    for (int loadRowIdx = startLoadSubRowA ; loadRowIdx < rowsPerBlock ; loadRowIdx += loadingIncr) {
        if (kDispl + kSubA < k) {
            subA[INDEX(kSubA, loadRowIdx, MB)] = A[INDEX(startLoadRowA + loadRowIdx, kDispl + kSubA, pitchA)] ;
        }
    }

    int startLoadSubColB = threadIdx.x % loadingIncr ;
    int startLoadColB = NB * blockIdx.x ;
    int kSubB = threadIdx.x / loadingIncr ;
    int colsPerBlock = min(NB, n - NB * blockIdx.x) ;

    for (int loadColIdx = startLoadSubColB ; loadColIdx < colsPerBlock ; loadColIdx += loadingIncr) {
        if (kDispl + kSubB < k) {
            subB[INDEX(kSubB, loadColIdx, NB)] = B[INDEX(kDispl + kSubB, startLoadColB + loadColIdx, pitchB)] ;
        }
    }
}

template <const int MB, const int KB, const int NB, const int TILE_A, const int TILE_B>
__global__ void gpuProduct_5(
    Matrix A, Matrix B, Matrix C, 
    int m, int k , int n, 
    int pitchA, int pitchB, int pitchC
) {
    __shared__ MatrixElemType subA[KB * MB] ;
    __shared__ MatrixElemType subB[KB * NB] ;

    MatrixElemType cAccMatrix[TILE_A][TILE_B] = {0.0} ;
    MatrixElemType subColA[TILE_A] = {0.0} ;
    MatrixElemType subRowB[TILE_B] = {0.0} ;

    int rowsPerBlock = min(MB, m - MB * blockIdx.y) ;
    int colsPerBlock = min(NB, n - NB * blockIdx.x) ;

    int numTilesB = ((colsPerBlock - 1) / TILE_B) + 1 ;
    int numTilesA = ((rowsPerBlock - 1) / TILE_A) + 1 ;

    int thrX = threadIdx.x % numTilesB ;
    int thrY = threadIdx.x / numTilesB ;

    int currTileSizeA = min(TILE_A, rowsPerBlock - TILE_A * thrY) ;
    int currTileSizeB = min(TILE_B, colsPerBlock - TILE_B * thrX) ;

    for (int kDispl = 0 ; kDispl < k ; kDispl += KB) {

        // Loading subA and subB
        loadSubMatrices_5
            <MB, KB, NB>
            (A, B, m, k, n, pitchA, pitchB, kDispl, subA, subB) ;
        __syncthreads() ;

        int currKLen = min(KB, k - kDispl) ;

        // Each thread computes a little rectangle of C
        if (thrY < numTilesA) {
            for (int dotIdx = 0 ; dotIdx < currKLen ; dotIdx++) {
                // Loading A Column and B Row in register cache
                for (int j = 0 ; j < currTileSizeA ; j++) {
                    subColA[j] = subA[INDEX(dotIdx, j + thrY * TILE_A, MB)] ;
                }
                for (int j = 0 ; j < currTileSizeB ; j++) {
                    subRowB[j] = subB[INDEX(dotIdx, j + thrX * TILE_B, NB)] ;
                }

                for (int rowIdx = 0 ; rowIdx < currTileSizeA ; rowIdx++) {
                    for (int colIdx = 0 ; colIdx < currTileSizeB ; colIdx++) {
                        cAccMatrix[rowIdx][colIdx] += subColA[rowIdx] * subRowB[colIdx] ;
                    }
                }
            }
        }
        __syncthreads() ;
    }
    
    // Moving back to C
    int cRowStart = blockIdx.y * MB + thrY * TILE_A ;
    int cColStart = blockIdx.x * NB + thrX * TILE_B ;
    if (cRowStart < m && cColStart < n) {
        for (int tileIdxA = 0 ; tileIdxA < currTileSizeA ; tileIdxA++) {
            for (int tileIdxB = 0 ; tileIdxB < currTileSizeB ; tileIdxB++) {
                C[INDEX(tileIdxA + cRowStart, tileIdxB + cColStart, pitchC)] += cAccMatrix[tileIdxA][tileIdxB] ;
            }
        }
    }
}