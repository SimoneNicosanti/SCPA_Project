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

template <const int MB, const int KB, const int NB, const int TILE_A, const int TILE_B>
__global__ void gpuProduct_4(
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

    int thrTileBIdx = threadIdx.x ;
    int thrTileAIdx = threadIdx.y ;

    int currTileSizeA = min(TILE_A, rowsPerBlock - TILE_A * thrTileAIdx) ;
    int currTileSizeB = min(TILE_B, colsPerBlock - TILE_B * thrTileBIdx) ;

    for (int kDispl = 0 ; kDispl < k ; kDispl += KB) {

        // Loading subA and subB
        loadSubMatrices
            <MB, KB, NB>
            (A, B, m, k, n, pitchA, pitchB, kDispl, subA, subB, MB, NB, true) ;
        __syncthreads() ;


        int currKLen = min(KB, k - kDispl) ;

        // Each thread computes a little rectangle of C
        if (thrTileAIdx < numTilesA && thrTileBIdx < numTilesB) {
            for (int dotIdx = 0 ; dotIdx < currKLen ; dotIdx++) {
                // Loading A Column and B Row in register cache
                for (int i = 0 ; i < currTileSizeA ; i++) {
                    subColA[i] = subA[INDEX(dotIdx, i + thrTileAIdx * TILE_A, MB)] ;
                }
                for (int j = 0 ; j < currTileSizeB ; j++) {
                    subRowB[j] = subB[INDEX(dotIdx, j + thrTileBIdx * TILE_B, NB)] ;
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
    int cRowStart = blockIdx.y * MB + thrTileAIdx * TILE_A ;
    int cColStart = blockIdx.x * NB + thrTileBIdx * TILE_B ;
    if (cRowStart < m && cColStart < n) {
        for (int tileIdxA = 0 ; tileIdxA < currTileSizeA ; tileIdxA++) {
            for (int tileIdxB = 0 ; tileIdxB < currTileSizeB ; tileIdxB++) {
                C[INDEX(tileIdxA + cRowStart, tileIdxB + cColStart, pitchC)] += cAccMatrix[tileIdxA][tileIdxB] ;
            }
        }
    }
}