// Includes CUDA
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <stdio.h>

#include "Matrix.h"
#include "CudaProduct.h"

#define DEF_MB 50
#define DEF_NB 50

const int M_BLOCK_SIZE = 64 ;
const int N_BLOCK_SIZE = 64 ;
const int K_BLOCK_SIZE = 8 ;

const int A_TILE_SIZE = 8 ;
const int B_TILE_SIZE = 8 ;



__device__ void loadSubMatrices(Matrix A, Matrix B, int m, int k, int n, int pitchA, int pitchB, int kDispl, Matrix subA, Matrix subB) {
    int startLoadSubRowA = threadIdx.x / K_BLOCK_SIZE ;
    int startLoadRowA = M_BLOCK_SIZE * blockIdx.y ;
    int kSubA = threadIdx.x % K_BLOCK_SIZE ;
    int rowsPerBlock = min(M_BLOCK_SIZE, m - M_BLOCK_SIZE * blockIdx.y) ;

    int loadingIncr = blockDim.x / K_BLOCK_SIZE ;
    for (int loadRowIdx = startLoadSubRowA ; loadRowIdx < rowsPerBlock ; loadRowIdx += loadingIncr) {
        if (kDispl + kSubA < k) {
            subA[INDEX(loadRowIdx, kSubA, K_BLOCK_SIZE)] = A[INDEX(startLoadRowA + loadRowIdx, kDispl + kSubA, pitchA)] ;
        }
    }

    int startLoadSubColB = threadIdx.x % loadingIncr ;
    int startLoadColB = N_BLOCK_SIZE * blockIdx.x ;
    int kSubB = threadIdx.x / loadingIncr ;
    int colsPerBlock = min(N_BLOCK_SIZE, n - N_BLOCK_SIZE * blockIdx.x) ;

    for (int loadColIdx = startLoadSubColB ; loadColIdx < colsPerBlock ; loadColIdx += loadingIncr) {
        if (kDispl + kSubB < k) {
            subB[INDEX(kSubB, loadColIdx, N_BLOCK_SIZE)] = B[INDEX(kDispl + kSubB, startLoadColB + loadColIdx, pitchB)] ;
        }
    }

    

}

void moveMatricesFromHostToDevice(Matrix hostMatrix, Matrix *devMatrix, int rows, int cols, size_t *pitchPtr) ;

// Implem_4
__global__ void gpuProduct(Matrix A, Matrix B, Matrix C, int m, int k , int n, int pitchA, int pitchB, int pitchC) {
    __shared__ MatrixElemType subA[M_BLOCK_SIZE * K_BLOCK_SIZE] ;
    __shared__ MatrixElemType subB[K_BLOCK_SIZE * N_BLOCK_SIZE] ;

    MatrixElemType cAccMatrix[A_TILE_SIZE][B_TILE_SIZE] = {0.0} ;
    MatrixElemType subColA[A_TILE_SIZE] = {0.0} ;
    MatrixElemType subRowB[B_TILE_SIZE] = {0.0} ;

    int rowsPerBlock = min(M_BLOCK_SIZE, m - M_BLOCK_SIZE * blockIdx.y) ;
    int colsPerBlock = min(N_BLOCK_SIZE, n - N_BLOCK_SIZE * blockIdx.x) ;

    int numTilesB = ((colsPerBlock - 1) / B_TILE_SIZE) + 1 ;
    int numTilesA = ((rowsPerBlock - 1) / A_TILE_SIZE) + 1 ;

    int thrX = threadIdx.x % numTilesB ;
    int thrY = threadIdx.x / numTilesB ;

    int currTileSizeA = min(A_TILE_SIZE, rowsPerBlock - A_TILE_SIZE * thrY) ;
    int currTileSizeB = min(B_TILE_SIZE, colsPerBlock - B_TILE_SIZE * thrX) ;

    for (int kDispl = 0 ; kDispl < k ; kDispl += K_BLOCK_SIZE) {

        // Loading subA and subB
        loadSubMatrices(A, B, m, k, n, pitchA, pitchB, kDispl, subA, subB) ;
        __syncthreads() ;

        int currKLen = min(K_BLOCK_SIZE, k - kDispl) ;

        // Each thread computes a little rectangle of C
        if (thrY < numTilesA) {
            for (int dotIdx = 0 ; dotIdx < currKLen ; dotIdx++) {
                // Loading A Column and B Row in register cache
                for (int i = 0 ; i < currTileSizeA ; i++) {
                    subColA[i] = subA[INDEX(i + thrY * A_TILE_SIZE, dotIdx, K_BLOCK_SIZE)] ;
                }
                for (int j = 0 ; j < currTileSizeB ; j++) {
                    subRowB[j] = subB[INDEX(dotIdx, j + thrX * B_TILE_SIZE, N_BLOCK_SIZE)] ;
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
    int cRowStart = blockIdx.y * M_BLOCK_SIZE + thrY * A_TILE_SIZE ;
    int cColStart = blockIdx.x * N_BLOCK_SIZE + thrX * B_TILE_SIZE ;
    if (cRowStart < m && cColStart < n) {
        for (int tileIdxA = 0 ; tileIdxA < currTileSizeA ; tileIdxA++) {
            for (int tileIdxB = 0 ; tileIdxB < currTileSizeB ; tileIdxB++) {
                C[INDEX(tileIdxA + cRowStart, tileIdxB + cColStart, pitchC)] += cAccMatrix[tileIdxA][tileIdxB] ;
            }
        }
    }
}


// Implem_3
/*
__global__ void gpuProduct(Matrix A, Matrix B, Matrix C, int m, int k , int n, int pitchA, int pitchB, int pitchC) {
    __shared__ MatrixElemType subA[BLOCK_SIZE][K_BLOCK_LEN] ;
    __shared__ MatrixElemType subB[K_BLOCK_LEN][BLOCK_SIZE] ;

    int thrY = threadIdx.x / BLOCK_SIZE ;
    int thrX = threadIdx.x % BLOCK_SIZE ;

    MatrixElemType cAccArray[TILE_SIZE] = {0.0} ;

    int loadingSubRowA = threadIdx.x / K_BLOCK_LEN ;
    int loadingRowA = loadingSubRowA + BLOCK_SIZE * blockIdx.y ;
    int kSubA = threadIdx.x % K_BLOCK_LEN ;

    int loadingSubColB = threadIdx.x % BLOCK_SIZE ;
    int loadingColB = loadingSubColB + BLOCK_SIZE * blockIdx.x ;
    int kSubB = threadIdx.x / BLOCK_SIZE ;

    int rowsPerBlock = min(BLOCK_SIZE, m - BLOCK_SIZE * blockIdx.y) ;
    int colsPerBlock = min(BLOCK_SIZE, n - BLOCK_SIZE * blockIdx.x) ;

    for (int kDispl = 0 ; kDispl < k ; kDispl += K_BLOCK_LEN) {
        // Loading subA and subB
        int currKLen = min(K_BLOCK_LEN, k - kDispl) ;

        if (loadingRowA < m && kDispl + kSubA < k) {
            subA[loadingSubRowA][kSubA] = A[INDEX(loadingRowA, kDispl + kSubA, pitchA)] ;
        }
        if (loadingColB < n && kDispl + kSubB < k) {
            subB[kSubB][loadingSubColB] = B[INDEX(kDispl + kSubB, loadingColB, pitchB)] ;
        }
        __syncthreads() ;

        // Doing tile product
        if (thrX < colsPerBlock) {
            int currTileSize = min(TILE_SIZE, min(BLOCK_SIZE, rowsPerBlock) - TILE_SIZE * thrY) ;
            for (int dotIdx = 0 ; dotIdx < currKLen ; dotIdx++) {
                MatrixElemType bElem = subB[dotIdx][thrX] ;
                for (int tileIdx = 0 ; tileIdx < currTileSize ; tileIdx++) {
                    cAccArray[tileIdx] += subA[tileIdx + thrY * TILE_SIZE][dotIdx] * bElem ;
                }
            }
        }
        __syncthreads() ;
    }
    
    // Moving back to C
    int cRowStart = blockIdx.y * BLOCK_SIZE + thrY * TILE_SIZE ;
    if (cRowStart < m && loadingColB < n) {
        int currTileSize = min(TILE_SIZE, min(BLOCK_SIZE, rowsPerBlock) - TILE_SIZE * thrY) ;
        for (int tileIdx = 0 ; tileIdx < currTileSize ; tileIdx++) {
            C[INDEX(tileIdx + cRowStart, loadingColB, pitchC)] += cAccArray[tileIdx] ;
            
        }
    }
}
*/

// Implem_2
/*
// BOH NON FUNZIONA BENE...
__global__ void gpuProduct(Matrix A, Matrix B, Matrix C, int m, int k , int n, int pitchA, int pitchB, int pitchC) {

    __shared__ MatrixElemType subA[BLOCK_SIZE][K_BLOCK_LEN] ;
    __shared__ MatrixElemType subB[K_BLOCK_LEN][BLOCK_SIZE] ;

    int thrY = threadIdx.x / BLOCK_SIZE ;
    int thrX = threadIdx.x % BLOCK_SIZE ;

    int row = thrY + BLOCK_SIZE * blockIdx.y ;
    int col = thrX + BLOCK_SIZE * blockIdx.x ;

    MatrixElemType cAccArray[BLOCK_SIZE] = {0.0} ;

    int rowsPerBlock = min(BLOCK_SIZE, m - BLOCK_SIZE * blockIdx.y) ;
    int colsPerBlock = min(BLOCK_SIZE, n - BLOCK_SIZE * blockIdx.x) ;

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

        if (threadIdx.x < colsPerBlock) {
            for (int dotIdx = 0 ; dotIdx < currKLen ; dotIdx++) {
                MatrixElemType elemMatB = subB[dotIdx][threadIdx.x] ;
                for (int tileIdx = 0 ; tileIdx < rowsPerBlock ; tileIdx++) {
                    cAccArray[tileIdx] += subA[tileIdx][dotIdx] * elemMatB ;
                }
            }
        }
        __syncthreads() ;
    }

    int cRow = threadIdx.x + BLOCK_SIZE * blockIdx.y ;
    int cCol = threadIdx.x + BLOCK_SIZE * blockIdx.x ;
    if (cRow < m && cCol < n) {
        for (int tileIdx = 0 ; tileIdx < rowsPerBlock ; tileIdx++) {
            C[INDEX(tileIdx + BLOCK_SIZE * blockIdx.y, cCol, pitchC)] += cAccArray[tileIdx] ;
        } 
    }
    
}
*/

// Implem_1
/*
__global__ void gpuProduct(Matrix A, Matrix B, Matrix C, int m, int k , int n, int pitchA, int pitchB, int pitchC) {

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
*/


void CudaProduct(Matrix hostA, Matrix hostB, Matrix hostC, int m, int k, int n, int mb, int nb, Info *infoPtr) {

    if (mb <= 0) {
        mb = DEF_MB ;
    }
    if (nb <= 0) {
        nb = DEF_NB ;
    }

    Matrix devA, devB, devC ;
    size_t pitchA, pitchB, pitchC ;
    moveMatricesFromHostToDevice(hostA, &devA, m, k, &pitchA) ;
    moveMatricesFromHostToDevice(hostB, &devB, k, n, &pitchB) ;
    moveMatricesFromHostToDevice(hostC, &devC, m, n, &pitchC) ;

    // dim3 BLOCK_DIM(BLOCK_SIZE * BLOCK_SIZE) ;
    // dim3 GRID_DIM(((n - 1) / BLOCK_SIZE) + 1, ((m - 1) / BLOCK_SIZE) + 1) ;

    // Implem_3
    // dim3 BLOCK_DIM((BLOCK_SIZE * BLOCK_SIZE) / TILE_SIZE) ;
    // dim3 GRID_DIM(((n - 1) / BLOCK_SIZE) + 1, ((m - 1) / BLOCK_SIZE) + 1) ;

    // Implem_4
    dim3 BLOCK_DIM((M_BLOCK_SIZE * N_BLOCK_SIZE) / (A_TILE_SIZE * B_TILE_SIZE)) ;
    dim3 GRID_DIM(((n - 1) / N_BLOCK_SIZE) + 1, ((m - 1) / M_BLOCK_SIZE) + 1) ;

    StopWatchInterface* timer = 0;
    sdkCreateTimer(&timer);

    timer->start();
    gpuProduct<<<GRID_DIM, BLOCK_DIM>>>(devA, devB, devC, m, k, n, pitchA, pitchB, pitchC);
    checkCudaErrors(cudaDeviceSynchronize());
    timer->stop();
    
    infoPtr->productTime = timer->getTime() ;

    checkCudaErrors(
        cudaMemcpy2D(hostC, sizeof(MatrixElemType) * n, devC, pitchC * sizeof(MatrixElemType), sizeof(MatrixElemType) * n, m, cudaMemcpyDeviceToHost)
    ) ;
    // checkCudaErrors(
    //     cudaMemcpy(hostC, devC, sizeof(MatrixElemType) * m * n, cudaMemcpyDeviceToHost) 
    // ) ;

    cudaFree(devA) ;
    cudaFree(devB) ;
    cudaFree(devC) ;

    return ;
}

void moveMatricesFromHostToDevice(Matrix hostMatrix, Matrix *devMatrixPtr, int rows, int cols, size_t *pitchPtr) {
    checkCudaErrors(
        cudaHostRegister(hostMatrix, sizeof(MatrixElemType) * rows * cols, cudaHostRegisterDefault)
    ) ;

    // checkCudaErrors(
    //     cudaMalloc((void **) devMatrixPtr, sizeof(MatrixElemType) * rows * cols)
    // ) ;
    // checkCudaErrors(
    //     cudaMemcpy(*devMatrixPtr, hostMatrix, sizeof(MatrixElemType) * rows * cols, cudaMemcpyHostToDevice) 
    // ) ;
    //*pitchPtr = 0 ;

    checkCudaErrors(
        cudaMallocPitch((void **) devMatrixPtr, pitchPtr, sizeof(MatrixElemType) * cols, rows)
    ) ;
    checkCudaErrors(
        cudaMemcpy2D(*devMatrixPtr, *pitchPtr, hostMatrix, sizeof(MatrixElemType) * cols, sizeof(MatrixElemType) * cols, rows, cudaMemcpyHostToDevice)
    ) ;
    *pitchPtr = *pitchPtr / sizeof(MatrixElemType) ;
}