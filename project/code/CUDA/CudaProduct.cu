// Includes CUDA
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <stdio.h>

#include "Matrix.h"
#include "CudaProduct.h"

#define DEF_MB 50
#define DEF_NB 50

const int BLOCK_SIZE = 64 ;
const int TILE_SIZE = 8 ;
const int K_BLOCK_LEN = 8 ;

void moveMatricesFromHostToDevice(Matrix hostMatrix, Matrix *devMatrix, int rows, int cols, size_t *pitchPtr) ;

// Implem_3
__global__ void gpuProduct(Matrix A, Matrix B, Matrix C, int m, int k , int n, int pitchA, int pitchB, int pitchC) {
    __shared__ MatrixElemType subA[BLOCK_SIZE][K_BLOCK_LEN] ;
    __shared__ MatrixElemType subB[K_BLOCK_LEN][BLOCK_SIZE] ;

    int thrY = threadIdx.x / BLOCK_SIZE ;
    int thrX = threadIdx.x % BLOCK_SIZE ;

    int loadingRow = thrY + BLOCK_SIZE * blockIdx.y ;
    int loadingCol = thrX + BLOCK_SIZE * blockIdx.x ;

    MatrixElemType cAccArray[TILE_SIZE] = {0.0} ;

    int rowsPerBlock = min(BLOCK_SIZE, m - BLOCK_SIZE * blockIdx.y) ;
    int colsPerBlock = min(BLOCK_SIZE, n - BLOCK_SIZE * blockIdx.x) ;

    for (int kDispl = 0 ; kDispl < k ; kDispl += K_BLOCK_LEN) {
        // Loading subA and subB
        int currKLen = min(K_BLOCK_LEN, k - kDispl) ;

        int kSubA = threadIdx.x % BLOCK_SIZE ;
        int kSubB = threadIdx.x / BLOCK_SIZE ;

        if (loadingRow < m && kDispl + kSubA < k) {
            subA[thrY][kSubA] = A[INDEX(loadingRow, kDispl + kSubA, pitchA)] ;
        }
        if (loadingCol < n && kDispl + kSubB < k) {
            subB[kSubB][thrX] = B[INDEX(kDispl + kSubB, loadingCol, pitchB)] ;
        }
        __syncthreads() ;

        // TODO Handle different tile size

        // Doing tile product
        if (loadingCol < n) {
            int currTileSize = min(TILE_SIZE, BLOCK_SIZE - TILE_SIZE * thrY) ;
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
    int cRowStart = blockIdx.x * BLOCK_SIZE + thrY * TILE_SIZE ;
    if (cRowStart < m && loadingCol < n) {
        int currTileSize = min(TILE_SIZE, BLOCK_SIZE - TILE_SIZE * thrY) ;
        for (int tileIdx = 0 ; tileIdx < currTileSize ; tileIdx++) {
            C[INDEX(tileIdx + cRowStart, loadingCol, pitchC)] += cAccArray[tileIdx] ;
        }
    }
}

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

    dim3 BLOCK_DIM((BLOCK_SIZE * BLOCK_SIZE) / TILE_SIZE) ;
    dim3 GRID_DIM(((n - 1) / BLOCK_SIZE) + 1, ((m - 1) / BLOCK_SIZE) + 1) ;

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