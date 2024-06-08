// Includes CUDA
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <stdio.h>

#include "Matrix.h"
#include "CudaProduct.h"

#define DEF_MB 50
#define DEF_NB 50

const int BLOCK_SIZE = 32 ;
const int K_BLOCK_LEN = BLOCK_SIZE ;

void moveMatricesFromHostToDevice(Matrix hostMatrix, Matrix *devMatrix, int rows, int cols, size_t *pitchPtr) ;


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
        if (col == 0) {
            printf("LOADING %d %d %f\n", threadIdx.x, kIdxB, B[INDEX(kDispl + kIdxB, col, pitchB)]) ;
        }
        if (kIdxB < currKLen && col < n) {
            subB[kIdxB][thrX] = B[INDEX(kDispl + kIdxB, col, pitchB)] ;
        }
        __syncthreads() ;

        if (row < m && col < n) {
            for (int kIdx = 0 ; kIdx < currKLen ; kIdx++) {
                if (row == 0 && col == 0) {
                    printf("PROD %f %f\n", subA[thrY][kIdx], subB[kIdx][thrX]) ;
                }
                cAcc += subA[thrY][kIdx] * subB[kIdx][thrX] ;
            }
        }
        __syncthreads() ;
    }
    if (row < m && col < n) {
        C[INDEX(row, col, pitchC)] += cAcc ;
    }

    // int thrY = threadIdx.x / BLOCK_SIZE ;
    // int thrX = threadIdx.x % BLOCK_SIZE ;

    // int row = thrY + BLOCK_SIZE * blockIdx.y ;
    // int col = thrX + BLOCK_SIZE * blockIdx.x ;

    // MatrixElemType cAcc = 0.0 ;

    // if (row < m && col < n) {
    //     for (int kIdx = 0 ; kIdx < k ; kIdx++) {
    //         cAcc += A[INDEX(row, kIdx, pitchA)] * B[INDEX(kIdx, col, pitchB)] ;
    //     }
    //     C[INDEX(row, col, pitchC)] += cAcc ;
    // }
    
}


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

    dim3 BLOCK_DIM(BLOCK_SIZE * BLOCK_SIZE) ;
    dim3 GRID_DIM(((n - 1) / BLOCK_SIZE) + 1, ((m - 1) / BLOCK_SIZE) + 1) ;

    StopWatchInterface* timer = 0;
    sdkCreateTimer(&timer);

    timer->start();
    gpuProduct<<<GRID_DIM, BLOCK_DIM>>>(devA, devB, devC, m, k, n, pitchA, pitchB, pitchC);
    checkCudaErrors(cudaDeviceSynchronize());
    timer->stop();
    
    infoPtr->productTime = timer->getTime() ;

    checkCudaErrors(
        cudaMemcpy2D(hostC, sizeof(MatrixElemType) * m, devC, pitchC * sizeof(MatrixElemType), sizeof(MatrixElemType) * m, n, cudaMemcpyDeviceToHost)
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
    checkCudaErrors(
        cudaMallocPitch((void **) devMatrixPtr, pitchPtr, sizeof(MatrixElemType) * cols, rows)
    ) ;
    // checkCudaErrors(
    //     cudaMalloc((void **) devMatrixPtr, sizeof(MatrixElemType) * rows * cols)
    // ) ;

    checkCudaErrors(
        cudaMemcpy2D(*devMatrixPtr, *pitchPtr, hostMatrix, sizeof(MatrixElemType) * cols, sizeof(MatrixElemType) * cols, rows, cudaMemcpyHostToDevice)
    ) ;
    // checkCudaErrors(
    //     cudaMemcpy(*devMatrixPtr, hostMatrix, sizeof(MatrixElemType) * rows * cols, cudaMemcpyHostToDevice) 
    // ) ;
    *pitchPtr = *pitchPtr / sizeof(MatrixElemType) ;
    //*pitchPtr = 0 ;
}