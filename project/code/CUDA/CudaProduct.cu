// Includes CUDA
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

extern "C" {
    #include "Matrix.h"
}

#include "CudaProduct.h"

#define DEF_MB 50
#define DEF_NB 50

const dim3 BLOCK_DIM(32, 32) ;

void moveMatricesFromHostToDevice(Matrix hostMatrix, Matrix *devMatrix, int rows, int cols, size_t *pitchPtr) ;


__global__ void gpuProduct(Matrix A, Matrix B, Matrix C, int m, int k , int n, int pitchA, int pitchB, int pitchC) {

    int row = threadIdx.y + blockIdx.y * blockDim.y ;
    __shared__ MatrixElemType subCalc[BLOCK_DIM.y][BLOCK_DIM.x] ;

    subCalc[threadIdx.y][threadIdx.x] = 0.0 ;

    for (int col = 0 ; col < n ; col++) {
        if (row < m) {
            float subProd = 0.0 ;
            for (int idx = threadIdx.x ; idx < k ; idx += blockDim.x) {
                subProd += A[INDEX(row, idx, m + pitchA)] * B[INDEX(idx, col, n + pitchB)] ;
            }
            subCalc[threadIdx.y][threadIdx.x] = subProd ;
        }

        __syncthreads() ;

        for (unsigned int s = (blockDim.x >> 1) ; s > 0 ; s >>= 1) {
            if (threadIdx.x < s) {
                subCalc[threadIdx.y][threadIdx.x] += subCalc[threadIdx.y][threadIdx.x + s] ;
            }
            __syncthreads() ;
        }

        if (threadIdx.x == 0 && row < m) {
            C[INDEX(row, col, n + pitchC)] += subCalc[threadIdx.y][threadIdx.x] ;
        }

        __syncthreads() ;
    }
    
}



// TODO CHECK FOR CUDA ERRORS
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

    dim3 gridDim(1, ((m - 1) / BLOCK_DIM.y) + 1) ;

    StopWatchInterface* timer = 0;
    sdkCreateTimer(&timer);

    timer->start();
    gpuProduct<<<gridDim, BLOCK_DIM>>>(devA, devB, devC, m, k, n, pitchA, pitchB, pitchC);
    checkCudaErrors(cudaDeviceSynchronize());
    timer->stop();
    
    infoPtr->productTime = timer->getTime() ;

    // checkCudaErrors(
    //     cudaMemcpy2D(hostC, sizeof(MatrixElemType) * m, devC, pitchC, sizeof(MatrixElemType) * m, n, cudaMemcpyDeviceToHost)
    // ) ;
    checkCudaErrors(
        cudaMemcpy(hostC, devC, sizeof(MatrixElemType) * m * n, cudaMemcpyDeviceToHost) 
    ) ;

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
    //     cudaMallocPitch((void **) devMatrixPtr, pitchPtr, sizeof(MatrixElemType) * cols, rows)
    // ) ;
    checkCudaErrors(
        cudaMalloc((void **) devMatrixPtr, sizeof(MatrixElemType) * rows * cols)
    ) ;

    // checkCudaErrors(
    //     cudaMemcpy2D(*devMatrixPtr, *pitchPtr, hostMatrix, sizeof(MatrixElemType) * cols, sizeof(MatrixElemType) * cols, rows, cudaMemcpyHostToDevice)
    // ) ;
    checkCudaErrors(
        cudaMemcpy(*devMatrixPtr, hostMatrix, sizeof(MatrixElemType) * rows * cols, cudaMemcpyHostToDevice) 
    ) ;
    // *pitchPtr = *pitchPtr / sizeof(MatrixElemType) ;
    *pitchPtr = 0 ;
}