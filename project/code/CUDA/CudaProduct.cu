// Includes CUDA
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include "Matrix.h"
#include "CudaProduct.h"

#define DEF_MB 50
#define DEF_NB 50

const dim3 BLOCK_DIM(16, 8) ;

void moveMatricesFromHostToDevice(Matrix hostMatrix, Matrix *devMatrix, int rows, int cols, size_t *pitchPtr) ;


__global__ void gpuProduct(Matrix A, Matrix B, Matrix C, int m, int k , int n, int pitchA, int pitchB, int pitchC) {

    __shared__ MatrixElemType subA[BLOCK_DIM.y][BLOCK_DIM.x] ;
    __shared__ MatrixElemType subB[BLOCK_DIM.x][BLOCK_DIM.y] ;

    float subCElem = 0.0 ;
    
    int rowA = threadIdx.y + blockIdx.y * blockDim.y ;
    int colB = threadIdx.x + blockIdx.x * blockDim.x ;

    if (rowA < m && colB < n) {
        for (int kIndex = threadIdx.x ; kIndex < k ; kIndex += blockDim.x) {
            subA[threadIdx.y][threadIdx.x] = A[INDEX(rowA, kIndex, pitchA)] ;
            subB[threadIdx.x][threadIdx.y] = B[INDEX(kIndex, colB, pitchB)] ;

            __syncthreads() ;

            for (int i = 0 ; i < min(k - kIndex, blockDim.x) ; i++) {
                subCElem += subA[threadIdx.y][i] * subB[i][threadIdx.y] ;
            }
            
            __syncthreads() ;
        }
        C[INDEX(rowA, colB, pitchC)] = subCElem ;
    }
    
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

    dim3 GRID_DIM(((n - 1) / BLOCK_DIM.x) + 1, ((m - 1) / BLOCK_DIM.y) + 1) ;

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