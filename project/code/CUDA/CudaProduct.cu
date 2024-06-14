// Includes CUDA
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <stdio.h>

#include "Matrix.h"
#include "CudaProduct.h"
#include "Kernel_4.cuh"

#define DEF_MB 50
#define DEF_NB 50

void moveMatricesFromHostToDevice(Matrix hostMatrix, Matrix *devMatrix, int rows, int cols, size_t *pitchPtr) ;
float callKernel(Matrix A, Matrix B, Matrix C, int m, int k, int n, int pitchA, int pitchB, int pitchC) ;


float callKernel(Matrix A, Matrix B, Matrix C, int m, int k, int n, int pitchA, int pitchB, int pitchC) {

    StopWatchInterface* timer = 0;
    sdkCreateTimer(&timer);

    // Implem_4
    const int M_BLOCK_SIZE = 128 ;
    const int N_BLOCK_SIZE = 128 ;
    const int K_BLOCK_SIZE = 8 ;

    const int A_TILE_SIZE = 4 ;
    const int B_TILE_SIZE = 4 ;

    dim3 BLOCK_DIM((M_BLOCK_SIZE * N_BLOCK_SIZE) / (A_TILE_SIZE * B_TILE_SIZE)) ;
    dim3 GRID_DIM(((n - 1) / N_BLOCK_SIZE) + 1, ((m - 1) / M_BLOCK_SIZE) + 1) ;

    timer->start();
    gpuProduct_4
    <M_BLOCK_SIZE, K_BLOCK_SIZE, N_BLOCK_SIZE, A_TILE_SIZE, B_TILE_SIZE> 
    <<<GRID_DIM, BLOCK_DIM>>>(
        A, B, C, 
        m, k, n, 
        pitchA, pitchB, pitchC
    );
    checkCudaErrors(cudaDeviceSynchronize());
    timer->stop();

    return timer->getTime() ;

}

void CudaProduct(
    Matrix hostA, Matrix hostB, Matrix hostC, 
    int m, int k, int n, 
    int mb, int nb, 
    Info *infoPtr
) {

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

    float kernelTime = callKernel(devA, devB, devC, m, k, n, pitchA, pitchB, pitchC) ;
    infoPtr->productTime = kernelTime ;

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