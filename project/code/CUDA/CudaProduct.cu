// Includes CUDA
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <stdio.h>

#include "Matrix.h"
#include "CudaProduct.h"

#include "Kernel_0.cuh"
#include "Kernel_1.cuh"
#include "Kernel_2.cuh"
#include "Kernel_3.cuh"
#include "Kernel_4.cuh"

#define DEF_MB 50
#define DEF_NB 50

void moveMatricesFromHostToDevice(Matrix hostMatrix, Matrix *devMatrix, int rows, int cols, size_t *pitchPtr) ;
void removeMatricesFromDevice(Matrix hostMat, Matrix devMat, int m, int k) ;
float callKernel(Matrix A, Matrix B, Matrix C, int m, int k, int n, int pitchA, int pitchB, int pitchC, Version version) ;


void callKernel_0(Matrix A, Matrix B, Matrix C, int m, int k, int n, int pitchA, int pitchB, int pitchC) {
    printf("CUDA Product Version >>> 0\n") ;
    const int BLOCK_SIZE = 32 ;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE) ;
    dim3 gridDim((m - 1) / BLOCK_SIZE + 1, (n - 1) / BLOCK_SIZE + 1) ;

    gpuProduct_0
        <<<gridDim, blockDim>>>
        (A, B, C, m, k, n, pitchA, pitchB, pitchC) ;
    checkCudaErrors(cudaDeviceSynchronize());
}

void callKernel_1(Matrix A, Matrix B, Matrix C, int m, int k, int n, int pitchA, int pitchB, int pitchC) {
    printf("CUDA Product Version >>> 1\n") ;
    const int BLOCK_SIZE = 32 ;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE) ;
    dim3 gridDim((n - 1) / BLOCK_SIZE + 1, (m - 1) / BLOCK_SIZE + 1) ;

    gpuProduct_1
        <BLOCK_SIZE>
        <<<gridDim, blockDim>>>
        (A, B, C, m, k, n, pitchA, pitchB, pitchC) ;
    checkCudaErrors(cudaDeviceSynchronize());
}

void callKernel_2(Matrix A, Matrix B, Matrix C, int m, int k, int n, int pitchA, int pitchB, int pitchC) {
    printf("CUDA Product Version >>> 2\n") ;
    const int BLOCK_SIZE = 32 ;
    const int KB = 32 ;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE) ;
    dim3 gridDim((n - 1) / BLOCK_SIZE + 1, (m - 1) / BLOCK_SIZE + 1) ;

    gpuProduct_2
        <BLOCK_SIZE, KB>
        <<<gridDim, blockDim>>>
        (A, B, C, m, k, n, pitchA, pitchB, pitchC) ;
    checkCudaErrors(cudaDeviceSynchronize());
}

void callKernel_3(Matrix A, Matrix B, Matrix C, int m, int k, int n, int pitchA, int pitchB, int pitchC) {
    printf("CUDA Product Version >>> 3\n") ;
    const int M_BLOCK_SIZE = 128 ;
    const int N_BLOCK_SIZE = 128 ;
    const int K_BLOCK_SIZE = 16 ;

    const int A_TILE_SIZE = 8 ;
    const int B_TILE_SIZE = 8 ;

    //dim3 BLOCK_DIM((M_BLOCK_SIZE * N_BLOCK_SIZE) / (A_TILE_SIZE * B_TILE_SIZE)) ;
    dim3 blockDim((N_BLOCK_SIZE / B_TILE_SIZE), (M_BLOCK_SIZE / A_TILE_SIZE)) ;
    dim3 gridDim(((n - 1) / N_BLOCK_SIZE) + 1, ((m - 1) / M_BLOCK_SIZE) + 1) ;

    gpuProduct_3
        <M_BLOCK_SIZE, K_BLOCK_SIZE, N_BLOCK_SIZE, A_TILE_SIZE, B_TILE_SIZE> 
        <<<gridDim, blockDim>>>(
            A, B, C, 
            m, k, n, 
            pitchA, pitchB, pitchC
        );
    checkCudaErrors(cudaDeviceSynchronize());
}

void callKernel_4(Matrix A, Matrix B, Matrix C, int m, int k, int n, int pitchA, int pitchB, int pitchC) {
    printf("CUDA Product Version >>> 4\n") ;
    const int M_BLOCK_SIZE = 128 ;
    const int N_BLOCK_SIZE = 128 ;
    const int K_BLOCK_SIZE = 16 ;

    const int A_TILE_SIZE = 8 ;
    const int B_TILE_SIZE = 8 ;

    dim3 blockDim((N_BLOCK_SIZE / B_TILE_SIZE), (M_BLOCK_SIZE / A_TILE_SIZE)) ;
    dim3 gridDim(((n - 1) / N_BLOCK_SIZE) + 1, ((m - 1) / M_BLOCK_SIZE) + 1) ;

    gpuProduct_4
        <M_BLOCK_SIZE, K_BLOCK_SIZE, N_BLOCK_SIZE, A_TILE_SIZE, B_TILE_SIZE> 
        <<<gridDim, blockDim>>>(
            A, B, C, 
            m, k, n, 
            pitchA, pitchB, pitchC
        );
    checkCudaErrors(cudaDeviceSynchronize());
}

float callKernel(Matrix A, Matrix B, Matrix C, int m, int k, int n, int pitchA, int pitchB, int pitchC, Version version) {

    StopWatchInterface* timer = 0;
    sdkCreateTimer(&timer);

    timer->start();
    switch (version)
    {
        case ZERO:
            callKernel_0(A, B, C, m, k, n, pitchA, pitchB, pitchC) ;
            break ;
        case ONE:
            callKernel_1(A, B, C, m, k, n, pitchA, pitchB, pitchC) ;
            break ;
        case TWO:
            callKernel_2(A, B, C, m, k, n, pitchA, pitchB, pitchC) ;
            break ;
        case THREE:
            callKernel_3(A, B, C, m, k, n, pitchA, pitchB, pitchC) ;
            break ;
        case FOUR:
            callKernel_4(A, B, C, m, k, n, pitchA, pitchB, pitchC) ;
            break ;
        case DEFAULT:
            callKernel_4(A, B, C, m, k, n, pitchA, pitchB, pitchC) ;
    }
    timer->stop();    

    return timer->getTime() ;

}

void CudaProduct(
    Matrix hostA, Matrix hostB, Matrix hostC, 
    int m, int k, int n, 
    int mb, int nb, 
    Version version,
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

    float kernelTime = callKernel(devA, devB, devC, m, k, n, pitchA, pitchB, pitchC, version) ;
    infoPtr->productTime = kernelTime ;

    checkCudaErrors(
        cudaMemcpy2D(hostC, sizeof(MatrixElemType) * n, devC, pitchC * sizeof(MatrixElemType), sizeof(MatrixElemType) * n, m, cudaMemcpyDeviceToHost)
    ) ;

    removeMatricesFromDevice(hostA, devA, m, k) ;
    removeMatricesFromDevice(hostB, devB, k, n) ;
    removeMatricesFromDevice(hostC, devC, m, n) ;

    return ;
}



void moveMatricesFromHostToDevice(Matrix hostMatrix, Matrix *devMatrixPtr, int rows, int cols, size_t *pitchPtr) {
    checkCudaErrors(
        cudaHostRegister(hostMatrix, sizeof(MatrixElemType) * rows * cols, cudaHostRegisterDefault)
    ) ;
    checkCudaErrors(
        cudaMallocPitch((void **) devMatrixPtr, pitchPtr, sizeof(MatrixElemType) * cols, rows)
    ) ;
    checkCudaErrors(
        cudaMemcpy2D(*devMatrixPtr, *pitchPtr, hostMatrix, sizeof(MatrixElemType) * cols, sizeof(MatrixElemType) * cols, rows, cudaMemcpyHostToDevice)
    ) ;
    *pitchPtr = *pitchPtr / sizeof(MatrixElemType) ;
}

void removeMatricesFromDevice(Matrix hostMat, Matrix devMat, int m, int k) {
    checkCudaErrors(
        cudaFree(devMat) 
    ) ;
    
    checkCudaErrors(
        cudaHostUnregister(hostMat)
    ) ;

}

void convertVersion(int versionInt, Version *versionPtr) {
    switch (versionInt)
    {
    case 0:
        *versionPtr = ZERO ;
        break ;
    case 1:
        *versionPtr = ONE ;
        break ;
    case 2 :
        *versionPtr = TWO ;
        break ;
    case 3 :
        *versionPtr = THREE ;
        break ;
    case 4:
        *versionPtr = FOUR ;
        break;
    default:
        *versionPtr = DEFAULT ;
        break;
    }
}