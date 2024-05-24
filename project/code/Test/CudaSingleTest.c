#include "Matrix.h"
#include "ResultWriter.h"

// Includes CUDA
#include <cuda_runtime.h>

// // Utilities and timing functions
// #include <helper_functions.h>  // includes cuda.h and cuda_runtime_api.h

// // CUDA helper functions
// #include <helper_cuda.h>  // helper functions for CUDA error check

int extractParams(int argc, char *argv[], int *mPtr, int *kPtr, int *nPtr, int *blockRowsPtr, int *blockColsPtr) ;
void moveMatricesFromHostToDevice(Matrix hostMatrix, Matrix devMatrix, int rows, int cols, size_t *pitchPtr) ;


// TODO CHECK FOR CUDA ERRORS
void main(int argc, char *argv[]) {

    int m, k, n, blockRows = 0, blockCols = 0 ;
    extractParams(argc, argv, &m, &k, &n, &blockRows, &blockCols) ;

    Matrix A = allocRandomMatrix(m, k) ; // TODO Check if can change with cudaHostAllocMapped --> Need to change the allocations
    Matrix B = allocRandomMatrix(k, n) ;
    Matrix C = allocRandomMatrix(m, n) ;

    Matrix devA, devB, devC ;
    size_t pitchA, pitchB, pitchC ;
    moveMatricesFromHostToDevice(A, devA, m, k, &pitchA) ;
    moveMatricesFromHostToDevice(B, devB, k, n, &pitchB) ;
    moveMatricesFromHostToDevice(C, devC, m, n, &pitchC) ;

    /*
        3. Call Cuda multiplication
        4. Meanwhile, if needed, do sequential multiplication
        5. Take times and stuff
        6. Check relative error if needed
    */

    cudaFree(devA) ;
    cudaFree(devB) ;
    cudaFree(devC) ;

    free(A) ;
    free(B) ;
    free(C) ;
}

void moveMatricesFromHostToDevice(Matrix hostMatrix, Matrix devMatrix, int rows, int cols, size_t *pitchPtr) {
    cudaHostRegister(hostMatrix, sizeof(MatrixElemType) * rows * cols, cudaHostRegisterDefault) ;
    cudaMallocPitch((void **) &devMatrix, pitchPtr, sizeof(MatrixElemType) * cols, rows) ;

    // TODO Check first sizeof(MatrixElemType) * cols
    cudaMemcpy2D(devMatrix, *pitchPtr, hostMatrix, sizeof(MatrixElemType) * cols, sizeof(MatrixElemType) * cols, rows, cudaMemcpyHostToDevice) ;
    *pitchPtr = *pitchPtr / sizeof(MatrixElemType) ;
}

int extractParams(int argc, char *argv[], int *mPtr, int *kPtr, int *nPtr, int *blockRowsPtr, int *blockColsPtr) {
    
    // if (argc < 6) {
    //     return 0 ;
    // }
    
    for (int i = 0 ; i < argc - 1 ; i++) {
        if (strcmp(argv[i], "-m") == 0) {
            *mPtr = atoi(argv[i+1]) ;
        }
        if (strcmp(argv[i], "-k") == 0) {
            *kPtr = atoi(argv[i+1]) ;
        }
        if (strcmp(argv[i], "-n") == 0) {
            *nPtr = atoi(argv[i+1]) ;
        }
        if (strcmp(argv[i], "-blockRows") == 0) {
            *blockRowsPtr = atoi(argv[i+1]) ;
        }
        if (strcmp(argv[i], "-blockCols") == 0) {
            *blockColsPtr = atoi(argv[i+1]) ;
        }
        // if (strcmp(argv[i], "-nb") == 0) {
        //     *nbPtr = atoi(argv[i+1]) ;
        // }
    }
    if (*mPtr <= 0 || *kPtr <= 0 || *nPtr <= 0 || *blockRowsPtr < 0 || *blockColsPtr < 0) {
        return 0 ;
    }

    return 1 ;
}