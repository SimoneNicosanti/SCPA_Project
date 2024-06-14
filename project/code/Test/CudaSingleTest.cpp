#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <helper_functions.h>

#include "CudaProduct.h"
#include "PrintUtils.h"


#include "Matrix.h"
#include "ResultWriter.h"



int extractParams(int argc, char *argv[], int *mPtr, int *kPtr, int *nPtr, int *blockRowsPtr, int *blockColsPtr, Version *versionPtr) ;
void convertVersion(int versionInt, Version *version) ;


int main(int argc, char *argv[]) {
    int m, k, n, blockRows = 0, blockCols = 0 ;
    Version version ;
    extractParams(argc, argv, &m, &k, &n, &blockRows, &blockCols, &version) ;

    Matrix A = allocRandomMatrix(m, k) ; // TODO Check if can change with cudaHostAllocMapped --> Need to change the allocations
    Matrix B = allocRandomMatrix(k, n) ;
    Matrix C = allocRandomMatrix(m, n) ;

    for (int i = 0 ; i < m ; i++) {
        for (int j = 0 ; j < k ; j++) {
            A[INDEX(i,j,k)] = i*k + j ;
        }
    }
    for (int i = 0 ; i < k ; i++) {
        for (int j = 0 ; j < n ; j++) {
            B[INDEX(i,j,n)] = i*n + j ;
        }
    }
    for (int i = 0 ; i < m ; i++) {
        for (int j = 0 ; j < n ; j++) {
            C[INDEX(i,j,n)] = i*n + j ;
        }
    }

    Matrix parC = allocMatrix(m, n) ;
    copyMatrix(parC, C, m, n) ;

    Matrix seqC = allocMatrix(m, n) ;
    copyMatrix(seqC, C, m, n) ;

    Info info ;
    Version ver ;
    
    CudaProduct(A, B, parC, m, k, n, blockRows, blockCols, version, &info) ;

    StopWatchInterface* timer = 0;
    sdkCreateTimer(&timer);
    printf("GPU Time >>> %f\n", info.productTime) ;
    printf("GPU GFLOPS >>> %f\n", (2.e-6 * m * k * n) / info.productTime) ;

    timer->start();
    matrixProduct(A, B, seqC, m, k, n) ;
    timer->stop();

    float seqTime = timer->getTime() ;
    printf("CPU Time >>> %f\n", seqTime) ;
    printf("CPU GFLOPS >>> %f\n", (2.e-6 * m * k * n) / seqTime) ;

    double relErr = computeRelativeError(seqC, parC, m, n) ;
    printf("Relative Error >>> %f\n", relErr) ;

    // Content cont ;
    // cont.matrix = seqC ;
    // printMessage("SEQ MATRIX >>> ", cont, MATRIX, 0, 0, m, n, 1) ;

    // cont.matrix = parC ;
    // printMessage("PAR MATRIX >>> ", cont, MATRIX, 0, 0, m, n, 1) ;
    

    free(A) ;
    free(B) ;
    free(C) ;

    return 0 ;
}



int extractParams(int argc, char *argv[], int *mPtr, int *kPtr, int *nPtr, int *blockRowsPtr, int *blockColsPtr, Version *versionPtr) {
    
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
        if (strcmp(argv[i], "-v") == 0) {
            convertVersion(atoi(argv[i+1]), versionPtr) ;
        }
    }
    if (*mPtr <= 0 || *kPtr <= 0 || *nPtr <= 0 || *blockRowsPtr < 0 || *blockColsPtr < 0 || *versionPtr < 0) {
        return 0 ;
    }

    return 1 ;
}

void convertVersion(int versionInt, Version *versionPtr) {
    switch (versionInt)
    {
    case 4:
        *versionPtr = FOUR ;
        break;
    case 5:
        *versionPtr = FIVE ;
    default:
        *versionPtr = DEFAULT ;
        break;
    }
}