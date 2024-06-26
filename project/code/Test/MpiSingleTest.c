#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Matrix.h"
#include "MpiProduct.h"
#include "ResultWriter.h"
#include "PrintUtils.h"

#define MAX_ATT 3
#define MAX_PROB_DIM 10100
#define MAX_SEQ_DIM 2001
#define START_PROB_DIM 100

double doParTest(Matrix A, Matrix B, Matrix C, int m, int k, int n, int blockRows, int blockCols) ;
double doSeqTest(Matrix A, Matrix B, Matrix C, int m, int k, int n) ;
int extractParams(int argc, char *argv[], int *mPtr, int *kPtr, int *nPtr, int *blockRowsPtr, int *blockColsPtr) ;


void main(int argc, char *argv[]) {

    int myRank ;
    int procNum ;
    MPI_Init(&argc, &argv) ;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);

    Matrix A, B, C, parC, seqC ;
    double seqTime = 0, parTime = 0 ;
    double relativeError = 0 ;

    int m, k, n, blockRows = 0, blockCols = 0 ;
    extractParams(argc, argv, &m, &k, &n, &blockRows, &blockCols) ;

    if (myRank == 0) {
        A = allocRandomMatrix(m, k) ;
        B = allocRandomMatrix(k, n) ;
        C = allocRandomMatrix(m, n) ;

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

        parC = allocMatrix(m, n) ;
        seqC = allocMatrix(m, n) ;
    }

    if (myRank == 0) {
        copyMatrix(parC, C, m, n) ;
        copyMatrix(seqC, C, m, n) ;
    }

    parTime = doParTest(A, B, parC, m, k, n, blockRows, blockCols) ;
    if (myRank == 0) {
        printf("PARALLEL TIME > %f\n", parTime) ;
    }

    // ONLY 0 does the parallel product
    if (myRank == 0 && m <= 2000) {
        seqTime = doSeqTest(A, B, seqC, m, k, n) ;
        
        relativeError = computeRelativeError(seqC, parC, m, n) ;
        printf("SEQUENTIAL TIME > %f\n", seqTime) ;
        printf("RELATIVE ERROR > %f\n", relativeError) ;

        // Content cont ;
        // cont.matrix = seqC ;
        // printMessage("SEQUENTIAL MATRIX >>> ", cont, MATRIX, 0, 0, m, n, 1) ;
    }

    if (myRank == 0) {
        freeMatrix(A) ;
        freeMatrix(B) ;
        freeMatrix(C) ;
        freeMatrix(parC) ;
        freeMatrix(seqC) ;
    }

    MPI_Finalize() ;
}

double doParTest(Matrix A, Matrix B, Matrix C, int m, int k, int n, int blockRows, int blockCols) {
    Info info ;
    MpiProduct(A, B, C, m, k, n, blockRows, blockCols, &info) ;

    return info.productTime ;
}

double doSeqTest(Matrix A, Matrix B, Matrix C, int m, int k, int n) {
    double seqStart = MPI_Wtime() ;
    tileProduct(A, B, C, m, k, n) ;
    double seqEnd = MPI_Wtime() ;

    return seqEnd - seqStart ;
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