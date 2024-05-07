#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Matrix.h"
#include "MpiProduct.h"
#include "Sequential.h"
#include "ResultWriter.h"
#include "PrintUtils.h"

#define MAX_ATT 3
#define MAX_PROB_DIM 10100
#define MAX_SEQ_DIM 2001
#define START_PROB_DIM 100

double doParTest(float **A, float **B, float **C, int m, int k, int n) ;
double doSeqTest(float **A, float **B, float **C, int m, int k, int n) ;
int extractParams(int argc, char *argv[], int *mPtr, int *kPtr, int *nPtr, int *mbPtr, int *kbPtr, int *nbPtr) ;


void main(int argc, char *argv[]) {

    int myRank ;
    int procNum ;
    MPI_Init(&argc, &argv) ;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);

    float **A, **B, **C, **parC, **seqC ;
    double seqTime, parTime ;
    float relativeError ;

    int m, k, n, mb, kb, nb ;
    extractParams(argc, argv, &m, &k, &n, &mb, &kb, &nb) ;

    if (myRank == 0) {
        A = allocRandomMatrix(m, k) ;
        B = allocRandomMatrix(k, n) ;
        C = allocRandomMatrix(m, n) ;
        parC = allocMatrix(m, n) ;
        seqC = allocMatrix(m, n) ;
    }

    if (myRank == 0) {
        memcpy(&(parC[0][0]), &(C[0][0]), sizeof(float) * m * n) ;
        memcpy(&(seqC[0][0]), &(C[0][0]), sizeof(float) * m * n) ;
    }

    double testTime = doParTest(A, B, parC, m, k, n) ;
    if (myRank == 0) {
        printf("TestTime > %f\n", testTime) ;
    }

    // ONLY 0 does the parallel product
    if (myRank == 0 && m < 0) {
        doSeqTest(A, B, seqC, m, k, n) ;
        computeRelativeError(seqC, parC, m, n) ;
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

double doParTest(float **A, float **B, float **C, int m, int k, int n) {
    MPI_Barrier(MPI_COMM_WORLD) ;
    double startTime = MPI_Wtime() ;

    MpiProduct(A, B, C, m, k, n) ;

    MPI_Barrier(MPI_COMM_WORLD) ;
    double endTime = MPI_Wtime() ;

    return endTime - startTime ;
}

double doSeqTest(float **A, float **B, float **C, int m, int k, int n) {
    double seqStart = MPI_Wtime() ;
    sequentialMultiplication(A, B, C, m, k, n) ;
    double seqEnd = MPI_Wtime() ;

    return seqEnd - seqStart ;
}

int extractParams(int argc, char *argv[], int *mPtr, int *kPtr, int *nPtr, int *mbPtr, int *kbPtr, int *nbPtr) {
    
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
        if (strcmp(argv[i], "-mb") == 0) {
            *mbPtr = atoi(argv[i+1]) ;
        }
        if (strcmp(argv[i], "-kb") == 0) {
            *kbPtr = atoi(argv[i+1]) ;
        }
        // if (strcmp(argv[i], "-nb") == 0) {
        //     *nbPtr = atoi(argv[i+1]) ;
        // }
    }
    *nbPtr = *nPtr ;
    if (*mPtr <= 0 || *kPtr <= 0 || *nPtr <= 0 || *mbPtr <= 0 || *kbPtr <= 0 || *nbPtr <= 0) {
        return 0 ;
    }

    return 1 ;
}