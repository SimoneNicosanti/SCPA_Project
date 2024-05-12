#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Matrix.h"
#include "MpiProduct.h"
#include "ResultWriter.h"

#define MAX_ATT 3
#define MAX_PROB_DIM 10001
#define MAX_SEQ_DIM 2001
#define START_PROB_DIM 250
#define SIZE_INCREMENT 250

double doParTest(Matrix A, Matrix B, Matrix C, int m, int k, int n) ;
double doSeqTest(Matrix A, Matrix B, Matrix C, int m, int k, int n) ;
int extractParams(int argc, char *argv[], int *mPtr, int *kPtr, int *nPtr, int *mbPtr, int *kbPtr, int *nbPtr, int *testSeqPtr) ;


void main(int argc, char *argv[]) {

    int myRank ;
    int procNum ;
    MPI_Init(&argc, &argv) ;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);

    Matrix A, B, C, parC, seqC ;
    double seqTime, parTime ;
    double relativeError ;
    TestResult testResult ;
    testResult.processNum = procNum ;
    for (int probDim = START_PROB_DIM ; probDim < MAX_PROB_DIM ; probDim += SIZE_INCREMENT) {

        if (myRank == 0) {
            printf("TEST CON (%d, %d, %d)\n", probDim, probDim, probDim) ;
        }
        testResult.m = probDim ;
        testResult.k = probDim ;
        testResult.n = probDim ;
        if (myRank == 0) {
            A = allocRandomMatrix(probDim, probDim) ;
            B = allocRandomMatrix(probDim, probDim) ;
            C = allocRandomMatrix(probDim, probDim) ;
            parC = allocMatrix(probDim, probDim) ;
            seqC = allocMatrix(probDim, probDim) ;
        }

        for (int att = 0 ; att < MAX_ATT ; att++) {

            // Have to copy in two different C matrices as the result is overwritten
            if (myRank == 0) {
                memcpy(parC, C, sizeof(MatrixElemType) * probDim * probDim) ;
                memcpy(seqC, C, sizeof(MatrixElemType) * probDim * probDim) ;
            }

            testResult.parallelTime = doParTest(A, B, parC, probDim, probDim, probDim) ;

            // ONLY 0 does the parallel product
            if (myRank == 0 && probDim < 0) {
                testResult.sequentialTime = doSeqTest(A, B, seqC, probDim, probDim, probDim) ;
                testResult.relativeError = computeRelativeError(seqC, parC, probDim, probDim) ;
            } else {
                testResult.sequentialTime = -1 ;
                testResult.relativeError = -1 ;
            }

            // Only zero writes on the CSV file
            if (myRank == 0) {
                unsigned long gflopsNum = probDim * probDim * probDim ;
                testResult.gFLOPS = gflopsNum / testResult.parallelTime ;

                writeTestResult("../Results/MPI/MPI_Test.csv", &testResult) ;
            }
        }

        if (myRank == 0) {
            freeMatrix(A) ;
            freeMatrix(B) ;
            freeMatrix(C) ;
            freeMatrix(parC) ;
            freeMatrix(seqC) ;
        }

    }

    MPI_Finalize() ;
}

double doParTest(Matrix A, Matrix B, Matrix C, int m, int k, int n) {
    MPI_Barrier(MPI_COMM_WORLD) ;
    double startTime = MPI_Wtime() ;

    MpiProduct(A, B, C, m, k, n, 0, 0) ;

    MPI_Barrier(MPI_COMM_WORLD) ;
    double endTime = MPI_Wtime() ;

    return endTime - startTime ;
}

double doSeqTest(Matrix A, Matrix B, Matrix C, int m, int k, int n) {
    double seqStart = MPI_Wtime() ;
    matrixProduct(A, B, C, m, k, n) ;
    double seqEnd = MPI_Wtime() ;

    return seqEnd - seqStart ;
}

int extractParams(int argc, char *argv[], int *mPtr, int *kPtr, int *nPtr, int *mbPtr, int *kbPtr, int *nbPtr, int *testSeqPtr) {
    
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
        if (strcmp(argv[i], "-seq") == 0) {
            *testSeqPtr = atoi(argv[i+1]) ;
            printf("CIAO\n") ;
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