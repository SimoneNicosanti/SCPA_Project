#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Matrix.h"
#include "MpiProduct.h"
#include "ResultWriter.h"

#define MAX_ATT 3
#define MAX_PROB_DIM 10001
#define MAX_SEQ_DIM 5001
#define START_PROB_DIM 250
#define SIZE_INCREMENT 250

double doParTest(Matrix A, Matrix B, Matrix C, int m, int k, int n) ;
double doSeqTest(Matrix A, Matrix B, Matrix C, int m, int k, int n) ;
int extractParams(int argc, char *argv[], int *mPtr, int *kPtr, int *nPtr, int *mbPtr, int *kbPtr, int *nbPtr, int *testSeqPtr) ;
void testProducts(int myRank, int procNum, int m, int k, int n, char *resultFile) ;
void squareTests(int procNum, int myRank) ;
void rectangularTests(int procNum, int myRank) ;



void squareTests(int procNum, int myRank) {
    for (int probDim = START_PROB_DIM ; probDim < MAX_PROB_DIM ; probDim += SIZE_INCREMENT) {
        if (myRank == 0) {
            printf("#Processi %d > TEST CON (%d, %d, %d)\n", procNum, probDim, probDim, probDim) ;
        }
        char *outputPath = "../Results/MPI/Tests/MPI_Square_Test.csv" ;
        testProducts(myRank, procNum, probDim, probDim, probDim, outputPath) ;
    }
}

void rectangularTests(int procNum, int myRank) {
    Matrix A, B, C, parC, seqC ;
    double seqTime, parTime ;
    double relativeError ;

    int kSizesList[] = {32, 64, 128, 156} ;
    int otherSizes = 10000 ;

    for (int i = 0 ; i < 4 ; i++) {
        int k = kSizesList[i] ;
        if (myRank == 0) {
            printf("#Processi %d > TEST CON (%d, %d, %d)\n", procNum, otherSizes, k, otherSizes) ;
        }

        char *outputPath = "../Results/MPI/Tests/MPI_Rect_Test.csv" ;
        testProducts(myRank, procNum, otherSizes, k, otherSizes, outputPath) ;
    }
}

void testProducts(int myRank, int procNum, int m, int k, int n, char *resultFile) {
    Matrix A, B, C, parC, seqC ;
    double seqTime, parTime ;
    double relativeError ;
    TestResult testResult ;

    testResult.processNum = procNum ;
    testResult.m = m ;
    testResult.k = k ;
    testResult.n = n ;
    if (myRank == 0) {
        A = allocRandomMatrix(m, k) ;
        B = allocRandomMatrix(k, n) ;
        C = allocRandomMatrix(m, n) ;
        parC = allocMatrix(m, n) ;
        seqC = allocMatrix(m, n) ;
    }

    for (int att = 0 ; att < MAX_ATT ; att++) {

        // Have to copy in two different C matrices as the result is overwritten
        if (myRank == 0) {
            copyMatrix(parC, C, m, n) ;
            copyMatrix(seqC, C, m, n) ;
        }

        testResult.parallelTime = doParTest(A, B, parC, m, k, n) ;

        // ONLY 0 does the parallel product
        if (myRank == 0 && k < MAX_SEQ_DIM) {
            testResult.sequentialTime = doSeqTest(A, B, seqC, m, k, n) ;
            testResult.relativeError = computeRelativeError(seqC, parC, m, n) ;
        } else {
            testResult.sequentialTime = -1 ;
            testResult.relativeError = -1 ;
        }

        // Only zero writes on the CSV file
        if (myRank == 0) {
            writeTestResult(resultFile, &testResult, 0) ;
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


void main(int argc, char *argv[]) {

    int myRank ;
    int procNum ;
    MPI_Init(&argc, &argv) ;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);
    
    srand(987654) ;
    squareTests(procNum, myRank) ;
    rectangularTests(procNum, myRank) ;
    
    MPI_Finalize() ;
}


double doParTest(Matrix A, Matrix B, Matrix C, int m, int k, int n) {
    Info info ;
    MpiProduct(A, B, C, m, k, n, 0, 0, &info) ;

    return info.productTime ;
}

double doSeqTest(Matrix A, Matrix B, Matrix C, int m, int k, int n) {
    double seqStart = MPI_Wtime() ;
    tileProduct(A, B, C, m, k, n) ;
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