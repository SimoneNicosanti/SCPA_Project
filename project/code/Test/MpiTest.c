#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Matrix.h"
#include "Sequential.h"
#include "Test.h"

#define MAX_ATT 3


void main(int argc, char *argv[]) {

    int m = 0 ;
    int k = 0 ;
    int n = 0 ;
    int mb = 0 ;
    int kb = 0 ;
    int nb = 0 ;
    int testSeq = 0 ;
    int succ = extractParams(argc, argv, &m, &k, &n, &mb, &kb, &nb, &testSeq) ;
    if (!succ) {
        //ERRORE
    }

    int myRank ;
    int procNum ;
    MPI_Init(&argc, &argv) ;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);

    float **A, **B, **C, **parC, **seqC ;
    double seqTime, parTime ;
    float relativeError ;
    TestResult testResult ;
    for (int probDim = 100 ; probDim < 10100 ; probDim += 100) {
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
                memcpy(&(parC[0][0]), &(C[0][0]), sizeof(float) * m * n) ;
                memcpy(&(seqC[0][0]), &(C[0][0]), sizeof(float) * m * n) ;
            }

            testResult.parallelTime = doParTest(A, B, parC, probDim, probDim, probDim) ;

            // ONLY 0 does the parallel product
            if (myRank == 0) {
                testResult.sequentialTime = doSeqTest(A, B, seqC, probDim, probDim, probDim) ;
                testResult.relativeError = computeRelativeError(seqC, parC, m, n) ;
            }

            // Only zero writes on the CSV file
            if (myRank == 0) {
                unsigned long gflopsNum = probDim * probDim * probDim ;
                testResult.gFLOPS = gflopsNum / testResult.parallelTime ;

                writeTestResult("../Results/MPI_Test.csv", &testResult) ;
            }
        }

    }

    MPI_Finalize() ;
}

double doParTest(float **A, float **B, float **C, int m, int k, int n) {
    MPI_Barrier(MPI_COMM_WORLD) ;
    double startTime = MPI_Wtime() ;
    //CALL TO PARALLEL PRODUCT
    double endTime = MPI_Wtime() ;

    return endTime - startTime ;
}

double doSeqTest(float **A, float **B, float **C, int m, int k, int n) {
    double seqStart = MPI_Wtime() ;
    sequentialMultiplication(A, B, C, m, k, n) ;
    double seqEnd = MPI_Wtime() ;

    return seqEnd - seqStart ;
}