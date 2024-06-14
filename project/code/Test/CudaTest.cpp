#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Matrix.h"
#include "CudaProduct.h"
#include "ResultWriter.h"

#define MAX_ATT 3
#define MAX_PROB_DIM 10001
#define MAX_SEQ_DIM 2001
#define START_PROB_DIM 250
#define SIZE_INCREMENT 250

double doParTest(Matrix A, Matrix B, Matrix C, int m, int k, int n) ;
double doSeqTest(Matrix A, Matrix B, Matrix C, int m, int k, int n) ;
void testProducts(int m, int k, int n, char *resultFile) ;
void squareTests() ;
void rectangularTests() ;



void squareTests() {
    for (int probDim = START_PROB_DIM ; probDim < MAX_PROB_DIM ; probDim += SIZE_INCREMENT) {
        char *outputPath = "../Results/CUDA/Tests/CUDA_Square_Test.csv" ;
        testProducts(probDim, probDim, probDim, outputPath) ;
    }
}

void rectangularTests() {
    Matrix A, B, C, parC, seqC ;
    double seqTime, parTime ;
    double relativeError ;

    int kSizesList[] = {32, 64, 128, 156} ;
    int otherSizes = 10000 ;

    for (int i = 0 ; i < 4 ; i++) {
        int k = kSizesList[i] ;
        char *outputPath = "../Results/CUDA/Tests/CUDA_Rect_Test.csv" ;
        testProducts(otherSizes, k, otherSizes, outputPath) ;
    }
}

void testProducts(int m, int k, int n, char *resultFile) {
    Matrix A, B, C, parC, seqC ;
    double seqTime, parTime ;
    double relativeError ;
    TestResult testResult ;

    testResult.processNum = -1 ;
    testResult.m = m ;
    testResult.k = k ;
    testResult.n = n ;

    A = allocRandomMatrix(m, k) ;
    B = allocRandomMatrix(k, n) ;
    C = allocRandomMatrix(m, n) ;
    parC = allocMatrix(m, n) ;
    seqC = allocMatrix(m, n) ;

    for (int att = 0 ; att < MAX_ATT ; att++) {

        // Have to copy in two different C matrices as the result is overwritten
        memcpy(parC, C, sizeof(MatrixElemType) * m * n) ;
        memcpy(seqC, C, sizeof(MatrixElemType) * m * n) ;

        testResult.parallelTime = doParTest(A, B, parC, m, k, n) ;

        // ONLY 0 does the parallel product
        if (k < 0) {
            testResult.sequentialTime = doSeqTest(A, B, seqC, m, k, n) ;
            testResult.relativeError = computeRelativeError(seqC, parC, m, n) ;
        } else {
            testResult.sequentialTime = -1 ;
            testResult.relativeError = -1 ;
        }

        // Only zero writes on the CSV file
        writeTestResult(resultFile, &testResult) ;

    }

    freeMatrix(A) ;
    freeMatrix(B) ;
    freeMatrix(C) ;
    freeMatrix(parC) ;
    freeMatrix(seqC) ;

} 


int main(int argc, char *argv[]) {
    
    srand(987654) ;
    squareTests() ;
    rectangularTests() ;

    return 0 ;
    
}

// TODO Change -> Make return Info
double doParTest(Matrix A, Matrix B, Matrix C, int m, int k, int n) {
    Info info ;
    Version version = DEFAULT ;
    CudaProduct(A, B, C, m, k, n, 0, 0, version, &info) ;

    return info.productTime ;
}

double doSeqTest(Matrix A, Matrix B, Matrix C, int m, int k, int n) {
    // TODO Change time take
    //double seqStart = MPI_Wtime() ;
    tileProduct(A, B, C, m, k, n) ;
    //double seqEnd = MPI_Wtime() ;

    return 0 ;
}