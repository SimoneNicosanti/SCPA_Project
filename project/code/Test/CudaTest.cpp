#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <helper_functions.h>

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
        copyMatrix(parC, C, m, n) ;
        copyMatrix(seqC, C, m, n) ;

        testResult.parallelTime = doParTest(A, B, parC, m, k, n) ;
        testResult.parallelTime = testResult.parallelTime * 1.e-3 ;
        // ONLY 0 does the parallel product
        if (k < 0) {
            testResult.sequentialTime = doSeqTest(A, B, seqC, m, k, n) ;
            testResult.sequentialTime = testResult.sequentialTime * 1.e-3 ;

            testResult.relativeError = computeRelativeError(seqC, parC, m, n) ;
        } else {
            testResult.sequentialTime = -1 ;
            testResult.relativeError = -1 ;
        }

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

double doParTest(Matrix A, Matrix B, Matrix C, int m, int k, int n) {
    Info info ;
    Version version = DEFAULT ;
    CudaProduct(A, B, C, m, k, n, 0, 0, version, &info) ;

    return info.productTime ;
}

double doSeqTest(Matrix A, Matrix B, Matrix C, int m, int k, int n) {

    StopWatchInterface* timer = 0;
    sdkCreateTimer(&timer);

    timer->start() ;
    tileProduct(A, B, C, m, k, n) ;
    timer->stop() ;

    return timer->getTime() ;
}