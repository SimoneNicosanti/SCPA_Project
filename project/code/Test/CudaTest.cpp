#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <helper_functions.h>

#include "Matrix.h"
#include "CudaProduct.h"
#include "ResultWriter.h"

#define MAX_ATT 3
#define MAX_PROB_DIM 10001
#define MAX_SEQ_DIM 5001
#define START_PROB_DIM 250
#define SIZE_INCREMENT 250

double doParTest(Matrix A, Matrix B, Matrix C, int m, int k, int n, Version version) ;
double doSeqTest(Matrix A, Matrix B, Matrix C, int m, int k, int n) ;
void testProducts(int m, int k, int n, char *resultFile, Version version) ;
void squareTests(Version version) ;
void rectangularTests(Version version) ;
int extractParams(int argc, char *argv[], Version *versionPtr) ;



void squareTests(Version version) {
    for (int probDim = START_PROB_DIM ; probDim < MAX_PROB_DIM ; probDim += SIZE_INCREMENT) {
        char *outputPath = "../Results/CUDA/Tests/CUDA_Square_Test.csv" ;
        printf("Test con >>> (%d, %d, %d)\n", probDim, probDim, probDim) ;
        testProducts(probDim, probDim, probDim, outputPath, version) ;
    }
}

void rectangularTests(Version version) {
    Matrix A, B, C, parC, seqC ;
    double seqTime, parTime ;
    double relativeError ;

    int kSizesList[] = {32, 64, 128, 156} ;
    int otherSizes = 10000 ;

    for (int i = 0 ; i < 4 ; i++) {
        int k = kSizesList[i] ;
        char *outputPath = "../Results/CUDA/Tests/CUDA_Rect_Test.csv" ;
        printf("Test con >>> (%d, %d, %d)\n", otherSizes, k, otherSizes) ;
        testProducts(otherSizes, k, otherSizes, outputPath, version) ;
    }
}

void testProducts(int m, int k, int n, char *resultFile, Version version) {
    Matrix A, B, C, parC, seqC ;
    double seqTime, parTime ;
    double relativeError ;
    TestResult testResult ;

    testResult.processNum = version ;
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

        testResult.parallelTime = doParTest(A, B, parC, m, k, n, version) ;
        testResult.parallelTime = testResult.parallelTime ;
        // ONLY 0 does the parallel product
        if (k < MAX_SEQ_DIM) {
            testResult.sequentialTime = doSeqTest(A, B, seqC, m, k, n) ;
            testResult.sequentialTime = testResult.sequentialTime ;

            testResult.relativeError = computeRelativeError(seqC, parC, m, n) ;
        } else {
            testResult.sequentialTime = -1 ;
            testResult.relativeError = -1 ;
        }

        writeTestResult(resultFile, &testResult, 1) ;
    }

    freeMatrix(A) ;
    freeMatrix(B) ;
    freeMatrix(C) ;
    freeMatrix(parC) ;
    freeMatrix(seqC) ;

} 


int main(int argc, char *argv[]) {

    Version version ;
    extractParams(argc, argv, &version) ;
    
    srand(987654) ;
    squareTests(version) ;
    rectangularTests(version) ;

    return 0 ;
    
}

double doParTest(Matrix A, Matrix B, Matrix C, int m, int k, int n, Version version) {
    Info info ;
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

int extractParams(int argc, char *argv[], Version *versionPtr) {
    
    for (int i = 0 ; i < argc - 1 ; i++) {
        
        if (strcmp(argv[i], "-v") == 0) {
            convertVersion(atoi(argv[i+1]), versionPtr) ;
        }
    }

    return 1 ;
}