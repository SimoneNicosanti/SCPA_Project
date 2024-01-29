#pragma once
#include "../Matrix/Matrix.h"

typedef struct TestStruct {
    int m ;
    int n ;
    int k ;

    int parallelUnitNum ;
    double parallelTime ;

    double nonParallelTime ;

    double error ;
} TestStruct ;


double compareMatrix(Matrix *parResult, Matrix *nonParResult) ;

void nonParallelMultiplicate(double **matrixA, double **matrixB, double **matrixC, int m, int n, int k) ;

void buildTestResult(TestStruct *testResult, int parallelUnitNum, double testValue, int m, int n, int k, double error) ;
void writeTestResult(char *fileName, TestStruct *testResult) ;