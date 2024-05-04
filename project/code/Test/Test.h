#pragma once

typedef struct TestResult {
    int m ;
    int n ;
    int k ;

    int processNum ;
    double parallelTime ;

    double sequentialTime ;

    double relativeError ;
    double gFLOPS ;
} TestResult ;

void writeTestResult(char *fileName, TestResult *testResult) ;