#include <stdio.h>
#include "Test.h"


void buildTestResult(TestStruct *testResult, int parallelUnitNum, double testValue, int m, int n, int k, double error) {
    testResult->parallelUnitNum = parallelUnitNum ;
    testResult->parallelTime = testValue ;
    testResult->m = m ;
    testResult->n = n ;
    testResult->k = k ;
    testResult->error = error ;
}


void writeTestResult(char *fileName, TestStruct *testResult) {
    FILE *fileDesc ;
    fileDesc = fopen(fileName, "r") ;
    if (fileDesc == NULL) {
        // File non esiste
        prepareResultFile(fileName) ;
    } else {
        fclose(fileDesc) ;
    }

    fileDesc = fopen(fileName, "a+") ;
    if (fileDesc == NULL) {
        //TODO Error in printing result
    }
    fprintf(fileDesc, "%d;%f;%f;%d;%d;%d\n", testResult->parallelUnitNum, testResult->parallelTime, testResult->error, testResult->m, testResult->n, testResult->k) ;
    fclose(fileDesc) ;
}


void prepareResultFile(char *fileName) {
    FILE *fileDesc = fopen(fileName, "w+") ;
    if (fileDesc == NULL) {
        //TODO Error in creating file result
    }
    fprintf(fileDesc, "%s;%s;%s;%s;%s;%s\n", "ParallelUnitNum", "Time", "Error", "m", "n", "k") ;
    fclose(fileDesc) ;
}