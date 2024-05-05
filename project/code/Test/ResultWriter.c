#include <stdio.h>
#include <unistd.h>
#include "ResultWriter.h"


void writeTestResult(char *fileName, TestResult *testResult) {
    int fileAccess = access(fileName, F_OK) ;
    if (fileAccess == -1) {
        // File non esiste -> Create it
        prepareResultFile(fileName) ;
    }
    
    FILE *fileDesc = fopen(fileName, "a+") ;
    fprintf(fileDesc, "%d,%d,%d,%d,%f,%f,%f\n",
        testResult->m, testResult->n, testResult->k,
        testResult->processNum, testResult->parallelTime,
        testResult->sequentialTime,
        testResult->relativeError
    ) ;
    fclose(fileDesc) ;
}

void prepareResultFile(char *fileName) {
    FILE *fileDesc = fopen(fileName, "w+") ;
    if (fileDesc == NULL) {
        //TODO Error in creating file result
    }
    fprintf(fileDesc, "%s,%s,%s,%s,%s,%s,%s\n", "M", "N", "K", "ProcessNum", "ParallelTime", "SequentialTime", "RelativeError") ;
    fclose(fileDesc) ;
}