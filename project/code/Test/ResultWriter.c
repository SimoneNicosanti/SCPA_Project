#include <stdio.h>
#include <unistd.h>
#include "ResultWriter.h"

void prepareResultFile(char *fileName, int framework) {
    FILE *fileDesc = fopen(fileName, "w+") ;
    if (fileDesc == NULL) {
        //TODO Error in creating file result
    }

    fprintf(fileDesc, "%s,%s,%s,%s,%s,%s,%s\n", "M", "N", "K", framework == 0 ? "ProcessNum" : "KernelVersion", "ParallelTime", "SequentialTime", "RelativeError") ;
    fclose(fileDesc) ;
}

void writeTestResult(char *fileName, TestResult *testResult, int framework) {
    int fileAccess = access(fileName, F_OK) ;
    if (fileAccess == -1) {
        // File non esiste -> Create it
        prepareResultFile(fileName, framework) ;
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

