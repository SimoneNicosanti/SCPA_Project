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

#ifdef __cplusplus
extern "C"
#endif
void writeTestResult(char *fileName, TestResult *testResult) ;