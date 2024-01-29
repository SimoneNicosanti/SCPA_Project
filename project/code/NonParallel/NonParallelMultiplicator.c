#include <time.h>
/*
    Matrix product: C <-- C + A*B
    * A is m x k
    * B is k x n
    * C is m x n
*/
time_t nonParallelMultiplicate(double **matrixA, double **matrixB, double **matrixC, int m, int n, int k) {
    time_t startTime = time(NULL) ;

    for (int rowIndex = 0 ; rowIndex < m ; rowIndex++) {
        double *vecA = matrixA[rowIndex] ;
        double vecB[k] ;
        for (int colIndex = 0 ; colIndex < n ; colIndex++) {
            extractColumn(matrixB, vecB, k, colIndex) ;
            double rowColProd = vectorProduct(vecA, vecB, k) ;
            matrixC[rowIndex][colIndex] += rowColProd ;
        }
    }

    time_t endTime = time(NULL) ;

    return endTime - startTime ;
}

void extractColumn(double **matrix, double *columnVec, int k, int colIndex) {
    for (int i = 0 ; i < k ; i++) {
        columnVec[i] = matrix[i][colIndex] ;
    }
}

double vectorProduct(double *vec_1, double *vec_2, double vecLen) {
    double result = 0 ;
    for (int index = 0 ; index < vecLen ; index++) {
        result += vec_1[index] * vec_2[index] ;
    }
    return result ;
}