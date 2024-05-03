#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "Matrix.h"


#define RAND_LOWER_BOUND -100
#define RAND_UPPER_BOUND 100

float generateRandomNumber(int min, int max) {

    float scaled = (float)rand() / RAND_MAX ;
    return min + scaled * (max - min);
}

/*
    Allocs a matrix with continuous data but giving back
    an array pointing to the beginning of each row.
    The matrix is initialized to zero
*/
float **allocMatrix(int rowsNum, int colsNum) {
    float *matrixData = calloc(rowsNum * colsNum, sizeof(float));
    if (matrixData == NULL) {
        return NULL ;
    }

    float **matrix = malloc(rowsNum * sizeof(float *));
    if (matrix == NULL) {
        return NULL ;
    }

    for (int i = 0 ; i < rowsNum ; i++)
        matrix[i] = &(matrixData[i * colsNum]);
    
    return matrix;
}


/*
    Allocs a random matrix
*/
float **allocRandomMatrix(int rowsNum, int colsNum) {
    float **matrix = allocMatrix(rowsNum, colsNum) ;
    if (matrix == NULL) {
        return NULL ;
    }

    for (int i = 0 ; i < rowsNum ; i++) {
        for (int j = 0 ; j < colsNum ; j++) {
            matrix[i][j] = generateRandomNumber(RAND_LOWER_BOUND, RAND_UPPER_BOUND) ;
        }
    }

    return matrix ;
}

float computeRelativeError(float **A, float **B, int m, int n) {
    double normNum = 0 ;
    double normDen = 0 ;

    for (int i = 0 ; i < m ; i++) {
        double rowNorm = 0 ;
        for (int j = 0 ; j < n ; j++) {
            rowNorm += abs(A[i][j] - B[i][j]) ;
        }

        if (normNum < rowNorm) {
            normNum = rowNorm ;
        }
    }

    for (int i = 0 ; i < m ; i++) {
        double rowNorm = 0 ;
        for (int j = 0 ; j < n ; j++) {
            rowNorm += abs(A[i][j]) ;
        }

        if (normDen < rowNorm) {
            normDen = rowNorm ;
        }
    }

    return normNum / normDen ;

}