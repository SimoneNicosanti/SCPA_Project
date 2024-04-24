#include <stdlib.h>
#include <stdio.h>
#include "Matrix.h"


#define RAND_LOWER_BOUND -100
#define RAND_UPPER_BOUND 100
#define RAND_MAX 1000

double generateRandomNumber(int min, int max) {
    return min + ((double)rand() / RAND_MAX) * (max - min);
}

/*
    Allocs a matrix with continuous data but giving back
    an array pointing to the beginning of each row
*/
double **allocMatrix(int rowsNum, int colsNum) {
    double *matrixData = calloc(rowsNum * colsNum, sizeof(double));
    if (matrixData == NULL) {
        return NULL ;
    }

    double **matrix = malloc(rowsNum * sizeof(double *));
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
double **allocRandomMatrix(int rowsNum, int colsNum) {
    double **matrix = allocMatrix(rowsNum, colsNum) ;
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

// // Prints a matrix
// void printMatrix(double **matrix, int rowsNum, int colsNum) {
//     for (int i = 0 ; i < rowsNum ; i++) {
//         for (int j = 0 ; j < colsNum ; j++) {
//             printf("%f ", matrix[i][j]) ;
//         }
//         printf("\n") ;
//     }
// }