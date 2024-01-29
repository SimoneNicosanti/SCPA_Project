#include <stdlib.h>
#include "Matrix.h"


#define RAND_LOWER_BOUND -100
#define RAND_UPPER_BOUND 100
#define RAND_MAX 10000

double generateRandomNumber(int min, int max) {
    return min + ((double)rand() / RAND_MAX) * (max - min);
}

Matrix *randomMatrixConstructor(int rowNum, int colNum) {
    Matrix *matrix = matrixConstructor(rowNum, colNum) ;
    if (matrix == NULL) {
        return NULL ;
    }

    for (int row = 0 ; row < matrix->rowNum ; row++) {
        for (int col = 0 ; col < matrix->colNum ; col++) {
            matrix->matrix[row][col] = generateRandomNumber(RAND_LOWER_BOUND, RAND_UPPER_BOUND) ;
        }
    }
    return matrix ;
}

Matrix *matrixConstructor(int rowNum, int colNum) {

    Matrix *matrixStruct = (Matrix *) malloc(sizeof(Matrix)) ;

    double **matrix = (double **) malloc(sizeof(double *) * rowNum) ;
    if (matrix == NULL) {
        return NULL ;
    }
    for (int row = 0 ; row < rowNum ; row++) {
        matrix[row] = (double *) calloc(colNum, sizeof(double)) ;
        if (matrix[row] == NULL) {
            return NULL ;
        }
    }

    matrixStruct->matrix = matrix ;
    matrixStruct->rowNum = rowNum ;
    matrixStruct->colNum = colNum ;

    return matrixStruct ;
}

Matrix *copyMatrix(Matrix *matrix) {
    double **copy = matrixConstructor(matrix->rowNum, matrix->colNum) ;
    if (copy == NULL) {
        return NULL ;
    }
    for (int i = 0 ; i < matrix->rowNum ; i++) {
        for (int j = 0 ; j < matrix->colNum ; j++) {
            copy[i][j] = matrix->matrix[i][j] ;
        }
    }
    return copy ;
}

double computeMatrixModule(Matrix *matrix) {
    double module = 0 ;
    for (int i = 0 ; i < matrix->rowNum ; i++) {
        for (int j = 0 ; j < matrix->colNum ; j++) {
            module = module + pow(matrix->matrix[i][j], 2) ;
        }
    }
    return sqrt(module) ;
}

//TODO Controlla che il calcolo errore sulle differenze dei moduli
double compareMatrix(Matrix *parResult, Matrix *nonParResult) {
    double parModule = computeMatrixModule(parResult) ;
    double nonParModule = computeMatrixModule(nonParResult) ;
    double diff = parModule - nonParModule ;
    
    return (diff > 0) ? diff : -diff ;
}