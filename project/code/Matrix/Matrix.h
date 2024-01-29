#pragma once

typedef struct Matrix {
    double **matrix ;
    int rowNum ;
    int colNum ;
} Matrix ;

Matrix *randomMatrixConstructor(int rowNum, int colNum) ;
Matrix *matrixConstructor(int rowNum, int colNum) ;
Matrix *copyMatrix(Matrix *matrix) ;

