#pragma once

typedef struct Matrix {
    double *data ;
    int totalRows ;
    int totalCols ;
    int totalRowBlocks ;
    int totalColBlocks ;
    int *rowsPerBlockArray ;
    int *colsPerBlockArray ;
} Matrix ;

Matrix *randomMatrixConstructor(int totalRows, int totalCols, int totalRowBlocks, int totalColBlocks) ;
void computeBlockSize(int totalRows, int totalCols, int totalRowBlocks, int totalColBlocks, int rowBlockIndex, int colBlockIndex, int *rowsInBlock, int *colsInBlock) ;
Matrix *matrixConstructor(int totalRows, int totalCols, int totalRowBlocks, int totalColBlocks) ;
Matrix *copyMatrix(Matrix *matrix) ;