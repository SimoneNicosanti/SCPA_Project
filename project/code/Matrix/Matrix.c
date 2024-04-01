#include <stdlib.h>
#include "Matrix.h"


#define RAND_LOWER_BOUND -100
#define RAND_UPPER_BOUND 100
#define RAND_MAX 1000

double generateRandomNumber(int min, int max) {
    return min + ((double)rand() / RAND_MAX) * (max - min);
}

/*
    Allocs matrix per blocks and fills it with random generated numbers
*/
Matrix *randomMatrixConstructor(int totalRows, int totalCols, int totalRowBlocks, int totalColBlocks) {
    Matrix *matrix = matrixConstructor(totalRows, totalCols, totalRowBlocks, totalColBlocks) ;
    if (matrix == NULL) {
        return NULL ;
    }

    for (int i = 0 ; i < totalRows * totalCols ; i++) {
        matrix->data[i] = generateRandomNumber(RAND_LOWER_BOUND, RAND_UPPER_BOUND) ;
    }

    return matrix ;
}

void computeBlockSize(int totalRows, int totalCols, int totalRowBlocks, int totalColBlocks, int rowBlockIndex, int colBlockIndex, int *rowsInBlock, int *colsInBlock) {
    int rowsDiv = totalRows / totalRowBlocks ;
    int rowsMod = totalRows % totalRowBlocks ;

    int colsDiv = totalCols / totalColBlocks ;
    int colsMod = totalCols % totalColBlocks ;

    if (rowBlockIndex < rowsMod) {
        *rowsInBlock = rowsDiv + 1 ;
    } else {
        *rowsInBlock = rowsDiv ;
    }

    if (colBlockIndex < colsMod) {
        *colsInBlock = colsDiv + 1 ;
    } else {
        *colsInBlock = colsDiv ;
    }
}

/*
    Allocs matrix per blocks.
    In detail: we have a double pointer paradigm where :
    * the first pointer points to a flat array of block pointers
    * the second pointer points to a flat array of double (which is the block)
    
    Params:
    * totalRows = num rows in matrix
    * totalCols = num cols in matrix
    * totalRowBlocks = num of row blocks
    * totalColsBloks = num of col blocks
    
    Note:
    If totalRowBlocks == 1 & totalColBlocks == 1, then we have the classical flat allocation for matrix
*/
Matrix *matrixConstructor(int totalRows, int totalCols, int totalRowBlocks, int totalColBlocks) {

    int totalElems = totalRows * totalCols ;

    Matrix *matrix = (Matrix *) malloc(sizeof(Matrix)) ;
    if (matrix == NULL) {
        return NULL ;
    }

    int rowsInBlock ;
    int colsInBlock ;

    matrix->data = (double *) malloc(sizeof(double) * totalElems) ;
    matrix->rowsPerBlockArray = (int *) malloc(sizeof(int) * totalRowBlocks) ;
    matrix->colsPerBlockArray = (int *) malloc(sizeof(int) * totalColBlocks) ;
    if (matrix->data == NULL || matrix->rowsPerBlockArray == NULL || matrix->colsPerBlockArray == NULL) {
        return NULL ;
    }

    for (int rowBlockIndex = 0 ; rowBlockIndex < totalRowBlocks ; rowBlockIndex++) {
        for (int colBlockIndex = 0 ; colBlockIndex < totalColBlocks ; colBlockIndex++) {
            computeBlockSize(totalRows, totalCols, totalRowBlocks, totalColBlocks, rowBlockIndex, colBlockIndex, &rowsInBlock, &colsInBlock) ;
            matrix->rowsPerBlockArray[rowBlockIndex] = rowsInBlock ;
            matrix->colsPerBlockArray[colBlockIndex] = colsInBlock ;
        }
    }

    matrix->totalRows = totalRows ;
    matrix->totalCols = totalCols ;
    matrix->totalRowBlocks = totalRowBlocks ;
    matrix->totalColBlocks = totalColBlocks ;
}

Matrix *copyMatrix(Matrix *matrix) {
    Matrix *copy = matrixConstructor(matrix->totalRows, matrix->totalCols, matrix->totalRowBlocks, matrix->totalColBlocks) ;
    if (copy == NULL) {
        return NULL ;
    }

    for (int i = 0 ; i < matrix->totalRows * matrix->totalCols ; i++) {
        copy->data[i] = matrix->data[i] ;
    }

    return copy ;
}

// double computeMatrixModule(Matrix *matrix) {
//     double module = 0 ;
//     for (int i = 0 ; i < matrix->rowNum ; i++) {
//         for (int j = 0 ; j < matrix->colNum ; j++) {
//             module = module + pow(matrix->matrix[i][j], 2) ;
//         }
//     }
//     return sqrt(module) ;
// }

// //TODO Controlla che il calcolo errore sulle differenze dei moduli
// double compareMatrix(Matrix *parResult, Matrix *nonParResult) {
//     double parModule = computeMatrixModule(parResult) ;
//     double nonParModule = computeMatrixModule(nonParResult) ;
//     double diff = parModule - nonParModule ;
    
//     return (diff > 0) ? diff : -diff ;
// }