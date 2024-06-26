#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "Matrix.h"


#define RAND_LOWER_BOUND -100
#define RAND_UPPER_BOUND 100


#define TILE_SIZE 64 

int min(int a, int b) {
    if (a < b) {
        return a ;
    }
    return b ;
}


MatrixElemType generateRandomNumber(int min, int max) {

    MatrixElemType scaled = (MatrixElemType)rand() / RAND_MAX ;
    return min + scaled * (max - min);
}


Matrix allocMatrix(int rowsNum, int colsNum) {
    Matrix matrixData = (Matrix) calloc(rowsNum * colsNum, sizeof(MatrixElemType));
    if (matrixData == NULL) {
        return NULL ;
    }
    
    return matrixData;
}

void copyMatrix(Matrix dstMat, Matrix srcMat, int rows, int cols) {
    memcpy(dstMat, srcMat, sizeof(MatrixElemType) * rows * cols) ;
}


/*
    Allocs a random matrix
*/
Matrix allocRandomMatrix(int rowsNum, int colsNum) {
    Matrix matrix = allocMatrix(rowsNum, colsNum) ;
    if (matrix == NULL) {
        return NULL ;
    }

    for (int i = 0 ; i < rowsNum ; i++) {
        for (int j = 0 ; j < colsNum ; j++) {
            matrix[INDEX(i,j,colsNum)] = generateRandomNumber(RAND_LOWER_BOUND, RAND_UPPER_BOUND) ;
        }
    }

    return matrix ;
}

void freeMatrix(Matrix matrix) {
    free(matrix) ;
}


void tileProduct(Matrix A, Matrix B, Matrix C, int m, int k, int n) {
    
    for (int iTile = 0 ; iTile < m ; iTile = iTile + TILE_SIZE) {
        for (int tTile = 0 ; tTile < k ; tTile = tTile + TILE_SIZE) {
            for (int jTile = 0 ; jTile < n ; jTile = jTile + TILE_SIZE) {
                
                for (int i = iTile ; i < min(iTile + TILE_SIZE, m) ; i++) {
                    for (int t = tTile ; t < min(tTile + TILE_SIZE ,k) ; t++) {
                        for (int j = jTile ; j < min(jTile + TILE_SIZE, n) ; j++) {
                            C[INDEX(i,j,n)] += A[INDEX(i,t,k)] * B[INDEX(t,j,n)] ;
                        }
                    }
                }
            }
        }
    }
}


void matrixProduct(Matrix A, Matrix B, Matrix C, int m, int k, int n) {
    for (int i = 0 ; i < m ; i++) {
        for (int t = 0 ; t < k ; t++) {
            for (int j = 0 ; j < n ; j++) {
                C[INDEX(i,j,n)] += A[INDEX(i,t,k)] * B[INDEX(t,j,n)] ;
            }
        }
    }
}


double computeRelativeError(Matrix seqMat, Matrix parMat, int rows, int cols) {
    double normNum = 0.0 ;
    double normDen = 0.0 ;

    for (int i = 0 ; i < rows ; i++) {
        double rowNormNum = 0.0 ;
        double rowNormDen = 0.0 ;
        for (int j = 0 ; j < cols ; j++) {
            rowNormNum += fabsf(parMat[INDEX(i,j,cols)] - seqMat[INDEX(i,j,cols)]) ;
            rowNormDen += fabsf(seqMat[INDEX(i,j,cols)]) ;
        }

        if (normNum < rowNormNum) {
            normNum = rowNormNum ;
        }
        if (normDen < rowNormDen) {
            normDen = rowNormDen ;
        }
    }

    return normNum / normDen ;

}