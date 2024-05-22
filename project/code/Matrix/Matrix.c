#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "Matrix.h"


#define RAND_LOWER_BOUND -100
#define RAND_UPPER_BOUND 100

// TODO Change Tile Size to one like those he wrote in the assignment
#define TILE_SIZE 100 

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

/*
    Allocs a matrix with continuous data but giving back
    an array pointing to the beginning of each row.
    The matrix is initialized to zero
*/
Matrix allocMatrix(int rowsNum, int colsNum) {
    Matrix matrixData = (Matrix) calloc(rowsNum * colsNum, sizeof(MatrixElemType));
    if (matrixData == NULL) {
        return NULL ;
    }

    // float **matrix = malloc(rowsNum * sizeof(float *));
    // if (matrix == NULL) {
    //     return NULL ;
    // }

    // for (int i = 0 ; i < rowsNum ; i++)
    //     matrix[i] = &(matrixData[i * colsNum]);
    
    return matrixData;
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

//TODO Forse se scambio i cicli di tTile e jTile è meglio --> Uso meglio la località
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


double computeRelativeError(Matrix A, Matrix B, int m, int n) {
    double normNum = 0 ;
    double normDen = 0 ;

    for (int i = 0 ; i < m ; i++) {
        double rowNormNum = 0 ;
        double rowNormDen = 0 ;
        for (int j = 0 ; j < n ; j++) {
            rowNormNum += fabs(B[INDEX(i,j,n)] - A[INDEX(i,j,n)]) ;
            rowNormDen += fabs(A[INDEX(i,j,n)]) ;
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