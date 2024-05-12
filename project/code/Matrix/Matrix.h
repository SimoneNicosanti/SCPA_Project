#pragma once

#define INDEX(i, j, numCols) i * numCols + j

typedef float * Matrix ;
typedef float MatrixElemType ;

void matrixProduct(Matrix A, Matrix B, Matrix C, int m, int k, int n) ;
Matrix allocMatrix(int rowsNum, int colsNum) ;
Matrix allocRandomMatrix(int rowsNum, int colsNum) ;
void freeMatrix(Matrix matrix) ;
double computeRelativeError(Matrix A, Matrix B, int m, int n) ;