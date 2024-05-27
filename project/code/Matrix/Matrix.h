#pragma once

#define INDEX(i, j, numCols) i * (numCols) + j

typedef float * Matrix ;
typedef float MatrixElemType ;

#ifdef __cplusplus
extern "C"
#endif
void matrixProduct(Matrix A, Matrix B, Matrix C, int m, int k, int n) ;

#ifdef __cplusplus
extern "C"
#endif
void tileProduct(Matrix A, Matrix B, Matrix C, int m, int k, int n) ;

#ifdef __cplusplus
extern "C"
#endif
Matrix allocMatrix(int rowsNum, int colsNum) ;

#ifdef __cplusplus
extern "C"
#endif
Matrix allocRandomMatrix(int rowsNum, int colsNum) ;

#ifdef __cplusplus
extern "C"
#endif
void freeMatrix(Matrix matrix) ;

#ifdef __cplusplus
extern "C"
#endif
double computeRelativeError(Matrix A, Matrix B, int m, int n) ;