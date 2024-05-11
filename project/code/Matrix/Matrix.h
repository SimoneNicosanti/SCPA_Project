#pragma once

float **allocRandomMatrix(int rowsNum, int colsNum) ;
float **allocMatrix(int rowsNum, int colsNum) ;
double computeRelativeError(float **A, float **B, int m, int n) ;
void freeMatrix(float **matrix, int rows, int cols) ;