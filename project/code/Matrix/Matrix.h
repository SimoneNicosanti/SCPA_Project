#pragma once

float **allocRandomMatrix(int rowsNum, int colsNum) ;
float **allocMatrix(int rowsNum, int colsNum) ;

float computeRelativeError(float **A, float **B, int m, int n) ;