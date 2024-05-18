#pragma once

typedef struct Info {
    double scatterTime ;
    double productTime ;
    double gatherTime ;
    int productInfo ;
    int error ;
} Info ;

void MpiProduct(Matrix A, Matrix B, Matrix C, int m, int k, int n, int blockRows, int blockCols, Info *infoPtr) ;