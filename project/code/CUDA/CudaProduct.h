#pragma once

#include "Matrix.h"

typedef struct Info {
    float productTime ;
} Info ;


void CudaProduct(Matrix hostA, Matrix hostB, Matrix hostC, int m, int k, int n, int mb, int nb, Info *infoPtr) ;