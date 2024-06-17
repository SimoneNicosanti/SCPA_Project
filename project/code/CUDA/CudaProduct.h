#pragma once

#include "Matrix.h"

typedef struct Info {
    float productTime ;
} Info ;

typedef enum Version {
    DEFAULT,
    FOUR,
    FIVE,
    SIX
} Version ;


void CudaProduct(Matrix hostA, Matrix hostB, Matrix hostC, int m, int k, int n, int mb, int nb, Version version, Info *infoPtr) ;