#pragma once

#include "Matrix.h"

typedef struct Info {
    float productTime ;
} Info ;

typedef enum Version {
    DEFAULT,
    ZERO,
    ONE,
    TWO,
    THREE,
    FOUR,
    FIVE,
} Version ;


void CudaProduct(Matrix hostA, Matrix hostB, Matrix hostC, int m, int k, int n, int mb, int nb, Version version, Info *infoPtr) ;