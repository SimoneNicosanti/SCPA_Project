#pragma once

#include "Matrix.h"

typedef struct Info {
    float productTime ;
} Info ;

typedef enum Version {
    ZERO = 0,
    ONE,
    TWO,
    THREE,
    FOUR,
    DEFAULT
} Version ;

void CudaProduct(Matrix hostA, Matrix hostB, Matrix hostC, int m, int k, int n, int mb, int nb, Version version, Info *infoPtr) ;

void convertVersion(int versionInt, Version *versionPtr) ;