#pragma once
#include "../Matrix/Matrix.h"

typedef struct TestStruct {
    int m ;
    int n ;
    int k ;

    int parallelUnitNum ;
    double parallelTime ;

    double nonParallelTime ;

    double error ;
} TestStruct ;