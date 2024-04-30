#include <stdio.h>
#include "Utils.h"



void printMessage(char *header, Content content, MESSAGE_TYPE type, int rank, int printerProcRank, int firstDimLen, int secDimLen, int newLine) {

    if (rank != printerProcRank) {
        return ;
    }

    printf("PROCESS %d LOG > \n", rank) ;
    switch (type)
    {
    case STRING:
        printf("%s: %s", header, content.string) ;
        break;
    
    case INT_ARRAY:
        printf("%s\n", header) ;
        printIntegerArray(content.intArray, firstDimLen) ;
        break ;
    
    case REAL_ARRAY:
        printf("%s\n", header) ;
        printRealArray(content.realArray, firstDimLen) ;
        break ;

    case MATRIX:
        printf("%s\n", header) ;
        printMatrix(content.matrix, firstDimLen, secDimLen) ;
        break ;

    case INTEGER :
        printf("%s: %d", header, content.integer) ;
        break ;

    case REAL :
        printf("%s: %d", header, content.real) ;
        break ;
    }

    if (newLine) {
        printf("\n") ;
    }
} 

void printIntegerArray(int *array, int dim) {
    printf("[") ;
    for (int i = 0 ; i < dim ; i++) {
        printf((i == dim - 1) ? "%d" : "%d ", array[i]) ;
    }
    printf("]\n") ;
}

void printRealArray(double *array, int dim) {
    printf("[") ;
    for (int i = 0 ; i < dim ; i++) {
        printf((i == dim - 1) ? "%f" : "%f ", array[i]) ;
    }
    printf("]\n") ;
}

void printMatrix(double **matrix, int firstDimLen, int secDimLen) {
    for (int i = 0 ; i < firstDimLen ; i++) {
        printRealArray(matrix[i], secDimLen) ;
    }
    printf("\n") ;
}