#include <time.h>

void sequentialMultiplication(float **A, float **B, float **C, int m, int k, int n) {
    for (int i = 0 ; i < m ; i++) {
        for (int j = 0 ; j < n ; j++) {
            for (int t = 0 ; t < k ; t++) {
                C[i][j] += A[i][t] * B[t][j] ;
            }
        }
    }
}