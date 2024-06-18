#include "Matrix.h"


template <const int MB, const int KB, const int NB>
__device__ void loadSubMatrices(
    Matrix A, Matrix B, 
    int m, int k, int n, 
    int pitchA, int pitchB, 
    int kDispl, 
    Matrix subA, Matrix subB,
    int colsSubA, int colsSubB
) {
    int startLoadSubRowA = threadIdx.y ;
    int startLoadRowA = MB * blockIdx.y ;
    int kSubA = threadIdx.x ;
    int rowsPerBlock = min(MB, m - MB * blockIdx.y) ;

    for (int kSubA = threadIdx.x ; kSubA < min(k - kDispl, KB) ; kSubA += blockDim.x) {
        for (int loadRowIdx = startLoadSubRowA ; loadRowIdx < rowsPerBlock ; loadRowIdx += blockDim.y) {
            subA[INDEX(loadRowIdx, kSubA, colsSubA)] = A[INDEX(startLoadRowA + loadRowIdx, kDispl + kSubA, pitchA)] ;
        }
    }

    int startLoadSubColB = threadIdx.x ;
    int startLoadColB = NB * blockIdx.x ;
    int kSubB = threadIdx.y  ;
    int colsPerBlock = min(NB, n - NB * blockIdx.x) ;

    for (int kSubB = threadIdx.y ; kSubB < min(k - kDispl, KB) ; kSubB += blockDim.y) {
        for (int loadColIdx = startLoadSubColB ; loadColIdx < colsPerBlock ; loadColIdx += blockDim.x) {
            subB[INDEX(kSubB, loadColIdx, colsSubB)] = B[INDEX(kDispl + kSubB, startLoadColB + loadColIdx, pitchB)] ;
        }
    }
}