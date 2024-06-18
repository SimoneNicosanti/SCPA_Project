#include "Matrix.h"

// TODO > Problem with kernel_3 and kernel_4 when KB > b_m or b_n


template <const int MB, const int KB, const int NB>
__device__ void loadSubMatrices(
    Matrix A, Matrix B, 
    int m, int k, int n, 
    int pitchA, int pitchB, 
    int kDispl, 
    Matrix subA, Matrix subB,
    int colsSubA, int colsSubB,
    bool isSubATransposed = false
) {
    int startLoadSubRowA = threadIdx.y ;
    int startLoadRowA = MB * blockIdx.y ;
    int rowsPerBlock = min(MB, m - MB * blockIdx.y) ;

    for (int kSubA = threadIdx.x ; kSubA < min(k - kDispl, KB) ; kSubA += blockDim.x) {
        for (int loadRowIdx = startLoadSubRowA ; loadRowIdx < rowsPerBlock ; loadRowIdx += blockDim.y) {
            if (!isSubATransposed) {
                subA[INDEX(loadRowIdx, kSubA, colsSubA)] = A[INDEX(startLoadRowA + loadRowIdx, kDispl + kSubA, pitchA)] ;
            } else {
                subA[INDEX(kSubA, loadRowIdx, colsSubA)] = A[INDEX(startLoadRowA + loadRowIdx, kDispl + kSubA, pitchA)] ;
            }
        }
    }

    int startLoadSubColB = threadIdx.x ;
    int startLoadColB = NB * blockIdx.x ;
    int colsPerBlock = min(NB, n - NB * blockIdx.x) ;

    for (int kSubB = threadIdx.y ; kSubB < min(k - kDispl, KB) ; kSubB += blockDim.y) {
        for (int loadColIdx = startLoadSubColB ; loadColIdx < colsPerBlock ; loadColIdx += blockDim.x) {
            subB[INDEX(kSubB, loadColIdx, colsSubB)] = B[INDEX(kDispl + kSubB, startLoadColB + loadColIdx, pitchB)] ;
        }
    }
}