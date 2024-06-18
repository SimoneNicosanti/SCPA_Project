// Implem_2

/*
    - Loading in cache con coalesce
    - Un thread carica un singolo elemento in shared memory
    - Uso accumulatore in registro
    - Thread calcola singolo prodotto riga per colonna in accumulatore
*/

#include "SubMatricesLoader.cuh"


template <const int MB, const int KB>
__global__ void gpuProduct_2(Matrix A, Matrix B, Matrix C, int m, int k , int n, int pitchA, int pitchB, int pitchC) {

    __shared__ MatrixElemType subA[MB * KB] ;
    __shared__ MatrixElemType subB[KB * MB] ;

    int rowSubA = threadIdx.y ;
    int colSubB = threadIdx.x ;

    int rowGlobA = threadIdx.y + blockDim.y * blockIdx.y ;
    int colGlobB = threadIdx.x + blockDim.x * blockIdx.x ;

    MatrixElemType cAcc = 0.0 ;

    for (int kDispl = 0 ; kDispl < k ; kDispl += KB) {
        int currKLen = min(KB, k - kDispl) ;

        loadSubMatrices
            <MB, KB, MB>
            (A, B, m, k, n, pitchA, pitchB, kDispl, subA, subB, KB, MB) ;
        __syncthreads() ;

        if (rowGlobA < m && colGlobB < n) {
            for (int kIdx = 0 ; kIdx < currKLen ; kIdx++) {
                cAcc += subA[INDEX(rowSubA, kIdx, KB)] * subB[INDEX(kIdx, colSubB, MB)] ;
            }
        }
        __syncthreads() ;
    }
    if (rowGlobA < m && colGlobB < n) {
        C[INDEX(rowGlobA, colGlobB, pitchC)] += cAcc ;
    }
    
}
