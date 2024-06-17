// Implem_2

/*
    - Loading in cache con coalesce
    - Un thread carica un singolo elemento in shared memory
    - Uso accumulatore in registro
    - Thread calcola singolo prodotto riga per colonna in accumulatore
    - Per semplicità consideriamo BLOCK_SIZE = KB, altrimenti bisognerebbe considerare tutti i vari casi
*/

template <const int MB, const int KB>
__device__ void loadSubMatrices_2(
    Matrix A, Matrix B, 
    int m, int k, int n, 
    int pitchA, int pitchB, 
    int kDispl, 
    Matrix subA, Matrix subB
) {
    int rowSubA = threadIdx.x / MB ;
    int colSubB = threadIdx.x % MB ;

    int rowGlobA = rowSubA + MB * blockIdx.y ;
    int colGlobB = colSubB + MB * blockIdx.x ;

    int kSubA = threadIdx.x % MB ;
    int kSubB = threadIdx.x / MB ;

    if (kSubA + kDispl < k && rowGlobA < m) {
        subA[INDEX(rowSubA, kSubA, MB)] = A[INDEX(rowGlobA, kDispl + kSubA, pitchA)] ;
    }

    if (kSubB + kDispl < k && colGlobB < n) {
        subB[INDEX(kSubB, colSubB, MB)] = B[INDEX(kDispl + kSubB, colGlobB, pitchB)] ;
    }

}

template <const int MB, const int KB>
__global__ void gpuProduct_2(Matrix A, Matrix B, Matrix C, int m, int k , int n, int pitchA, int pitchB, int pitchC) {

    __shared__ MatrixElemType subA[MB * MB] ;
    __shared__ MatrixElemType subB[MB * MB] ;

    int rowSubA = threadIdx.x / MB ;
    int colSubB = threadIdx.x % MB ;

    int rowGlobA = rowSubA + MB * blockIdx.y ;
    int colGlobB = colSubB + MB * blockIdx.x ;

    MatrixElemType cAcc = 0.0 ;

    for (int kDispl = 0 ; kDispl < k ; kDispl += MB) {
        int currKLen = min(MB, k - kDispl) ;

        loadSubMatrices_2
            <MB, KB>
            (A, B, m, k, n, pitchA, pitchB, kDispl, subA, subB) ;
        __syncthreads() ;

        if (rowGlobA < m && colGlobB < n) {
            for (int kIdx = 0 ; kIdx < currKLen ; kIdx++) {
                cAcc += subA[INDEX(rowSubA, kIdx, MB)] * subB[INDEX(kIdx, colSubB, MB)] ;
            }
        }
        __syncthreads() ;
    }
    if (rowGlobA < m && colGlobB < n) {
        C[INDEX(rowGlobA, colGlobB, pitchC)] += cAcc ;
    }
    
}
