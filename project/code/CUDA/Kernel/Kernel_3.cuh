// Implem_3

/*
    - Loading in cache con coalesce
    - Un thread carica un singolo elemento in shared memory
    - Thread calcola una sottocolonna della matrice C mettendola in accumulatore
    - Divido la subA in tiles
    - Per semplicità consideriamo BLOCK_SIZE = BLOCK_SIZE, altrimenti bisognerebbe considerare tutti i vari casi
*/


template <const int MB>
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


template<const int MB, const int KB, const int TILE_A>
__global__ void gpuProduct_3(Matrix A, Matrix B, Matrix C, int m, int k , int n, int pitchA, int pitchB, int pitchC) {
    __shared__ MatrixElemType subA[MB][KB] ;
    __shared__ MatrixElemType subB[KB][MB] ;

    int thrY = threadIdx.x / MB ;
    int thrX = threadIdx.x % MB ;

    MatrixElemType cAccArray[TILE_A] = {0.0} ;

    int loadingSubRowA = threadIdx.x / KB ;
    int loadingRowA = loadingSubRowA + MB * blockIdx.y ;
    int kSubA = threadIdx.x % KB ;

    int loadingSubColB = threadIdx.x % MB ;
    int loadingColB = loadingSubColB + MB * blockIdx.x ;
    int kSubB = threadIdx.x / MB ;

    int rowsPerBlock = min(MB, m - MB * blockIdx.y) ;
    int colsPerBlock = min(MB, n - MB * blockIdx.x) ;

    for (int kDispl = 0 ; kDispl < k ; kDispl += KB) {
        // Loading subA and subB
        int currKLen = min(KB, k - kDispl) ;

        if (loadingRowA < m && kDispl + kSubA < k) {
            subA[loadingSubRowA][kSubA] = A[INDEX(loadingRowA, kDispl + kSubA, pitchA)] ;
        }
        if (loadingColB < n && kDispl + kSubB < k) {
            subB[kSubB][loadingSubColB] = B[INDEX(kDispl + kSubB, loadingColB, pitchB)] ;
        }
        __syncthreads() ;

        // Doing tile product
        if (thrX < colsPerBlock) {
            int currTileSize = min(TILE_A, min(MB, rowsPerBlock) - TILE_A * thrY) ;
            for (int dotIdx = 0 ; dotIdx < currKLen ; dotIdx++) {
                MatrixElemType bElem = subB[dotIdx][thrX] ;
                for (int tileIdx = 0 ; tileIdx < currTileSize ; tileIdx++) {
                    cAccArray[tileIdx] += subA[tileIdx + thrY * TILE_A][dotIdx] * bElem ;
                }
            }
        }
        __syncthreads() ;
    }
    
    // Moving back to C
    int cRowStart = blockIdx.y * MB + thrY * TILE_A ;
    if (cRowStart < m && loadingColB < n) {
        int currTileSize = min(TILE_A, min(MB, rowsPerBlock) - TILE_A * thrY) ;
        for (int tileIdx = 0 ; tileIdx < currTileSize ; tileIdx++) {
            C[INDEX(tileIdx + cRowStart, loadingColB, pitchC)] += cAccArray[tileIdx] ;
            
        }
    }
}
