// Implem_3
/*
__global__ void gpuProduct(Matrix A, Matrix B, Matrix C, int m, int k , int n, int pitchA, int pitchB, int pitchC) {
    __shared__ MatrixElemType subA[BLOCK_SIZE][K_BLOCK_LEN] ;
    __shared__ MatrixElemType subB[K_BLOCK_LEN][BLOCK_SIZE] ;

    int thrY = threadIdx.x / BLOCK_SIZE ;
    int thrX = threadIdx.x % BLOCK_SIZE ;

    MatrixElemType cAccArray[TILE_SIZE] = {0.0} ;

    int loadingSubRowA = threadIdx.x / K_BLOCK_LEN ;
    int loadingRowA = loadingSubRowA + BLOCK_SIZE * blockIdx.y ;
    int kSubA = threadIdx.x % K_BLOCK_LEN ;

    int loadingSubColB = threadIdx.x % BLOCK_SIZE ;
    int loadingColB = loadingSubColB + BLOCK_SIZE * blockIdx.x ;
    int kSubB = threadIdx.x / BLOCK_SIZE ;

    int rowsPerBlock = min(BLOCK_SIZE, m - BLOCK_SIZE * blockIdx.y) ;
    int colsPerBlock = min(BLOCK_SIZE, n - BLOCK_SIZE * blockIdx.x) ;

    for (int kDispl = 0 ; kDispl < k ; kDispl += K_BLOCK_LEN) {
        // Loading subA and subB
        int currKLen = min(K_BLOCK_LEN, k - kDispl) ;

        if (loadingRowA < m && kDispl + kSubA < k) {
            subA[loadingSubRowA][kSubA] = A[INDEX(loadingRowA, kDispl + kSubA, pitchA)] ;
        }
        if (loadingColB < n && kDispl + kSubB < k) {
            subB[kSubB][loadingSubColB] = B[INDEX(kDispl + kSubB, loadingColB, pitchB)] ;
        }
        __syncthreads() ;

        // Doing tile product
        if (thrX < colsPerBlock) {
            int currTileSize = min(TILE_SIZE, min(BLOCK_SIZE, rowsPerBlock) - TILE_SIZE * thrY) ;
            for (int dotIdx = 0 ; dotIdx < currKLen ; dotIdx++) {
                MatrixElemType bElem = subB[dotIdx][thrX] ;
                for (int tileIdx = 0 ; tileIdx < currTileSize ; tileIdx++) {
                    cAccArray[tileIdx] += subA[tileIdx + thrY * TILE_SIZE][dotIdx] * bElem ;
                }
            }
        }
        __syncthreads() ;
    }
    
    // Moving back to C
    int cRowStart = blockIdx.y * BLOCK_SIZE + thrY * TILE_SIZE ;
    if (cRowStart < m && loadingColB < n) {
        int currTileSize = min(TILE_SIZE, min(BLOCK_SIZE, rowsPerBlock) - TILE_SIZE * thrY) ;
        for (int tileIdx = 0 ; tileIdx < currTileSize ; tileIdx++) {
            C[INDEX(tileIdx + cRowStart, loadingColB, pitchC)] += cAccArray[tileIdx] ;
            
        }
    }
}
*/