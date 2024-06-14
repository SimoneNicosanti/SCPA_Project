// Implem_2
/*
// BOH NON FUNZIONA BENE...
__global__ void gpuProduct(Matrix A, Matrix B, Matrix C, int m, int k , int n, int pitchA, int pitchB, int pitchC) {

    __shared__ MatrixElemType subA[BLOCK_SIZE][K_BLOCK_LEN] ;
    __shared__ MatrixElemType subB[K_BLOCK_LEN][BLOCK_SIZE] ;

    int thrY = threadIdx.x / BLOCK_SIZE ;
    int thrX = threadIdx.x % BLOCK_SIZE ;

    int row = thrY + BLOCK_SIZE * blockIdx.y ;
    int col = thrX + BLOCK_SIZE * blockIdx.x ;

    MatrixElemType cAccArray[BLOCK_SIZE] = {0.0} ;

    int rowsPerBlock = min(BLOCK_SIZE, m - BLOCK_SIZE * blockIdx.y) ;
    int colsPerBlock = min(BLOCK_SIZE, n - BLOCK_SIZE * blockIdx.x) ;

    for (int kDispl = 0 ; kDispl < k ; kDispl += K_BLOCK_LEN) {
        int currKLen = min(K_BLOCK_LEN, k - kDispl) ;

        int kIdxA = threadIdx.x % K_BLOCK_LEN ;
        int kIdxB = threadIdx.x / K_BLOCK_LEN ;
        
        if (kIdxA < currKLen && row < m) {
            subA[thrY][kIdxA] = A[INDEX(row, kDispl + kIdxA, pitchA)] ;
        }
        if (kIdxB < currKLen && col < n) {
            subB[kIdxB][thrX] = B[INDEX(kDispl + kIdxB, col, pitchB)] ;
        }
        __syncthreads() ;

        if (threadIdx.x < colsPerBlock) {
            for (int dotIdx = 0 ; dotIdx < currKLen ; dotIdx++) {
                MatrixElemType elemMatB = subB[dotIdx][threadIdx.x] ;
                for (int tileIdx = 0 ; tileIdx < rowsPerBlock ; tileIdx++) {
                    cAccArray[tileIdx] += subA[tileIdx][dotIdx] * elemMatB ;
                }
            }
        }
        __syncthreads() ;
    }

    int cRow = threadIdx.x + BLOCK_SIZE * blockIdx.y ;
    int cCol = threadIdx.x + BLOCK_SIZE * blockIdx.x ;
    if (cRow < m && cCol < n) {
        for (int tileIdx = 0 ; tileIdx < rowsPerBlock ; tileIdx++) {
            C[INDEX(tileIdx + BLOCK_SIZE * blockIdx.y, cCol, pitchC)] += cAccArray[tileIdx] ;
        } 
    }
    
}
*/