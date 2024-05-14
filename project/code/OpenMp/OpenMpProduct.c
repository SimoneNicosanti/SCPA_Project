#include "omp.h"
#include <stdio.h>
#include "Matrix.h"

void openMpProduct(Matrix A, Matrix B, Matrix C, int m, int k, int n, int blockRows) {
    /*
        TODO > Potrei specificare la dimensione del chunk come la dimensione del blocco di riga??
        In questo modo ogni thread si occupa di calcolare uno o più blocchi, però forse è meglio senza in questo modo ho
        un carico meglio bilanciato tra i diversi thread; oppure lo faccio dynamic e per blocchi
    */
    int i, j, t ;
    #pragma omp parallel for shared(A, B, C, m, n, k) private(i, j, t) schedule(dynamic)
    for (i = 0 ; i < m ; i++) {
        for (t = 0 ; t < k ; t++) {
            for (j = 0 ; j < n ; j++) {
                C[INDEX(i,j,n)] += A[INDEX(i,t,k)] * B[INDEX(t,j,n)] ;
            }
        }
    }
    return ;
}