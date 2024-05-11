#include "omp.h"
#include <stdio.h>

void openMpProduct(float **A, float **B, float **C, int m, int k, int n) {
    /*
        TODO > Potrei specificare la dimensione del chunk come la dimensione del blocco di riga??
        In questo modo ogni thread si occupa di calcolare uno o più blocchi, però forse è meglio senza in questo modo ho
        un carico meglio bilanciato tra i diversi thread; oppure lo faccio dynamic e per blocchi
        TODO > Prova a cambiare ordine di ciclo
    
    */
    
    #pragma omp parallel for shared(A, B, C, m, n, k) schedule(static)
    for (int i = 0 ; i < m ; i++) {
        for (int t = 0 ; t < k ; t++) {
            for (int j = 0 ; j < n ; j++) {
                C[i][j] += A[i][t] * B[t][j] ;
            }
        }
    }
    return ;
}