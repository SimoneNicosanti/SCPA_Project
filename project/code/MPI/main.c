#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <time.h>
#include <string.h>

#include "MpiProduct.h"
#include "../Matrix/Matrix.h"
#include "Utils.h"

#define CART_DIMS 3

MPI_Comm WORLD_COMM ;
MPI_Comm CART_COMM ;
MPI_Comm *sumCommArray ;

ProcInfo procInfo ;


int extractParams(int argc, char *argv[], int *mPtr, int *kPtr, int *nPtr) ;
void createSpreadCommunicator(int procRank, int *dims) ;
void createSumCommunicatorArray(int *dims) ;
// void scatterMatrixes(int procNum, int *dims, Matrix *matrixA, Matrix *matrixB, Matrix *matrixC, int m, int k, int n) ;



int main(int argc, char *argv[]) {
    // Prima di prendere il tempo devo generare le matrici, altrimenti il tempo di generazione è compreso
    int m = 0;
    int k = 0;
    int n = 0;
    int succ = extractParams(argc, argv, &m, &k, &n) ;
    if (!succ) {
        //ERRORE
    }

    MPI_Init(&argc, &argv);

    int myRank ;
    int procNum ;
    MPI_Comm_rank(WORLD_COMM, &myRank);
    MPI_Comm_size(WORLD_COMM, &procNum);

    /*
        Ottengo una terna (i, k, j) dove:
        * i = # di suddivisioni sulle righe di A
        * k = # di suddivisioni sulle colonne di A e sulle righe di B
        * j = # di suddivisioni sulle colonne di B
    */
    int dims[3] = {0} ;
    MPI_Dims_create(procNum, CART_DIMS, dims) ;

    Matrix *matrixA, *matrixB, *matrixC ;
    if (myRank == 0) {
        matrixA = randomMatrixConstructor(m, k, dims[0], dims[1]) ;
        matrixB = randomMatrixConstructor(k, n, dims[1], dims[2]) ;
        matrixC = randomMatrixConstructor(m, n, dims[0], dims[2]) ; 
    }

    createSpreadCommunicator(myRank, dims) ;
    createSumCommunicatorArray(dims) ;

    int subANumRows ;
    int subANumCols ;
    computeBlockSize(m, k, dims[0], dims[1], procInfo.procRowIndex, procInfo.procDepthIndex, &subANumRows, &subANumCols) ;
    double subBlockA[subANumRows * subANumCols] ;

    int subBNumRows ;
    int subBNumCols ;
    computeBlockSize(k, n, dims[1], dims[2], procInfo.procDepthIndex, procInfo.procColIndex, &subBNumRows, &subBNumCols) ;
    double subBlockB[subBNumRows * subBNumCols] ;

    // TODO DO THE SAME FOR C TOO !!!
    // double subBlockC[] ;

    //scatterMatrixes(procNum, dims, matrixA, matrixB, matrixC, m, k, n) ;

    // time_t startTime = time(NULL) ;
    // MPI_Comm_dup(MPI_COMM_WORLD, &MY_COMM_WORLD) ;
    
    // CALL TO MPI PRODUCT
    MPI_Finalize();

    time_t endTime = time(NULL) ;
    // CALL TO WRITE TEST RESULT

    return 0;
}


void createSpreadCommunicator(int procRank , int *dims) {
    int periods[CART_DIMS] = {0} ;
    MPI_Cart_create(WORLD_COMM, CART_DIMS, dims, periods, 0, &CART_COMM) ;

    int coords[CART_DIMS] ;
    int rank ;

    for (int i = 0 ; i < dims[0] ; i++) {
        for (int j = 0 ; j < dims[2] ; j++) {
            for (int h = 0 ; h < dims[1] ; h++) {
                coords[0] = i ;
                coords[1] = h ;
                coords[2] = j ;
                
                MPI_Cart_rank(CART_COMM, coords, rank) ;

                if (rank == procRank) {
                    procInfo.procRank = procRank ;
                    procInfo.procRowIndex = i ;
                    procInfo.procColIndex = j ;
                    procInfo.procDepthIndex = h ;
                }

            }
        }
    }
}

/*
    Crea un array di comunicatori per sommare i diversi prodotti del singolo prodotto riga - colonna.
    L'array rappresenta una matrice di comunicatori dove il comunicatore [i][j] ricostruisce il prodotto
    complessivo relativo ad A[i] * B[j]
*/
void createSumCommunicatorArray(int *dims) {

    sumCommArray = (MPI_Comm *) malloc(sizeof(MPI_Comm) * dims[0] * dims[2]) ;
    if (sumCommArray == NULL) {
        //ERRORE
    }

    int processRanks[dims[1]] ;
    MPI_Group oldGroup ;
    MPI_Group newGroup ;
    MPI_Comm_group(WORLD_COMM, &oldGroup) ;

    for (int i = 0 ; i < dims[0] ; i++) {
        for (int j = 0 ; j < dims[2] ; j++) {
            for (int h = 0 ; h < dims[1] ; h++) {
                int procRank ;
                int coords[CART_DIMS] = {i, h, j} ;
                MPI_Cart_rank(CART_COMM, coords, &procRank) ;
                processRanks[h] = procRank ;
            }

            MPI_Group_incl(oldGroup, dims[1], processRanks, newGroup) ;
            MPI_Comm_create(WORLD_COMM, newGroup, sumCommArray[j + i * dims[0]]) ;
        }
    }
}

void scatterMatrixes(int procNum, int *dims, 
    Matrix *matrixA, double *subBlockA, int sizeSubBlockA, 
    Matrix *matrixB, double *subBlockB, int sizeSubBlockB, 
    Matrix *matrixC, 
    int m, int k, int n) {

    int coords[CART_DIMS] ;
    int rank ;

    // Scatter A
    int matrixASendCounts[dims[0] * dims[1] * dims[2]] ;
    int matrixADispls[dims[0] * dims[1] * dims[2]] ;
    int procSubANumRows, procSubANumCols ;
    int displsA = 0 ;
    for (int i = 0 ; i < dims[0] ; i++) {
        for (int h = 0 ; h < dims[1] ; h++) {
            for (int j = 0 ; j < dims[2] ; j++) {
                coords[0] = i ;
                coords[1] = h ;
                coords[2] = j ;
                
                MPI_Cart_rank(CART_COMM, coords, rank) ;
                computeBlockSize(m, k, dims[0], dims[1], procInfo.procRowIndex, procInfo.procDepthIndex, &procSubANumRows, &procSubANumCols) ;
                
                matrixASendCounts[rank] = procSubANumRows * procSubANumCols ;
                matrixADispls[rank] = displsA ;
                displsA += procSubANumRows * procSubANumCols ;
            }
        }
    }
    MPI_Scatterv(matrixA->data, matrixASendCounts, matrixADispls, MPI_DOUBLE, subBlockA, sizeSubBlockA, MPI_DOUBLE, 0, CART_COMM) ;

    // Scatter B
    int matrixBSendCounts[dims[0] * dims[1] * dims[2]] ;
    int matrixBDispls[dims[0] * dims[1] * dims[2]] ;
    int procSubBNumRows, procSubBNumCols ;
    int *displsB = (int *) calloc(dims[1], sizeof(int)) ;
    int displsBOnRows = 0 ;
    for (int i = 0; i < dims[0] ; i++) {
        memset(displsB, 0, sizeof(int) * dims[1]) ;
        for (int j = 0 ; j < dims[2] ; j++) {
            displsBOnRows = 0 ;
            for (int h = 0 ; h < dims[1] ; h++) {
                coords[0] = i ;
                coords[1] = h ;
                coords[2] = j ;
                
                MPI_Cart_rank(CART_COMM, coords, rank) ;
                computeBlockSize(k, n, dims[1], dims[2], procInfo.procDepthIndex, procInfo.procColIndex, &procSubBNumRows, &procSubBNumCols) ;
                
                matrixBSendCounts[rank] = procSubBNumRows * procSubBNumCols ;
                matrixBDispls[rank] = displsB[h] + displsBOnRows ;

                displsB[h] += procSubBNumRows * procSubBNumCols ;
                displsBOnRows += procSubBNumRows * n ;
            }
        }
    }
    MPI_Scatterv(matrixB->data, matrixBSendCounts, matrixBDispls, MPI_DOUBLE, subBlockB, sizeSubBlockB, MPI_DOUBLE, 0, CART_COMM) ;


    // Scatter C : di C va fatta solo a coloro che la prendono tutta ?

}


int extractParams(int argc, char *argv[], int *mPtr, int *kPtr, int *nPtr) {
    for (int i = 0 ; i < argc - 1 ; i++) {
        if (strcmp(argv[i], "-m") == 0) {
            *mPtr = atoi(argv[i+1]) ;
        }
        if (strcmp(argv[i], "-k") == 0) {
            *kPtr = atoi(argv[i+1]) ;
        }
        if (strcmp(argv[i], "-n") == 0) {
            *nPtr = atoi(argv[i+1]) ;
        }
    }

    if (*mPtr <= 0 || *kPtr <= 0 || *nPtr <= 0) {
        return 0 ;
    }

    return 1 ;
}