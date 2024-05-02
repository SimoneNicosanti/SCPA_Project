#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <time.h>
#include <string.h>

#include "MpiProduct.h"
#include "../Matrix/Matrix.h"
#include "Utils.h"

#define PROCESS_GRID_DIMS 2
#define TYPE_MATRIX_NUM MPI_DOUBLE
#define ROOT_PROCESS 0
#define SEND_TAG 100

MPI_Comm WORLD_COMM ;
MPI_Comm CART_COMM ;
MPI_Comm *REDUCE_COMM_ARRAY ;

typedef struct ProcessInfo {
    int myRank ;
    int myCoords[2] ;
} ProcessInfo ;

ProcessInfo processInfo ;
int PROCESS_GRID[PROCESS_GRID_DIMS] = {0} ;

int extractParams(int argc, char *argv[], int *mPtr, int *kPtr, int *nPtr, int *mbPtr, int *kbPtr, int *nbPtr) ;
void createSendDataTypes(int rowsNum, int colsNum, int blockRows, int blockCols, int *processGrid, MPI_Datatype typesMatrix[3][3]) ;

MPI_Datatype computeSendTypeForProc(
    int rowRank, int colRank, int *processGrid,
    int rowsNum, int colsNum, 
    int blockRows, int blockCols,  
    MPI_Datatype typesMatrix[3][3]
) ;

void computeSubMatrixDimsPerProc(
    int rowRank, int colRank, int *processGrid, 
    int rowsNum, int colsNum, 
    int blockRows, int blockCols, 
    int *subMatRowsNum, int *subMatColsNum
) ;

void executeCompleteProduct(
    double **subA, double **subB, double **subC,
    int m, int k, int n,
    int mb, int kb, int nb,
    int subm, int subk, int subn,
    double **C
) ;

void matrixSend(double **matrix, int rows, int cols, int blockRows, int blockCols, int *processGrid) ;
double **matrixRecv(int rows, int cols, int blockRows, int blockCols, int *processGrid, int *subRowsPtr, int *subColsPtr) ;

void createReduceCommunicator() ;
void subMatrixProduct(double **subA, double **subB, double **cSubProd, int subm, int subk, int subn) ;


int main(int argc, char *argv[]) {
    // Prima di prendere il tempo devo generare le matrici, altrimenti il tempo di generazione è compreso

    //srand(time(NULL));

    int m = 0 ;
    int k = 0 ;
    int n = 0 ;
    int mb = 0 ;
    int kb = 0 ;
    int nb = 0 ;
    int succ = extractParams(argc, argv, &m, &k, &n, &mb, &kb, &nb) ;
    if (!succ) {
        //ERRORE
    }

    MPI_Init(&argc, &argv) ;
    MPI_Comm_dup(MPI_COMM_WORLD, &WORLD_COMM);

    int procNum ;
    MPI_Comm_rank(WORLD_COMM, &processInfo.myRank);
    MPI_Comm_size(WORLD_COMM, &procNum);

    //int PROCESS_GRID[PROCESS_GRID_DIMS] = {0} ;
    MPI_Dims_create(procNum, PROCESS_GRID_DIMS, PROCESS_GRID) ;
    if (processInfo.myRank == ROOT_PROCESS) {
        printf("Cartesian Grid: [%d, %d]\n", PROCESS_GRID[0], PROCESS_GRID[1]) ;
    }
    int periods[2] = {0, 0} ;
    MPI_Cart_create(WORLD_COMM, 2, PROCESS_GRID, periods, 0, &CART_COMM) ;
    MPI_Cart_coords(CART_COMM, processInfo.myRank, 2, &processInfo.myCoords) ;

    if (processInfo.myRank == ROOT_PROCESS) {
        double **A = allocRandomMatrix(m, k) ;
        for (int i = 0 ; i < m ; i++) {
            for (int j = 0 ; j < k ; j++) {
                A[i][j] = i * k + j ;
            }
        }

        // TODO > Capire se ha senso allocare B per colonne, ma in teoria bisognerebbe trovare un modo di "trasporre" il comunicatore
        // Se "traspongo" (alloco all'inverso) la matrice l'effetto è lo stesso
        // La trasposizione del comunicatore la posso fare invertendo le variabili nei cicli di send
        // Per ora lasciare così, poi si vede che in caso non è difficile sistemare
        double **B = allocRandomMatrix(n, k) ;
        for (int i = 0 ; i < n ; i++) {
            for (int j = 0 ; j < k ; j++) {
                B[j][i] = i * k + j ;
            }
        }

        matrixSend(A, m, k, mb, kb, PROCESS_GRID) ; // SEND A
        matrixSend(B, n, k, nb, kb, PROCESS_GRID) ; // SEND B

        //SEND C solo ai processi root di una REDUCE
        // int procCoords[2] ;
        // for (int j = 0 ; j < PROCESS_GRID[1] ; j++) {
        //     procCoords[0] = 0 ;
        //     procCoords[1] = j ;
        //     int destProc ;
        //     MPI_Cart_rank(CART_COMM, procCoords, &destProc) ;
            
        // }
    }

    int subm ;
    int subk ; 
    int subn ;
    double **subA = matrixRecv(m, k, mb, kb, PROCESS_GRID, &subm, &subk) ;
    double **subB = matrixRecv(n, k, nb, kb, PROCESS_GRID, &subn, &subk) ;
    double **subC = NULL ;
    // TODO IF IS ROOT OF REDUCE --> RECV subC
    double **C ;

    executeCompleteProduct(subA, subB, subC, m, k, n, mb, kb, nb, subm, subk, subn, C) ;

    return 0 ;
}

// Executes C <- A * B sui sottoblocchi
// TODO > Capire se va bene usare delle variabili globali per il rank e per il process grid
void executeCompleteProduct(
    double **subA, double **subB, double **subC,
    int m, int k, int n,
    int mb, int kb, int nb,
    int subm, int subk, int subn,
    double **C
) {

    Content content ;

    MPI_Datatype returnTypes[3][3] ;
    createSendDataTypes(m, n, mb, nb, PROCESS_GRID, returnTypes) ;

    createReduceCommunicator() ;
    
    for (int i = 0 ; i < PROCESS_GRID[0] ; i++) {
        double **cSubProd = allocMatrix(subm, subn) ;

        // Se non ho nulla dentro allora il processo non è una root di una REDUCE --> Init a 0
        // Altrimenti --> Init alla sotto matrice ricevuta per C
        if (subC != NULL) {
            memcpy(&(cSubProd[0][0]), &(subC[0][0]), sizeof(double) * subm * subn) ;
        }
        
        subMatrixProduct(subA, subB, cSubProd, subm, subk, subn) ;
        return ;

        reduceAndSendToRoot(C, cSubProd, m, n, mb, nb, subm, subn, returnTypes, i) ;

        exchangeSubMatrices(&subB, &subC, m, k, n, &subm, &subk, &subn, mb, kb, nb, i) ;

        free(&(cSubProd[0][0])) ;
    }
}

void exchangeSubMatrices(double ***subB, double ***subC, int m, int k, int n, int *submPtr, int *subkPtr, int *subnPtr, int mb, int kb, int nb, int i) {
    int nextRank ;
    int prevRank ;
    MPI_Cart_shift(CART_COMM, 1, i + 1, processInfo.myRank, &nextRank) ;
    MPI_Cart_shift(CART_COMM, 1, -i - 1, processInfo.myRank, &prevRank) ;
    
    int subm = *submPtr ;
    int subk = *subkPtr ;
    int subn = *subnPtr ;

    int procCoords[2] ;
    MPI_Cart_coords(CART_COMM, prevRank, 2, procCoords) ;
    int newSubm, newSubk, newSubn ;
    computeSubMatrixDimsPerProc(procCoords[0], procCoords[1], PROCESS_GRID, n, k, nb, kb, &newSubn, &newSubk) ;

    double **newSubB = allocMatrix(newSubn, newSubk) ;
    MPI_Sendrecv(
        &(subB[0][0]), subk * subn, TYPE_MATRIX_NUM, nextRank, SEND_TAG,
        &(newSubB[0][0]), newSubk * newSubn, TYPE_MATRIX_NUM, prevRank, SEND_TAG,
        CART_COMM, MPI_STATUS_IGNORE
    ) ;

    MPI_Cart_coords(CART_COMM, prevRank, 2, procCoords) ;
    computeSubMatrixDimsPerProc(procCoords[0], procCoords[1], PROCESS_GRID, m, n, mb, nb, &newSubm, &newSubn) ;
    double **newSubC = allocMatrix(newSubm, newSubk) ;
    MPI_Sendrecv(
        &(subC[0][0]), subm * subn, TYPE_MATRIX_NUM, nextRank, SEND_TAG,
        &(newSubC[0][0]), newSubm * newSubn, TYPE_MATRIX_NUM, prevRank, SEND_TAG,
        CART_COMM, MPI_STATUS_IGNORE
    ) ;

    free(&(subB[0][0])) ;
    free(&(subC[0][0])) ;

    *submPtr = newSubm ;
    *subkPtr = newSubk ;
    *subnPtr = newSubn ;

    *subB = newSubB ;
    *subC = newSubC ;
}

void reduceAndSendToRoot(double **C, double **subC, int m, int n, int mb, int nb, int subm, int subn, MPI_Datatype returnTypes[3][3], int i) {
    //MPI_Reduce(&(subC[0][0]), &(subC[0][0]), subm * subn, TYPE_MATRIX_NUM, MPI_SUM, 0, REDUCE_COMM) ;

    if (processInfo.myCoords[1] == 0) {
        // Significa che è root di un comunicatore di REDUCE
        // Devo mandare quello che ho calcolato alla ROOT
        MPI_Send(&(subC[0][0]), subm * subn, TYPE_MATRIX_NUM, ROOT_PROCESS, SEND_TAG, CART_COMM) ;
    }

    if (processInfo.myRank == ROOT_PROCESS) {
        for (int proc = 0 ; proc < PROCESS_GRID[0] ; proc++) {
            int senderProcCoords[2] ;
            MPI_Datatype returnType ;
            if (i == 0) {
                MPI_Cart_coords(CART_COMM, proc, 2, senderProcCoords) ;
            } else {
                int prevRank ;
                MPI_Cart_shift(CART_COMM, 1, -i - 1, processInfo.myRank, &prevRank) ;
                MPI_Cart_coords(CART_COMM, prevRank, 2, senderProcCoords) ;
            }
            returnType = computeSendTypeForProc(senderProcCoords[0], senderProcCoords[1], PROCESS_GRID, m, n, mb, nb, returnTypes) ;
            
            MPI_Recv(&(C[senderProcCoords[0] * mb][i * nb]), 1, returnType, proc, SEND_TAG, CART_COMM, MPI_STATUS_IGNORE) ;
        }
    }
}

void createReduceCommunicator() {
    REDUCE_COMM_ARRAY = (MPI_Comm *) malloc(sizeof(MPI_Comm) * PROCESS_GRID[0]) ;

    MPI_Group cartGroup ;
    MPI_Comm_group(CART_COMM, &cartGroup) ;
    int newGroupRanks[PROCESS_GRID[1]] ;
    int procCoords[2] ;
    for (int i = 0 ; i < PROCESS_GRID[0] ; i++) {
        for (int j = 0 ; j < PROCESS_GRID[1] ; j++) {
            procCoords[0] = i ;
            procCoords[1] = j ;
            MPI_Cart_rank(CART_COMM, procCoords, &newGroupRanks[j]) ;
        }

        MPI_Group newGroup ;
        MPI_Group_incl(cartGroup, PROCESS_GRID[1], newGroupRanks, &newGroup) ;
        MPI_Comm newComm ;
        MPI_Comm_create(CART_COMM, newGroup, &REDUCE_COMM_ARRAY[i]) ;
    }
}

void subMatrixProduct(double **subA, double **subB, double **subC, int subm, int subk, int subn) {
    for (int i = 0 ; i < subm ; i++) {
        for (int j = 0 ; j < subn ; j++) {
            for (int t = 0 ; t < subk ; t++) {
                //printf("A , B = %f , %f\n", subA[i][t], subB[j][t]) ;
                subC[i][j] += subA[i][t] * subB[j][t] ;
            }
            //printf("Value %f\n", subC[i][j]) ;
        }
    }
}


double **matrixRecv(int rows, int cols, int blockRows, int blockCols, int *processGrid, int *subRowsPtr, int *subColsPtr) {
    int subMatRows, subMatCols ;
    int procCoords[2] ;
    MPI_Cart_coords(CART_COMM, processInfo.myRank, 2, procCoords) ;

    computeSubMatrixDimsPerProc(procCoords[0], procCoords[1], processGrid, rows, cols, blockRows, blockCols, &subMatRows, &subMatCols) ;
    double **subMat = allocMatrix(subMatRows, subMatCols) ;
    MPI_Recv(&(subMat[0][0]), subMatRows * subMatCols, TYPE_MATRIX_NUM, ROOT_PROCESS, SEND_TAG, WORLD_COMM, MPI_STATUS_IGNORE) ;
    
    Content content ;
    content.matrix = subMat ;
    //printMessage("SUB MATRIX: ", content, MATRIX, processInfo.myRank, 0, subMatRows, subMatCols, 1) ;

    *subRowsPtr = subMatRows ;
    *subColsPtr = subMatCols ;
    return subMat ;
}

void matrixSend(double **matrix, int rows, int cols, int blockRows, int blockCols, int *processGrid) {
    MPI_Datatype matrixTypes[3][3] ;
    createSendDataTypes(rows, cols, blockRows, blockCols, processGrid, matrixTypes) ;

    for (int procRow = 0 ; procRow < processGrid[0] ; procRow++) {
        for (int procCol = 0 ; procCol < processGrid[1] ; procCol++) {
            int coords[2] = {procRow, procCol} ;
            int proc ;
            MPI_Cart_rank(CART_COMM, coords, &proc) ;

            MPI_Datatype sendType = computeSendTypeForProc(procRow, procCol, processGrid, rows, cols, blockRows, blockCols, matrixTypes) ;
            MPI_Request mpiRequest ;

            // Bisogna fare ISend per l'invio da zero a se stesso, altrimenti ho un blocco
            // TODO > Controlla se va bene oppure se è meglio usare la SendRecv invece
            MPI_Isend(&(matrix[procRow * blockRows][procCol * blockCols]), 1, sendType, proc, SEND_TAG, WORLD_COMM, &mpiRequest) ;
        }
    }
}

MPI_Datatype computeSendTypeForProc(
    int rowRank, int colRank, int *processGrid,
    int rowsNum, int colsNum, 
    int blockRows, int blockCols,  
    MPI_Datatype typesMatrix[3][3]
) {
    int timesGridInRows = rowsNum / (processGrid[0] * blockRows) ;
    int timesGridInCols = colsNum / (processGrid[1] * blockCols) ;

    int finalRowsNum = rowsNum % (processGrid[0] * blockRows) ;
    int finalColsNum = colsNum % (processGrid[1] * blockCols) ;

    int timesBlockInFinalRows = finalRowsNum / blockRows ;
    int timesBlockInFinalCols = finalColsNum / blockCols ;

    int typeIndexRow ;
    if (rowRank % processGrid[0] < timesBlockInFinalRows) {
        typeIndexRow = 0 ;
    } else if (rowRank % processGrid[0] == timesBlockInFinalRows) {
        typeIndexRow = 1 ;
    } else {
        typeIndexRow = 2 ;
    }

    int typeIndexCol ;
    if (colRank % processGrid[1] < timesBlockInFinalCols) {
        typeIndexCol = 0 ;
    } else if (colRank % processGrid[1] == timesBlockInFinalCols) {
        typeIndexCol = 1 ;
    } else {
        typeIndexCol = 2 ;
    }

    return typesMatrix[typeIndexRow][typeIndexCol] ;
}

void createSendDataTypes(int rowsNum, int colsNum, int blockRows, int blockCols, int *processGrid, MPI_Datatype typesMatrix[3][3]) {
    Content content ;

    int timesGridInRows = rowsNum / (processGrid[0] * blockRows) ;
    int timesGridInCols = colsNum / (processGrid[1] * blockCols) ;

    int finalRowsNum = rowsNum % (processGrid[0] * blockRows) ;
    int finalColsNum = colsNum % (processGrid[1] * blockCols) ;

    int timesBlockInFinalRows = finalRowsNum / blockRows ;
    int timesBlockInFinalCols = finalColsNum / blockCols ;

    MPI_Datatype indexedTypes[3] ;

    int blockLengths[timesGridInCols + 1] ;
    int displs[timesGridInCols + 1] ;
    for (int i = 0 ; i < timesGridInCols + 1 ; i++) {
        blockLengths[i] = blockCols ;
        displs[i] = i * processGrid[1] * blockCols ;
    }
    MPI_Type_indexed(timesGridInCols + 1, blockLengths, displs, TYPE_MATRIX_NUM, &indexedTypes[0]) ;
    
    blockLengths[timesGridInCols] = finalColsNum % blockCols ;
    MPI_Type_indexed(timesGridInCols + 1, blockLengths, displs, TYPE_MATRIX_NUM, &indexedTypes[1]) ;

    MPI_Type_indexed(timesGridInCols, blockLengths, displs, TYPE_MATRIX_NUM, &indexedTypes[2]) ;

    int count = blockRows * (timesGridInRows + 1) ;
    MPI_Aint displs_2[count] ;
    int currDispl = 0 ;
    for (int i = 0 ; i < count ; i++) {
        if (i % blockRows == 0) {
            currDispl = (i / blockRows) * blockRows * processGrid[0] * colsNum ; 
        } else {
            currDispl += colsNum ;
        }
        displs_2[i] = currDispl * sizeof(double) ;
    }

    for (int i = 0 ; i < 3 ; i++) {
        int currCount ;
        if (i == 0) {
            currCount = blockRows * (timesGridInRows + 1) ;
        } else if (i == 1) {
            currCount = blockRows * timesGridInRows + finalRowsNum % blockRows ;
        } else {
            currCount = blockRows * timesGridInRows ;
        }

        for (int j = 0 ; j < 3 ; j++) {
            MPI_Type_create_hindexed_block(currCount, 1, displs_2, indexedTypes[j], &typesMatrix[i][j]) ;
            MPI_Type_commit(&typesMatrix[i][j]) ;
        }
    }
 
}


void computeSubMatrixDimsPerProc(
    int rowRank, int colRank, int *processGrid, 
    int rowsNum, int colsNum, 
    int blockRows, int blockCols, 
    int *subMatRowsNum, int *subMatColsNum
) {

    int timesGridInRows = rowsNum / (processGrid[0] * blockRows) ;
    int timesGridInCols = colsNum / (processGrid[1] * blockCols) ;

    int finalRowsNum = rowsNum % (processGrid[0] * blockRows) ;
    int finalColsNum = colsNum % (processGrid[1] * blockCols) ;

    int timesBlockInFinalRows = finalRowsNum / blockRows ;
    int timesBlockInFinalCols = finalColsNum / blockCols ;

    //printf("Times %d %d\n", timesBlockInFinalRows, timesBlockInFinalCols) ;
    int residualRecvdRows ;
    if (rowRank % processGrid[0] < timesBlockInFinalRows) {
        residualRecvdRows = blockRows ;
    } else if (rowRank % processGrid[0] == timesBlockInFinalRows) {
        residualRecvdRows = finalRowsNum % blockRows ;
    } else {
        residualRecvdRows = 0 ;
    }

    int residualRecvdCols ;
    if (colRank % processGrid[1] < timesBlockInFinalCols) {
        residualRecvdCols = blockCols ;
    } else if (colRank % processGrid[1] == timesBlockInFinalCols) {
        residualRecvdCols = finalColsNum % blockCols ;
    } else {
        residualRecvdCols = 0 ;
    }

    *subMatRowsNum = blockRows * timesGridInRows + residualRecvdRows ;
    *subMatColsNum = blockCols * timesGridInCols + residualRecvdCols ;
}


int extractParams(int argc, char *argv[], int *mPtr, int *kPtr, int *nPtr, int *mbPtr, int *kbPtr, int *nbPtr) {
    
    // if (argc < 6) {
    //     return 0 ;
    // }
    
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
        if (strcmp(argv[i], "-mb") == 0) {
            *mbPtr = atoi(argv[i+1]) ;
        }
        if (strcmp(argv[i], "-kb") == 0) {
            *kbPtr = atoi(argv[i+1]) ;
        }
        if (strcmp(argv[i], "-nb") == 0) {
            *nbPtr = atoi(argv[i+1]) ;
        }
    }

    if (*mPtr <= 0 || *kPtr <= 0 || *nPtr <= 0 || *mbPtr <= 0 || *kbPtr <= 0 || *nbPtr <= 0) {
        return 0 ;
    }

    return 1 ;
}