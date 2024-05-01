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


int procRank ;
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
    double **subA, double **subB,
    int m, int k, int n,
    int mb, int kb, int nb,
    int subm, int subk, int subn
) ;

void matrixSend(double **matrix, int rows, int cols, int blockRows, int blockCols, int *processGrid) ;
double **matrixRecv(int rows, int cols, int blockRows, int blockCols, int *processGrid) ;



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

    MPI_Init(&argc, &argv);
    MPI_Comm_dup(MPI_COMM_WORLD, &WORLD_COMM);

    int myRank ;
    int procNum ;
    MPI_Comm_rank(WORLD_COMM, &myRank);
    MPI_Comm_size(WORLD_COMM, &procNum);
    procRank = myRank ;

    //int PROCESS_GRID[PROCESS_GRID_DIMS] = {0} ;
    MPI_Dims_create(procNum, PROCESS_GRID_DIMS, PROCESS_GRID) ;
    if (procRank == ROOT_PROCESS) {
        printf("Cartesian Grid: [%d, %d]\n", PROCESS_GRID[0], PROCESS_GRID[1]) ;
    }
    int periods[2] = {0, 0} ;
    MPI_Cart_create(WORLD_COMM, 2, PROCESS_GRID, periods, 0, &CART_COMM) ;

    if (myRank == ROOT_PROCESS) {
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
    }

    int subm ;
    int subk ; 
    int subn ;
    double **subA = matrixRecv(m, k, mb, kb, PROCESS_GRID) ;
    double **subB = matrixRecv(n, k, nb, kb, PROCESS_GRID) ;

    executeCompleteProduct(subA, subB, m, k, n, mb, kb, nb, subm, subk, subn) ;

    return 0 ;
}

// Executes C <- A * B sui sottoblocchi
// Capire se va bene usare delle variabili globali per il rank e per il process grid
void executeCompleteProduct(
    double **subA, double **subB,
    int m, int k, int n,
    int mb, int kb, int nb,
    int subm, int subk, int subn
) {

    for (int i = 0 ; i < PROCESS_GRID[1] ; i++) {
        double **subC = allocMatrix(subm, subn) ;
        subMatrixProduct(subA, subB, subC, subm, subk, subn) ;


        free(&(subC[0][0])) ;
    }
    


    //TODO  > Ciclo sullo scambio di blocchetti
    //TODO  > Faccio prodotto tra sotto blocchi
    //TODO  > Faccio un comunicatore sulle COLONNE della griglia
    //TODO  > Faccio la REDUCE delle sottomatrici prodotte e ottengo la sottomatrice per C.
    //          La GATHER la faccio con radice il sottoblocco di C
    //TODO  > Mi serve anche la GATHER da qualche parte probabilmente per ricostruire il tutto
    //TODO  > Mi conviene inviare una porzione di C anche alle root dei comunicatori di colonna 
    //      > in modo che anche la somma C + A*B sia distribuita allo stesso modo
    //TODO  > Scambio la sottomatrice B tra processi (MPI_CART_SHIFT + MPI_SENDRECV)
    //TODO  > Riunisco tutto quanto insieme
}

double **matrixRecv(int rows, int cols, int blockRows, int blockCols, int *processGrid) {
    int subMatRows, subMatCols ;
    int procCoords[2] ;
    MPI_Cart_coords(CART_COMM, procRank, 2, procCoords) ;

    computeSubMatrixDimsPerProc(procCoords[0], procCoords[1], processGrid, rows, cols, blockRows, blockCols, &subMatRows, &subMatCols) ;
    double **subMat = allocMatrix(subMatRows, subMatCols) ;
    MPI_Recv(&(subMat[0][0]), subMatRows * subMatCols, TYPE_MATRIX_NUM, ROOT_PROCESS, SEND_TAG, WORLD_COMM, MPI_STATUS_IGNORE) ;
    
    Content content ;
    content.matrix = subMat ;
    printMessage("SUB MATRIX: ", content, MATRIX, procRank, 0, subMatRows, subMatCols, 1) ;

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
            MPI_Send(&(matrix[procRow * blockRows][procCol * blockCols]), 1, sendType, proc, SEND_TAG, WORLD_COMM) ;
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