#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <time.h>
#include <string.h>
#include <math.h>

#include "../Matrix/Matrix.h"
#include "SendRecvUtils.h"
#include "../Utils/PrintUtils.h"

#define PROCESS_GRID_DIMS 2
#define TYPE_MATRIX_NUM MPI_FLOAT
#define ROOT_PROCESS 0
#define SEND_TAG 100

#define ROW_BLOCK_SIZE 100
#define COL_BLOCK_SIZE 100

MPI_Comm WORLD_COMM ;
MPI_Comm CART_COMM ;
MPI_Comm *REDUCE_COMM_ARRAY ;

typedef struct ProcessInfo {
    int myRank ;
    int myCoords[2] ;
} ProcessInfo ;

ProcessInfo processInfo ;
int PROCESS_GRID[PROCESS_GRID_DIMS] = {0} ;

void executeCompleteProduct(
    float **subA, float **subB, float **subC,
    int m, int k, int n,
    int mb, int kb, int nb,
    int subm, int subk, int subn,
    float **C
) ;

float **matrixSendToAll(float **matrix, int rows, int cols, int blockRows, int blockCols, int *processGrid, int invertGrid, int perGroupOfRows, int *subMatRows, int *subMatCols) ;
float **matrixRecvFromRoot(int rows, int cols, int blockRows, int blockCols, int *processGrid, int *subRowsPtr, int *subColsPtr, int invertGrid, int perGroupOfRows) ;

void createReduceCommunicator() ;
void freeReduceCommunicator() ;
void subMatrixProduct(float **subA, float **subB, float **cSubProd, int subm, int subk, int subn) ;


void MpiProduct(float **A, float **B, float **C, int m, int k, int n) {
    int mb = ROW_BLOCK_SIZE ;
    int kb = COL_BLOCK_SIZE ;
    int nb = n ;

    MPI_Comm_dup(MPI_COMM_WORLD, &WORLD_COMM);

    int procNum ;
    MPI_Comm_rank(WORLD_COMM, &processInfo.myRank);
    MPI_Comm_size(WORLD_COMM, &procNum);

    //int PROCESS_GRID[PROCESS_GRID_DIMS] = {0} ;
    MPI_Dims_create(procNum, PROCESS_GRID_DIMS, PROCESS_GRID) ;
    // if (processInfo.myRank == ROOT_PROCESS) {
    //     printf("Cartesian Grid: [%d, %d]\n", PROCESS_GRID[0], PROCESS_GRID[1]) ;
    // }
    int periods[2] = {0, 0} ;
    MPI_Cart_create(WORLD_COMM, 2, PROCESS_GRID, periods, 0, &CART_COMM) ;
    MPI_Cart_coords(CART_COMM, processInfo.myRank, 2, &processInfo.myCoords) ;

    createReduceCommunicator() ;

    int invertedGrid[2] ;
    invertedGrid[0] = PROCESS_GRID[1] ;
    invertedGrid[1] = PROCESS_GRID[0] ;

    int scatterCGrid[2] ;
    scatterCGrid[0] = PROCESS_GRID[0] ;
    scatterCGrid[1] = 1 ;
    float **subA, **subB, **subC ;
    int subm, subk, subn ;
    if (processInfo.myRank == ROOT_PROCESS) {
        subA = matrixSendToAll(A, m, k, mb, kb, PROCESS_GRID, 0, 0, &subm, &subk) ; // SEND A
        subB = matrixSendToAll(B, k, n, kb, nb, invertedGrid, 1, 1, &subk, &subn) ; // SEND B
        subC = matrixSendToAll(C, m, n, mb, nb, scatterCGrid, 0, 1, &subm, &subn) ; // SEND C
    
    } else {
        subA = matrixRecvFromRoot(m, k, mb, kb, PROCESS_GRID, &subm, &subk, 0, 0) ;
        subB =  matrixRecvFromRoot(k, n, kb, nb, invertedGrid, &subk, &subn, 1, 1) ;
        if (processInfo.myCoords[1] == 0) {
            subC = matrixRecvFromRoot(m, n, mb, nb, scatterCGrid, &subm, &subn, 0, 1) ;
        } else {
            subC = allocMatrix(subm, subn) ;
        }
    }
    
    executeCompleteProduct(subA, subB, subC, m, k, n, mb, kb, nb, subm, subk, subn, C) ;

    if (processInfo.myRank == ROOT_PROCESS) {
        Content content ;
        content.matrix = C ;
        //printMessage("FINAL MATRIX", content, MATRIX, processInfo.myRank, ROOT_PROCESS, m, n, 1) ;
    }

    MPI_Comm_free(&WORLD_COMM) ;
    MPI_Comm_free(&CART_COMM) ;

    freeReduceCommunicator(processInfo.myCoords[0]) ;

    freeMatrix(subA) ;
    freeMatrix(subB) ;
    freeMatrix(subC) ;

    return ;
}


// Executes C <- A * B sui sottoblocchi
// TODO > Capire se va bene usare delle variabili globali per il rank e per il process grid
// TODO > Modificare invio: invio tutti i pezzi che servono al singolo processo; in questo modo può cominciare a lavorare!!
// TODO > Fare deallocazione dei tipi usati per inviare i dati
// TODO > Fare Refactoring del codice
//          - Spostare qualcosa in altri file??
//          - Capire perché MPI_IN_PLACE non funge
void executeCompleteProduct(
    float **subA, float **subB, float **subC,
    int m, int k, int n,
    int mb, int kb, int nb,
    int subm, int subk, int subn,
    float **C
) {

    printf("DOING PRODUCT\n") ;

    Content content ;
    
    subMatrixProduct(subA, subB, subC, subm, subk, subn) ;

    //printf("Process %d; Finished Product\n", processInfo.myRank) ;

    float **reducedC ;
    if (processInfo.myCoords[1] == 0) {
        reducedC = allocMatrix(subm, subn) ;
    }
    
    MPI_Reduce(&(subC[0][0]), &(reducedC[0][0]), subm * subn, TYPE_MATRIX_NUM, MPI_SUM, 0, REDUCE_COMM_ARRAY[processInfo.myCoords[0]]) ;

    if (processInfo.myCoords[1] == 0) {
        printf("Process %d > REDUCED DONE\n", processInfo.myRank) ;
    }

    // IS ONE OF THE REDUCE ROOT
    if (processInfo.myCoords[1] == 0 && processInfo.myRank != ROOT_PROCESS) {
        MPI_Request mpiRequest ;
        MPI_Send(&(reducedC[0][0]), subm * subn, TYPE_MATRIX_NUM, ROOT_PROCESS, SEND_TAG, CART_COMM) ;
        printf("Process %d > SENT TO ROOT\n", processInfo.myRank) ;
    }
   
    if (processInfo.myRank == ROOT_PROCESS) {
        //RECV DEI PEZZI CHE VENGONO MANDATI E INSERIMENTO IN C
        MPI_Datatype returnTypes[3][3] ;
        int gatherGrid[2] = {PROCESS_GRID[0], 1} ;
        createSendDataTypes(m, n, mb, nb, gatherGrid, TYPE_MATRIX_NUM, returnTypes) ;

        for (int procRow = 0 ; procRow < PROCESS_GRID[0] ; procRow++) {
            
            int rowIndex = procRow ;
            int colIndex = 0 ;

            int proc ;
            int coords[2] = {rowIndex, colIndex} ;
            MPI_Cart_rank(CART_COMM, coords, &proc) ;

            MPI_Datatype returnType = computeSendTypeForProc(rowIndex, colIndex, gatherGrid, m, n, mb, nb, returnTypes) ;
            MPI_Request mpiRequest ;

            if (proc != processInfo.myRank) {
                MPI_Recv(&(C[rowIndex * mb][colIndex * nb]), 1, returnType, proc, SEND_TAG, CART_COMM, MPI_STATUS_IGNORE) ;
            } else {
                MPI_Sendrecv(
                    &(reducedC[0][0]), subm * subn, TYPE_MATRIX_NUM, processInfo.myRank, SEND_TAG, 
                    &(C[rowIndex * mb][colIndex * nb]), 1, returnType, proc, SEND_TAG,
                    CART_COMM, MPI_STATUS_IGNORE) ;
                printf("Process ROOT > Done SendRecv\n") ;
            }
        }

        freeSendDataTypes(returnTypes) ;
    }

    if (processInfo.myCoords[1] == 0) {
        freeMatrix(reducedC) ;
    }
    
}

void freeReduceCommunicator(int procRow) {
    for (int i = 0 ; i < PROCESS_GRID[0]; i++) {
        if (procRow == i) {
            MPI_Comm_free(&REDUCE_COMM_ARRAY[i]) ;
        }

    }
    free(REDUCE_COMM_ARRAY) ;
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

        Content content ;
        content.intArray = newGroupRanks ;
        printMessage("NEW GROUP", content, INT_ARRAY, processInfo.myRank, ROOT_PROCESS, PROCESS_GRID[1], 0, 1) ;

        MPI_Group newGroup ;
        MPI_Group_incl(cartGroup, PROCESS_GRID[1], newGroupRanks, &newGroup) ;
        MPI_Comm newComm ;
        MPI_Comm_create(CART_COMM, newGroup, &REDUCE_COMM_ARRAY[i]) ;
    }
}

void subMatrixProduct(float **subA, float **subB, float **subC, int subm, int subk, int subn) {
    for (int i = 0 ; i < subm ; i++) {
        for (int t = 0 ; t < subk ; t++) {
            for (int j = 0 ; j < subn ; j++) {
                subC[i][j] += subA[i][t] * subB[t][j] ;
            }
        }
    }
}


float **matrixRecvFromRoot(int rows, int cols, int blockRows, int blockCols, int *processGrid, int *subRowsPtr, int *subColsPtr, int invertGrid, int perGroupOfRows) {
    int subMatRows, subMatCols ;
    int procCoords[2] ;
    MPI_Cart_coords(CART_COMM, processInfo.myRank, 2, procCoords) ;

    printf("Process %d > WAITING DATA\n", processInfo.myRank) ;
    int rowIndex = procCoords[0] ;
    int colIndex = procCoords[1] ;

    if (invertGrid) {
        rowIndex = procCoords[1] ;
        colIndex = procCoords[0] ;
    }

    if (perGroupOfRows) {
        colIndex = 0 ;
    }

    computeSubMatrixDimsPerProc(rowIndex, colIndex, processGrid, rows, cols, blockRows, blockCols, &subMatRows, &subMatCols) ;
    float **subMat = allocMatrix(subMatRows, subMatCols) ;
    MPI_Recv(&(subMat[0][0]), subMatRows * subMatCols, TYPE_MATRIX_NUM, ROOT_PROCESS, SEND_TAG, CART_COMM, MPI_STATUS_IGNORE) ;
    
    Content content ;
    content.matrix = subMat ;
    // printMessage("SUB MATRIX: ", content, MATRIX, processInfo.myRank, 0, subMatRows, subMatCols, 1) ;

    *subRowsPtr = subMatRows ;
    *subColsPtr = subMatCols ;
    return subMat ;
}

float **matrixSendToAll(
    float **matrix, 
    int rows, int cols, 
    int blockRows, int blockCols, 
    int *processGrid, 
    int invertGrid, int perGroupOfRows,
    int *subMatRows, int *subMatCols
) {

    float **subMat ;
    MPI_Datatype matrixTypes[3][3] ;
    createSendDataTypes(rows, cols, blockRows, blockCols, processGrid, TYPE_MATRIX_NUM, matrixTypes) ;

    for (int procRow = 0 ; procRow < processGrid[0] ; procRow++) {
        for (int procCol = 0 ; procCol < processGrid[1] ; procCol++) {
            int coords[2] = {procRow, procCol} ;
            if (invertGrid) {
                coords[0] = procCol ;
                coords[1] = procRow ;
            }
            int proc ;
            MPI_Cart_rank(CART_COMM, coords, &proc) ;

            int rowIndex = procRow ;
            int colIndex = procCol ;
            if (perGroupOfRows) {
                rowIndex = procRow % processGrid[0] ;
                colIndex = 0 ;
            }

            MPI_Datatype sendType = computeSendTypeForProc(rowIndex, colIndex, processGrid, rows, cols, blockRows, blockCols, matrixTypes) ;
            MPI_Request mpiRequest ;

            if (proc != ROOT_PROCESS) {
                MPI_Send(&(matrix[rowIndex * blockRows][colIndex * blockCols]), 1, sendType, proc, SEND_TAG, CART_COMM) ;
            } else {
                computeSubMatrixDimsPerProc(rowIndex, colIndex, processGrid, rows, cols, blockRows, blockCols, subMatRows, subMatCols) ;
                subMat = allocMatrix(*subMatRows, *subMatCols) ;

                MPI_Sendrecv(
                    &(matrix[rowIndex * blockRows][colIndex * blockCols]), 1, sendType, proc, SEND_TAG,
                    &(subMat[0][0]), (*subMatRows) * (*subMatCols), TYPE_MATRIX_NUM, ROOT_PROCESS, SEND_TAG, CART_COMM, MPI_STATUS_IGNORE) ;
            }
            //MPI_Request_free(&mpiRequest) ;
        }
    }

    printf("TERMINATE SEND\n") ;

    freeSendDataTypes(matrixTypes) ;

    return subMat ;
}

