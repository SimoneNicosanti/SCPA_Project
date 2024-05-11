#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <time.h>
#include <string.h>
#include <math.h>

#include "../Matrix/Matrix.h"
#include "SendRecvUtils.h"
#include "../Utils/PrintUtils.h"

#include "../Sequential/Sequential.h"
#include "../OpenMp/OpenMpProduct.h"

#define PROCESS_GRID_DIMS 2
#define TYPE_MATRIX_NUM MPI_FLOAT
#define ROOT_PROCESS 0
#define SEND_TAG 100

#define ROW_BLOCK_SIZE 52
#define COL_BLOCK_SIZE 52

MPI_Comm WORLD_COMM ;
MPI_Comm CART_COMM ;
MPI_Comm *REDUCE_COMM_ARRAY ;

typedef struct ProcessInfo {
    int myRank ;
    int myCoords[2] ;
} ProcessInfo ;

ProcessInfo processInfo ;
int PROCESS_GRID[PROCESS_GRID_DIMS] = {0} ;

void scatterMatrix(
    float **matrix, 
    int rows, int cols, 
    int blockRows, int blockCols, 
    int *processGrid,
    int perGroupOfRows, int perGroupOfCols,
    float ***subMatrix, int *subRowsPtr, int *subColsPtr
) ;
float **matrixSendToAll(
    float **matrix, 
    int rows, int cols, int blockRows, int blockCols, 
    int *processGrid, 
    int perGroupOfRows, int perGroupOfCols, 
    int *subMatRows, int *subMatCols
) ;
float **matrixRecvFromRoot(
    int rows, int cols, int blockRows, int blockCols, 
    int *processGrid, 
    int perGroupOfRows, int perGroupOfCols,
    int *subRowsPtr, int *subColsPtr
) ;
void gatherFinalMatrix(
    float **subC,
    int m, int n,
    int mb, int nb,
    int subm, int subn,
    float **C
) ;


void MpiProduct(float **A, float **B, float **C, int m, int k, int n, int blockRows, int blockCols) {
    int mb = ROW_BLOCK_SIZE ;
    int nb = COL_BLOCK_SIZE ;
    if (blockRows > 0) {
        mb = blockRows ;
    }
    if (blockCols > 0) {
        nb = blockCols ;
    }
    
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

    float **subA, **subB, **subC ;
    int subm, subk, subn ;
    
    scatterMatrix(A, m, k, mb, k, PROCESS_GRID, 1, 0, &subA, &subm, &subk) ;
    scatterMatrix(B, k, n, k, nb, PROCESS_GRID, 0, 1, &subB, &subk, &subn) ;
    scatterMatrix(C, m, n, mb, nb, PROCESS_GRID, 0, 0, &subC, &subm, &subn) ;
    
    #ifdef OPEN_MP
        openMpProduct(subA, subB, subC, subm, subk, subn) ;
    #else
        matrixProduct(subA, subB, subC, subm, subk, subn) ;
    #endif 


    gatherFinalMatrix(subC, m, n, mb, nb, subm, subn, C) ;

    MPI_Comm_free(&WORLD_COMM) ;
    MPI_Comm_free(&CART_COMM) ;

    freeMatrix(subA, subm, subk) ;
    freeMatrix(subB, subk, subn) ;
    freeMatrix(subC, subm, subn) ;

    return ;
}

void scatterMatrix(
    float **matrix, 
    int rows, int cols, 
    int blockRows, int blockCols, 
    int *processGrid,
    int perGroupOfRows, int perGroupOfCols,
    float ***subMatrix, int *subRowsPtr, int *subColsPtr
) {
    float **returnMatrix ;
    if (processInfo.myRank == ROOT_PROCESS) {
        returnMatrix = matrixSendToAll(matrix, rows, cols, blockRows, blockCols, processGrid, perGroupOfRows, perGroupOfCols, subRowsPtr, subColsPtr) ;
    } else {
        returnMatrix = matrixRecvFromRoot(rows, cols, blockRows, blockCols, processGrid, perGroupOfRows, perGroupOfCols, subRowsPtr, subColsPtr) ;
    }

    *subMatrix = returnMatrix ;
}


void gatherFinalMatrix(
    float **subMatrix,
    int rows, int cols,
    int blockRows, int blockCols,
    int subMatRows, int subMatCols,
    float **matrix
) {

    if (processInfo.myRank != ROOT_PROCESS) {
        MPI_Send(&(subMatrix[0][0]), subMatRows * subMatCols, TYPE_MATRIX_NUM, ROOT_PROCESS, SEND_TAG, CART_COMM) ;
    }
   
    if (processInfo.myRank == ROOT_PROCESS) {
        float start = MPI_Wtime() ;
        //RECV DEI PEZZI CHE VENGONO MANDATI E INSERIMENTO IN C
        MPI_Datatype returnTypes[3][3] ;
        int gatherGrid[2] = {PROCESS_GRID[0], PROCESS_GRID[1]} ;
        createSendDataTypes(rows, cols, blockRows, blockCols, gatherGrid, TYPE_MATRIX_NUM, returnTypes) ;

        for (int procRow = 0 ; procRow < PROCESS_GRID[0] ; procRow++) {
            for (int procCol = 0 ; procCol < PROCESS_GRID[1] ; procCol++) {
                int rowIndex = procRow ;
                int colIndex = procCol ;

                int proc ;
                int coords[2] = {rowIndex, colIndex} ;
                MPI_Cart_rank(CART_COMM, coords, &proc) ;

                MPI_Datatype returnType = computeSendTypeForProc(rowIndex, colIndex, gatherGrid, rows, cols, blockRows, blockCols, returnTypes) ;
                MPI_Request mpiRequest ;

                if (proc != processInfo.myRank) {
                    MPI_Recv(&(matrix[rowIndex * blockRows][colIndex * blockCols]), 1, returnType, proc, SEND_TAG, CART_COMM, MPI_STATUS_IGNORE) ;
                } else {
                    MPI_Sendrecv(
                        &(subMatrix[0][0]), subMatRows * subMatCols, TYPE_MATRIX_NUM, processInfo.myRank, SEND_TAG, 
                        &(matrix[rowIndex * blockRows][colIndex * blockCols]), 1, returnType, proc, SEND_TAG,
                        CART_COMM, MPI_STATUS_IGNORE) ;
                    // printf("Process ROOT > Done SendRecv\n") ;
                }
            }
        }
        float end = MPI_Wtime() ;
        // printf("TIME GATHER > %f\n", end - start) ;
        freeSendDataTypes(returnTypes) ;
        // printf("COMPLETED PRODUCT\n") ;
    }
}


float **matrixRecvFromRoot(
    int rows, int cols, 
    int blockRows, int blockCols, 
    int *processGrid, 
    int perGroupOfRows, int perGroupOfCols,
    int *subRowsPtr, int *subColsPtr
) {
    int subMatRows, subMatCols ;
    int procCoords[2] ;
    MPI_Cart_coords(CART_COMM, processInfo.myRank, 2, procCoords) ;
    // if (processInfo.myRank == 2) {
    //     printf("Coords > %d %d\n", procCoords[0], procCoords[1]) ;
    // }

    //printf("Process %d > WAITING DATA\n", processInfo.myRank) ;
    int rowIndex = procCoords[0] ;
    int colIndex = procCoords[1] ;

    // if (processInfo.myRank == 2) {
    //     printf("Indexes > %d %d\n", rowIndex, colIndex) ;
    // }

    if (perGroupOfRows) {
        colIndex = 0 ;
    }
    if (perGroupOfCols) {
        rowIndex = 0 ;
    }

    // if (processInfo.myRank == 2) {
    //     printf("Indexes > %d %d\n", rowIndex, colIndex) ;
    // }
    computeSubMatrixDimsPerProc(rowIndex, colIndex, processGrid, rows, cols, blockRows, blockCols, &subMatRows, &subMatCols) ;
    float **subMat = allocMatrix(subMatRows, subMatCols) ;
    MPI_Recv(&(subMat[0][0]), subMatRows * subMatCols, TYPE_MATRIX_NUM, ROOT_PROCESS, SEND_TAG, CART_COMM, MPI_STATUS_IGNORE) ;
    
    Content content ;
    content.matrix = subMat ;

    *subRowsPtr = subMatRows ;
    *subColsPtr = subMatCols ;
    return subMat ;
}

float **matrixSendToAll(
    float **matrix, 
    int rows, int cols, 
    int blockRows, int blockCols, 
    int *processGrid, 
    int perGroupOfRows, int perGroupOfCols,
    int *subMatRows, int *subMatCols
) {

    float **subMat = NULL ;
    MPI_Datatype matrixTypes[3][3] ;
    createSendDataTypes(rows, cols, blockRows, blockCols, processGrid, TYPE_MATRIX_NUM, matrixTypes) ;

    for (int procRow = 0 ; procRow < processGrid[0] ; procRow++) {
        for (int procCol = 0 ; procCol < processGrid[1] ; procCol++) {
            int coords[2] = {procRow, procCol} ;
            // if (invertGrid) {
            //     coords[0] = procCol ;
            //     coords[1] = procRow ;
            // }
            int proc ;
            MPI_Cart_rank(CART_COMM, coords, &proc) ;

            int rowIndex = procRow ;
            int colIndex = procCol ;
            if (perGroupOfRows) {
                rowIndex = procRow % processGrid[0] ;
                colIndex = 0 ;
            }
            if (perGroupOfCols) {
                rowIndex = 0 ;
                colIndex = procCol % processGrid[1] ;
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
        }
    }

    freeSendDataTypes(matrixTypes) ;

    return subMat ;
}

