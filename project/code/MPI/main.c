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

MPI_Comm WORLD_COMM ;
MPI_Comm CART_COMM ;

MPI_Datatype TYPE_COLS_VECTOR ;
MPI_Datatype TYPE_SEQ_VECTORS ;
MPI_Datatype TYPE_SEQ_SEQ ;

MPI_Datatype RECV_SUBARRAYS[3] ;

int procRank ;

int extractParams(int argc, char *argv[], int *mPtr, int *kPtr, int *nPtr, int *mbPtr, int *kbPtr, int *nbPtr) ;
void createSendDataTypes(int rowsNum, int colsNum, int blockRows, int blockCols, int *processGrid, MPI_Datatype typesMatrix[3][3]) ;
MPI_Datatype computeSendTypeForProc(
    int rank, 
    int rowsNum, int colsNum, 
    int blockRows, int blockCols, 
    int *processGrid, MPI_Datatype typesMatrix[3][3]) ;
void computeSubMatrixDimsPerProc(
    int procRank, int *processGrid, 
    int rowsNum, int colsNum, 
    int blockRows, int blockCols, 
    int *subMatRowsNum, int *subMatColsNum
) ;


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

    int processGrid[PROCESS_GRID_DIMS] = {0} ;
    MPI_Dims_create(procNum, PROCESS_GRID_DIMS, processGrid) ;
    if (procRank == ROOT_PROCESS) {
        printf("Cartesian Grid: [%d, %d]\n", processGrid[0], processGrid[1]) ;
    }
    int periods[2] = {0, 0} ;
    MPI_Cart_create(WORLD_COMM, 2, processGrid, periods, 0, &CART_COMM) ;

    int M = 13 ; 
    int N = 13 ;
    int blockRows = 10 ;
    int blockCols = 10 ;
    if (myRank == ROOT_PROCESS) {
        MPI_Datatype matrixTypes[3][3] ;
        createSendDataTypes(M, N, blockRows, blockCols, processGrid, matrixTypes) ;
    
        double **A = allocRandomMatrix(M, N) ;
        for (int i = 0 ; i < M ; i++) {
            for (int j = 0 ; j < N ; j++) {
                A[i][j] = i * N + j ;
            }
        }
        
        for (int procRow = 0 ; procRow < processGrid[0] ; procRow++) {
            for (int procCol = 0 ; procCol < processGrid[0] ; procCol++) {
                int coords[2] = {procRow, procCol} ;
                int proc ;
                MPI_Cart_rank(CART_COMM, coords, &proc) ;

                printf("Process %d %d >> %d\n", procRow, procCol, proc) ;

                MPI_Datatype sendType = computeSendTypeForProc(proc, M, N, blockRows, blockCols, processGrid, matrixTypes) ;
                MPI_Send(&(A[procRow * blockRows][procCol * blockCols]), 1, sendType, proc, 10, WORLD_COMM) ;
            }
        }
    }

    int subMatRows, subMatCols ;
    computeSubMatrixDimsPerProc(myRank, processGrid, M, N, blockRows, blockCols, &subMatRows, &subMatCols) ;
    double **subA = allocMatrix(subMatRows, subMatCols) ;
    MPI_Recv(&(subA[0][0]), subMatRows * subMatCols, TYPE_MATRIX_NUM, ROOT_PROCESS, 10, WORLD_COMM, MPI_STATUS_IGNORE) ;
    Content content ;
    content.matrix = subA ;
    printMessage("SUB A: ", content, MATRIX, procRank, 2, subMatRows, subMatCols, 1) ;

    return 0 ;
}

MPI_Datatype computeSendTypeForProc(int rank, int rowsNum, int colsNum, int blockRows, int blockCols, int *processGrid, MPI_Datatype typesMatrix[3][3]) {
    int timesGridInRows = rowsNum / (processGrid[0] * blockRows) ;
    int timesGridInCols = colsNum / (processGrid[1] * blockCols) ;

    int finalRowsNum = rowsNum % (processGrid[0] * blockRows) ;
    int finalColsNum = colsNum % (processGrid[1] * blockCols) ;

    int timesBlockInFinalRows = finalRowsNum / blockRows ;
    int timesBlockInFinalCols = finalColsNum / blockCols ;

    int typeIndexRow ;
    if (rank % processGrid[0] < timesBlockInFinalRows) {
        typeIndexRow = 0 ;
    } else if (rank % processGrid[0] == timesBlockInFinalRows) {
        typeIndexRow = 1 ;
    } else {
        typeIndexRow = 2 ;
    }

    int typeIndexCol ;
    if (rank % processGrid[1] < timesBlockInFinalCols) {
        typeIndexCol = 0 ;
    } else if (rank % processGrid[1] == timesBlockInFinalCols) {
        typeIndexCol = 1 ;
    } else {
        typeIndexCol = 2 ;
    }

    printf("Type for %d >> %d %d\n", rank, typeIndexRow, typeIndexCol) ;

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
    
    //CREATED TYPES FOR INDEXED ROW

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
    int rank, int *processGrid, 
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

    printf("Times %d %d\n", timesBlockInFinalRows, timesBlockInFinalCols) ;
    int residualRecvdRows ;
    if (rank % processGrid[0] < timesBlockInFinalRows) {
        residualRecvdRows = blockRows ;
    } else if (rank % processGrid[0] == timesBlockInFinalRows) {
        residualRecvdRows = finalRowsNum % blockRows ;
    } else {
        residualRecvdRows = 0 ;
    }

    int residualRecvdCols ;
    if (rank % processGrid[1] < timesBlockInFinalCols) {
        residualRecvdCols = blockCols ;
    } else if (rank % processGrid[1] == timesBlockInFinalCols) {
        residualRecvdCols = finalColsNum % blockCols ;
    } else {
        residualRecvdCols = 0 ;
    }

    *subMatRowsNum = blockRows * timesGridInRows + residualRecvdRows ;
    *subMatColsNum = blockCols * timesGridInCols + residualRecvdCols ;

    printf("Dims per proc %d >> %d %d\n", rank, *subMatRowsNum, *subMatColsNum) ;
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