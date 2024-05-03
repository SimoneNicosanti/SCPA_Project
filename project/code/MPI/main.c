#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <time.h>
#include <string.h>
#include <math.h>

#include "MpiProduct.h"
#include "../Matrix/Matrix.h"
#include "Utils.h"

#define PROCESS_GRID_DIMS 2
#define TYPE_MATRIX_NUM MPI_FLOAT
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
    float **subA, float **subB, float **subC,
    int m, int k, int n,
    int mb, int kb, int nb,
    int subm, int subk, int subn,
    float **C
) ;

void matrixSendToAll(float **matrix, int rows, int cols, int blockRows, int blockCols, int *processGrid, int invertGrid, int perGroupOfRows) ;
float **matrixRecvFromRoot(int rows, int cols, int blockRows, int blockCols, int *processGrid, int *subRowsPtr, int *subColsPtr, int invertGrid, int perGroupOfRows) ;

void createReduceCommunicator() ;
void subMatrixProduct(float **subA, float **subB, float **cSubProd, int subm, int subk, int subn) ;


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

    double startTime, endTime ;
    if (processInfo.myRank == ROOT_PROCESS) {
        startTime = MPI_Wtime() ;
    }

    //int PROCESS_GRID[PROCESS_GRID_DIMS] = {0} ;
    MPI_Dims_create(procNum, PROCESS_GRID_DIMS, PROCESS_GRID) ;
    if (processInfo.myRank == ROOT_PROCESS) {
        printf("Cartesian Grid: [%d, %d]\n", PROCESS_GRID[0], PROCESS_GRID[1]) ;
    }
    int periods[2] = {0, 0} ;
    MPI_Cart_create(WORLD_COMM, 2, PROCESS_GRID, periods, 0, &CART_COMM) ;
    MPI_Cart_coords(CART_COMM, processInfo.myRank, 2, &processInfo.myCoords) ;

    int invertedGrid[2] ;
    invertedGrid[0] = PROCESS_GRID[1] ;
    invertedGrid[1] = PROCESS_GRID[0] ;

    int scatterCGrid[2] ;
    scatterCGrid[0] = PROCESS_GRID[0] ;
    scatterCGrid[1] = 1 ;
    float **C ;
    if (processInfo.myRank == ROOT_PROCESS) {
        float **A = allocRandomMatrix(m, k) ;
        // for (int i = 0 ; i < m ; i++) {
        //     for (int j = 0 ; j < k ; j++) {
        //         A[i][j] = i * k + j ;
        //     }
        // }

        float **B = allocRandomMatrix(k, n) ;
        // for (int i = 0 ; i < k ; i++) {
        //     for (int j = 0 ; j < n ; j++) {
        //         B[i][j] = i * n + j ;
        //     }
        // }

        C = allocRandomMatrix(m, n) ;
        // for (int i = 0 ; i < m ; i++) {
        //     for (int j = 0 ; j < n ; j++) {
        //         C[i][j] = i * n + j ;
        //     }
        // }

        matrixSendToAll(A, m, k, mb, kb, PROCESS_GRID, 0, 0) ; // SEND A
        matrixSendToAll(B, k, n, kb, nb, invertedGrid, 1, 1) ; // SEND B
        matrixSendToAll(C, m, n, mb, nb, scatterCGrid, 0, 1) ; // SEND C

        // SEND C solo ai processi root di una REDUCE
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
    float **subA = matrixRecvFromRoot(m, k, mb, kb, PROCESS_GRID, &subm, &subk, 0, 0) ;
    float **subB =  matrixRecvFromRoot(k, n, kb, nb, invertedGrid, &subk, &subn, 1, 1) ;
    float **subC = NULL ;
    if (processInfo.myCoords[1] == 0) {
        subC = matrixRecvFromRoot(m, n, mb, nb, scatterCGrid, &subm, &subn, 0, 1) ;
    } else {
        subC = allocMatrix(subm, subn) ;
    }
    
    executeCompleteProduct(subA, subB, subC, m, k, n, mb, kb, nb, subm, subk, subn, C) ;

    if (processInfo.myRank == ROOT_PROCESS) {
        Content content ;
        content.matrix = C ;
        //printMessage("FINAL MATRIX", content, MATRIX, processInfo.myRank, ROOT_PROCESS, m, n, 1) ;
    }

    if (processInfo.myRank == ROOT_PROCESS) {
        endTime = MPI_Wtime() ;
        double totalTime = endTime - startTime ;
        printf("Total Time: %f\n", totalTime) ;
        
        
        unsigned long num = 2 * m * k * n ;
        double GFLOPS = (num / totalTime) * pow(10, -9) ;
        printf("GFLOPS %f\n", GFLOPS ) ;
    }

    MPI_Finalize() ;
    

    return 0 ;
}

// Executes C <- A * B sui sottoblocchi
// TODO > Capire se va bene usare delle variabili globali per il rank e per il process grid
// TODO > Fare deallocazione dei tipi usati per inviare i dati
// TODO > Fare presa dei tempi
// TODO > Fare calcolo dell'errore relativo
// TODO > Fare Refactoring del codice
//          - Spostare qualcosa in altri file??
//          - Capire perché MPI_IN_PLACE non
// TODO > Change double to float
void executeCompleteProduct(
    float **subA, float **subB, float **subC,
    int m, int k, int n,
    int mb, int kb, int nb,
    int subm, int subk, int subn,
    float **C
) {

    Content content ;

    createReduceCommunicator() ;
    
    subMatrixProduct(subA, subB, subC, subm, subk, subn) ;

    printf("Process %d; Finished Product\n", processInfo.myRank) ;

    float **reducedC ;
    if (processInfo.myCoords[1] == 0) {
        reducedC = allocMatrix(subm, subn) ;
    }
    
    MPI_Reduce(&(subC[0][0]), &(reducedC[0][0]), subm * subn, TYPE_MATRIX_NUM, MPI_SUM, 0, REDUCE_COMM_ARRAY[processInfo.myCoords[0]]) ;

    printf("REDUCED DONE\n") ;

    // IS ONE OF THE REDUCE ROOT
    if (processInfo.myCoords[1] == 0 && processInfo.myRank != ROOT_PROCESS) {
        MPI_Request mpiRequest ;
        MPI_Send(&(reducedC[0][0]), subm * subn, TYPE_MATRIX_NUM, ROOT_PROCESS, SEND_TAG, WORLD_COMM) ;
    }
   
    if (processInfo.myRank == ROOT_PROCESS) {
        //RECV DEI PEZZI CHE VENGONO MANDATI E INSERIMENTO IN C
        MPI_Datatype returnTypes[3][3] ;
        int gatherGrid[2] = {PROCESS_GRID[0], 1} ;
        createSendDataTypes(m, n, mb, nb, gatherGrid, returnTypes) ;

        for (int procRow = 0 ; procRow < PROCESS_GRID[0] ; procRow++) {
            
            int rowIndex = procRow ;
            int colIndex = 0 ;

            int proc ;
            int coords[2] = {rowIndex, colIndex} ;
            MPI_Cart_rank(CART_COMM, coords, &proc) ;

            MPI_Datatype returnType = computeSendTypeForProc(rowIndex, colIndex, gatherGrid, m, n, mb, nb, returnTypes) ;
            MPI_Request mpiRequest ;

            if (proc != processInfo.myRank) {
                MPI_Recv(&(C[rowIndex * mb][colIndex * nb]), 1, returnType, proc, SEND_TAG, WORLD_COMM, MPI_STATUS_IGNORE) ;
            } else {
                MPI_Sendrecv(
                    &(reducedC[0][0]), subm * subn, TYPE_MATRIX_NUM, processInfo.myRank, SEND_TAG, 
                    &(C[rowIndex * mb][colIndex * nb]), 1, returnType, proc, SEND_TAG,
                    CART_COMM, MPI_STATUS_IGNORE) ;
            }
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
        for (int j = 0 ; j < subn ; j++) {
            for (int t = 0 ; t < subk ; t++) {
                //printf("A , B = %f , %f\n", subA[i][t], subB[j][t]) ;
                subC[i][j] += subA[i][t] * subB[t][j] ;
            }
        }
    }
}


float **matrixRecvFromRoot(int rows, int cols, int blockRows, int blockCols, int *processGrid, int *subRowsPtr, int *subColsPtr, int invertGrid, int perGroupOfRows) {
    int subMatRows, subMatCols ;
    int procCoords[2] ;
    MPI_Cart_coords(CART_COMM, processInfo.myRank, 2, procCoords) ;

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
    MPI_Recv(&(subMat[0][0]), subMatRows * subMatCols, TYPE_MATRIX_NUM, ROOT_PROCESS, SEND_TAG, WORLD_COMM, MPI_STATUS_IGNORE) ;
    
    Content content ;
    content.matrix = subMat ;
    // printMessage("SUB MATRIX: ", content, MATRIX, processInfo.myRank, 0, subMatRows, subMatCols, 1) ;

    *subRowsPtr = subMatRows ;
    *subColsPtr = subMatCols ;
    return subMat ;
}


void matrixSendToAll(float **matrix, int rows, int cols, int blockRows, int blockCols, int *processGrid, int invertGrid, int perGroupOfRows) {
    MPI_Datatype matrixTypes[3][3] ;
    createSendDataTypes(rows, cols, blockRows, blockCols, processGrid, matrixTypes) ;

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

            // Bisogna fare ISend per l'invio da zero a se stesso, altrimenti ho un blocco
            // TODO > Controlla se va bene oppure se è meglio usare la SendRecv invece
            MPI_Isend(&(matrix[rowIndex * blockRows][colIndex * blockCols]), 1, sendType, proc, SEND_TAG, WORLD_COMM, &mpiRequest) ;
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
        displs_2[i] = currDispl * sizeof(float) ;
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
        // if (strcmp(argv[i], "-nb") == 0) {
        //     *nbPtr = atoi(argv[i+1]) ;
        // }
    }
    *nbPtr = *nPtr ;
    if (*mPtr <= 0 || *kPtr <= 0 || *nPtr <= 0 || *mbPtr <= 0 || *kbPtr <= 0 || *nbPtr <= 0) {
        return 0 ;
    }

    return 1 ;
}