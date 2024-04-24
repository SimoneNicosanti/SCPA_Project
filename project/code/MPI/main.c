#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <time.h>
#include <string.h>

#include "MpiProduct.h"
#include "../Matrix/Matrix.h"
#include "Utils.h"

#define PROCESS_GRID_DIMS 2
#define MATRIX_NUM_TYPE MPI_DOUBLE
#define ROOT_PROCESS 0

MPI_Datatype LAST_PROC_ROW_TYPE ;
MPI_Datatype SECOND_PROC_ROW_TYPE ;
MPI_Datatype THIRD_PROC_ROW_TYPE ;

MPI_Datatype ROW_INDEXED_SUBTYPES[3] ;

MPI_Comm WORLD_COMM ;
MPI_Comm CART_COMM ;


int procRank ;




int extractParams(int argc, char *argv[], int *mPtr, int *kPtr, int *nPtr, int *mbPtr, int *kbPtr, int *nbPtr) ;
void createDataTypes(int rowsNum, int colsNum, int rowsNumBlock, int colsNumBlock, int *processGrid) ;



int main(int argc, char *argv[]) {
    // Prima di prendere il tempo devo generare le matrici, altrimenti il tempo di generazione è compreso
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

    // Capire se conviene usare un comunicatore cartesiano o meno...
    // int periods[PROCESS_GRID_DIMS] = {0} ;
    // MPI_Cart_create(WORLD_COMM, PROCESS_GRID_DIMS, processGrid, periods, 0, &CART_COMM) ;

    createDataTypes(1, 103, 20, 20, processGrid) ;

    return ;

    if (myRank == ROOT_PROCESS) {
        double **A = allocRandomMatrix(1, 50) ;
        double **B = allocRandomMatrix(k, n) ; // TODO Per ora allocata per righe, ma capire bene se conviene andare per colonne
        double **C = allocRandomMatrix(m, n) ;

        for (int j = 0 ; j < 50 ; j++) {
            A[0][j] = j ;
        }

        MPI_Send(&(A[0][0]), 1, ROW_INDEXED_SUBTYPES[0], 0, 10, WORLD_COMM) ;

        double output[100] = {0} ;
        MPI_Recv(output, 6, MATRIX_NUM_TYPE, 0, 10, WORLD_COMM, MPI_STATUS_IGNORE) ;

        for (int i = 0 ; i < 10 ; i++) {
            printf("%f ", output[i]) ;
        }
    }


    return 0 ;
}

void createDataTypes(int rowsNum, int colsNum, int rowsNumBlock, int colsNumBlock, int *processGrid) {
    Content content ;

    int numDivsRows = rowsNum / rowsNumBlock + ((rowsNum % rowsNumBlock == 0) ? 0 : 1) ;
    int numDivsCols = colsNum / colsNumBlock + ((colsNum % colsNumBlock == 0) ? 0 : 1) ;

    int numRowsLastProc = rowsNum % rowsNumBlock ;
    int numColsLastProc = colsNum % colsNumBlock ;

    // Times the grid fits in the matrix
    int timesGridInRowDiv = numDivsRows / processGrid[0] + (numDivsRows % processGrid[0] == 0 ? 0 : 1) ;
    int timesGridInColsDiv = colsNum / (processGrid[1] * colsNumBlock) + (numDivsCols % (processGrid[1] * colsNumBlock) == 0 ? 0 : 1);
    if (procRank == ROOT_PROCESS) {
        printf("(RowsPerProc, ColsPerProc) = (%d, %d)\n", numDivsRows, numDivsCols) ;
        printf("(RowsFinalPart, ColsFinalPart) = (%d, %d)\n", numRowsLastProc, numColsLastProc) ;
        printf("(TimesGridInRows, TimesGridInCols) = (%d, %d)\n", timesGridInRowDiv, timesGridInColsDiv) ;
    }

    /*
        Create three datatypes: 
            * One for last process, with all blocks of size numPerProc but the lastOne with numLastProc
            * One for middle process, with all blocks of size numPerProc but with n blocks
            * One for middle proc, with n+1 blocks all of the same size
    */

    //First type: for the process that gets the last part of the cols
    int displsArray[timesGridInColsDiv] ;
    displsArray[0] = 0 ;
    for (int i = 1 ; i < timesGridInColsDiv ; i++) {
        displsArray[i] = i * processGrid[1] * colsNumBlock ;
    }
    int firstBlockLengthArray[timesGridInColsDiv] ;
    for (int i = 0 ; i < timesGridInColsDiv - 1 ; i++) {
        firstBlockLengthArray[i] = colsNumBlock ; 
    }
    firstBlockLengthArray[timesGridInColsDiv - 1] = numColsLastProc ;
    MPI_Type_indexed(timesGridInColsDiv, firstBlockLengthArray, displsArray, MATRIX_NUM_TYPE, &ROW_INDEXED_SUBTYPES[0]) ;
    
    content.intArray = firstBlockLengthArray ;
    printMessage("First Counts: ", content, INT_ARRAY, procRank, ROOT_PROCESS, timesGridInColsDiv, 0, 0) ;
    content.intArray = displsArray ;
    printMessage("First Displs: ", content, INT_ARRAY, procRank, ROOT_PROCESS, timesGridInColsDiv, 0, 1) ;

    //Secondo tipo: per i processi che prendono blocchi completi
    int secondBlockLengthArray[timesGridInColsDiv] ;
    for (int i = 0 ; i < timesGridInColsDiv ; i++) {
        secondBlockLengthArray[i] = colsNumBlock ;
    }
    MPI_Type_indexed(timesGridInColsDiv, secondBlockLengthArray, displsArray, MATRIX_NUM_TYPE, &ROW_INDEXED_SUBTYPES[1]) ;
    content.intArray = secondBlockLengthArray ;
    printMessage("Second Counts: ", content, INT_ARRAY, procRank, ROOT_PROCESS, timesGridInColsDiv, 0, 0) ;
    content.intArray = displsArray ;
    printMessage("Second Displs: ", content, INT_ARRAY, procRank, ROOT_PROCESS, timesGridInColsDiv, 0, 1) ;

    //Terzo tipo: per i processi che prendono un blocco in meno
    int thirdBlockLengthArray[timesGridInColsDiv - 1] ;
    for (int i = 0 ; i < timesGridInColsDiv - 1 ; i++) {
        thirdBlockLengthArray[i] = colsNumBlock ;
    }
    int thirdDisplsArray[timesGridInColsDiv - 1] ;
    thirdDisplsArray[0] = 0 ;
    for (int i = 1 ; i < timesGridInColsDiv - 1 ; i++) {
        thirdDisplsArray[i] = i * processGrid[1] * colsNumBlock ;
    }
    MPI_Type_indexed(timesGridInColsDiv - 1, thirdBlockLengthArray, thirdDisplsArray, MATRIX_NUM_TYPE, &ROW_INDEXED_SUBTYPES[2]) ;
    content.intArray = thirdBlockLengthArray ;
    printMessage("Third Counts: ", content, INT_ARRAY, procRank, ROOT_PROCESS, timesGridInColsDiv - 1, 0, 0) ;
    content.intArray = thirdDisplsArray ;
    printMessage("Third Displs: ", content, INT_ARRAY, procRank, ROOT_PROCESS, timesGridInColsDiv - 1, 0, 1) ;

    // Commit of the created datatypes
    for (int i = 0 ; i < 3 ; i++) {
        MPI_Type_commit(&ROW_INDEXED_SUBTYPES[i]) ;
    }

    // FINO A QUI FUNZIONA TUTTO

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