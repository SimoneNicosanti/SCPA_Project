#include <mpi.h>

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

void createSendDataTypes(
    int rowsNum, int colsNum, 
    int blockRows, int blockCols, 
    int *processGrid, 
    MPI_Datatype baseType, MPI_Datatype typesMatrix[3][3]
) {

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
    MPI_Type_indexed(timesGridInCols + 1, blockLengths, displs, baseType, &indexedTypes[0]) ;
    
    blockLengths[timesGridInCols] = finalColsNum % blockCols ;
    MPI_Type_indexed(timesGridInCols + 1, blockLengths, displs, baseType, &indexedTypes[1]) ;

    MPI_Type_indexed(timesGridInCols, blockLengths, displs, baseType, &indexedTypes[2]) ;

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

void freeSendDataTypes(MPI_Datatype typesMatrix[3][3]) {
    for (int i = 0 ; i < 3 ; i++) {
        for (int j = 0 ; j < 3 ; j++) {
            MPI_Type_free(&(typesMatrix[i][j])) ;
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

