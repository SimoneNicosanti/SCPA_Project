#include <mpi.h>

MPI_Datatype computeSendTypeForProc(
    int rowRank, int colRank, int *processGrid,
    int rowsNum, int colsNum, 
    int blockRows, int blockCols,  
    MPI_Datatype typesMatrix[3][3]
) ;

void createSendDataTypes(
    int rowsNum, int colsNum, 
    int blockRows, int blockCols, 
    int *processGrid, 
    MPI_Datatype baseType, MPI_Datatype typesMatrix[3][3]
) ;

void computeSubMatrixDimsPerProc(
    int rowRank, int colRank, int *processGrid, 
    int rowsNum, int colsNum, 
    int blockRows, int blockCols, 
    int *subMatRowsNum, int *subMatColsNum
) ;

void freeSendDataTypes(MPI_Datatype typesMatrix[3][3]) ;