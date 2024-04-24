typedef enum MESSAGE_TYPE {
    STRING,
    INT_ARRAY,
    REAL_ARRAY,
    MATRIX,
    INTEGER,
    REAL
} MESSAGE_TYPE ;

typedef union Content
{
    char *string ;
    double *realArray ;
    int *intArray ;
    double **matrix ;
    int integer ;
    double real ;
} Content ;

void printMessage(char *header, Content content, MESSAGE_TYPE type, int rank, int printerProcRank, int firstDimLen, int secDimLen, int newLine) ;