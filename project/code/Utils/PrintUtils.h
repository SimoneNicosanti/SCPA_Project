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
    float *realArray ;
    int *intArray ;
    float *matrix ;
    int integer ;
    float real ;
} Content ;

#ifdef __cplusplus
extern "C"
#endif
void printMessage(char *header, Content content, MESSAGE_TYPE type, int rank, int printerProcRank, int firstDimLen, int secDimLen, int newLine) ;