
void main() {

    /*  
        1. for i = 0 --> 10000: Dim of matric
            - set seed
            - generate matrices
            - call function to execute the MPI code and take back the time
            - call function to execute the sequential code and take back the time
            - call error computation
            - compile struct of stats
            - compute GFLOPS ??
            - write values on CSV file
    */
   for (int probDim = 0 ; probDim < 10001 ; probDim += 100) { 
        float **A = allocRandomMatrix(probDim, probDim) ;
        float **B = allocRandomMatrix(probDim, probDim) ;
        float **C = allocRandomMatrix(probDim, probDim) ;
        float **cCopy = allocMatrix() ;
        
   }
}