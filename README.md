# SCPA_Project

## Compilazione
Per compilare il codice spostarsi nella directory _project_ ed eseguire il comando

```
./CMakeCaller.sh
```
I file eseguibili saranno prodotti nella directory _out_


## Esecuzione
Spostarsi nella directory _out_

### MPI
Per eseguire un singolo test con MPI eseguire il comando

```
mpiexec -np <numProc> ./MpiSingleTest.out -m <mValue> -k <kValue> -n <nValue> -mb <mBlockSize> -nb <nBlockSize>
```
In questo caso il test è eseguito di default con delle matrici inizializzate nel seguente modo:
```
M[i][j] = i * numRows + j
```

Per eseguire più test di seguito con diverse dimensioni e con matrici generate randomicamente eseguire:
```
mpiexec -np <numProc> ./MpiTest.out
```
In questo caso l'esecuzione scriverà i tempi in file nella directory _project/Results/MPI/Tests_: assicurarsi quindi che la direcory esista prima di eseguire.

Diverse configurazioni di esecuzione al variare del numero di processi possono essere eseguite con il comando eseguito dalla directory _project_:
```
./RunMpiTests.sh
```

### CUDA
Per eseguire un singolo test con CUDA eseguire il comando

```
./CudaSingleTest.out -m <mValue> -k <kValue> -n <nValue> -v <kernelVersion>
```
Se la versione del kernel non è specificata di default si usa il kernel 4.

In questo caso il test è eseguito di default con delle matrici inizializzate nel seguente modo:
```
M[i][j] = i * numRows + j
```

Per eseguire più test di seguito con diverse dimensioni e con matrici generate randomicamente eseguire:
```
./CudaTest.out -v <kernelVersion>
```
In questo caso l'esecuzione scriverà i tempi in file nella directory _project/Results/CUDA/Tests_: assicurarsi quindi che la direcory esista prima di eseguire.

Diverse configurazioni di esecuzione al variare della versione del kernel possono essere eseguite con il comando eseguito dalla directory _project_:
```
./RunCudaTests.sh
```