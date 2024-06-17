import pandas as pd
import matplotlib.pyplot as plt

FILE_LIST = ["Square", "Rect"]
FOLDER_LIST = ["MPI"]

def mpiSquareTestAnalyze(folder : str, case : str) :
    dataSet = pd.read_csv(f"../{folder}/Tests/{folder}_{case}_Test.csv")

    avgDataSet = dataSet.groupby(["M", "N", "K", "ProcessNum"]).mean().reset_index()
    avgDataSet["GFLOPS"] = (2 * avgDataSet["M"] * avgDataSet["N"] * avgDataSet["K"] / avgDataSet["ParallelTime"]) * (10 ** -9)

    comparisonDataSet = avgDataSet[avgDataSet["SequentialTime"] > 0] 
    comparisonDataSet["SpeedUp"] = comparisonDataSet["SequentialTime"] / comparisonDataSet["ParallelTime"]

    gflopsDataSet = avgDataSet.drop(columns = ["SequentialTime", "RelativeError"])
    comparisonDataSet.to_csv(f"../{folder}/Generated/{folder}_{case}_Comparisons.csv", index = False)
    gflopsDataSet.to_csv(f"../{folder}/Generated/{folder}_{case}_GFLOPS.csv", index = False)

    plt.figure(figsize = (15, 5), clear = True, layout = "tight")
    for procNum in gflopsDataSet["ProcessNum"].unique() :
        subDataSet = gflopsDataSet[(gflopsDataSet["ProcessNum"] == procNum)]
        plt.plot(subDataSet["K"], subDataSet["GFLOPS"], marker = "o", label = f"Process Num = {procNum}")
    plt.xticks(gflopsDataSet["K"].unique(), rotation = 30)
    plt.legend()
    plt.grid()
    plt.xlabel("K")
    plt.ylabel("GFLOPS")
    plt.title(f"{folder} - GFLOPS Trend - {case} Case")
    plt.savefig(f"../{folder}/Charts/{folder}_GFLOPS_{case}_Chart.png")
    plt.clf()

def mpiRectTestAnalyze(folder : str, case : str) :
    dataSet = pd.read_csv(f"../{folder}/Tests/{folder}_{case}_Test.csv")

    avgDataSet = dataSet.groupby(["M", "N", "K", "ProcessNum"]).mean().reset_index()
    avgDataSet["GFLOPS"] = (2 * avgDataSet["M"] * avgDataSet["N"] * avgDataSet["K"] / avgDataSet["ParallelTime"]) * (10 ** -9)

    comparisonDataSet = avgDataSet[avgDataSet["SequentialTime"] > 0] 
    comparisonDataSet["SpeedUp"] = comparisonDataSet["SequentialTime"] / comparisonDataSet["ParallelTime"]

    gflopsDataSet = avgDataSet.drop(columns = ["SequentialTime", "RelativeError"])
    comparisonDataSet.to_csv(f"../{folder}/Generated/{folder}_{case}_Comparisons.csv", index = False)
    gflopsDataSet.to_csv(f"../{folder}/Generated/{folder}_{case}_GFLOPS.csv", index = False)

    plt.figure(figsize = (15, 5), clear = True, layout = "tight")
    for procNum in gflopsDataSet["ProcessNum"].unique() :
        subDataSet = gflopsDataSet[gflopsDataSet["ProcessNum"] == procNum]
        plt.plot(subDataSet["K"], subDataSet["GFLOPS"], marker = "o", label = f"Process Num = {procNum}")
    plt.xticks(gflopsDataSet["K"].unique(), rotation = 30)
    plt.legend()
    plt.grid()
    plt.xlabel("K")
    plt.ylabel("GFLOPS")
    mValue = gflopsDataSet["M"].values[0]
    plt.title(f"{folder} - GFLOPS Trend - {case} Case")
    plt.title(f"{folder} - GFLOPS Trend - Rectangular Case - Fixed Dimension {mValue}")
    plt.savefig(f"../{folder}/Charts/{folder}_GFLOPS_{case}_Chart.png")
    plt.clf()


def cudaSquareTestAnalyze(folder : str, case : str) :
    pass

def cudaRectTestAnalyze(folder : str, case : str) :
    pass



def mpiAnalyze() :
    mpiSquareTestAnalyze("MPI", "Square")
    mpiRectTestAnalyze("MPI", "Rect") 


def cudaAnalyze() :
    cudaSquareTestAnalyze("CUDA", "Square")
    cudaRectTestAnalyze("CUDA", "Rect")


def main() :
    mpiAnalyze() 
    cudaAnalyze() 


if __name__ == "__main__" :
    main()