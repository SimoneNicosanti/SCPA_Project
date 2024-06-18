import pandas as pd
import matplotlib.pyplot as plt

FILE_LIST = ["Square", "Rect"]
FOLDER_LIST = ["MPI", "CUDA"]

def testsAnalyzer(folder : str, case : str, separatorColumn : str, timeMult : float) :
    dataSet = pd.read_csv(f"../{folder}/Tests/{folder}_{case}_Test.csv")

    avgDataSet = dataSet.groupby(["M", "N", "K", separatorColumn]).mean().reset_index()
    avgDataSet["GFLOPS"] = (2 * avgDataSet["M"] * avgDataSet["N"] * avgDataSet["K"] / avgDataSet["ParallelTime"]) * timeMult

    comparisonDataSet = avgDataSet[avgDataSet["SequentialTime"] > 0].copy()
    comparisonDataSet["SpeedUp"] = comparisonDataSet["SequentialTime"] / comparisonDataSet["ParallelTime"]

    gflopsDataSet = avgDataSet.drop(columns = ["SequentialTime", "RelativeError"])
    comparisonDataSet.to_csv(f"../{folder}/Generated/{folder}_{case}_Comparisons.csv", index = False)
    gflopsDataSet.to_csv(f"../{folder}/Generated/{folder}_{case}_GFLOPS.csv", index = False)

    gflopsChart(gflopsDataSet, separatorColumn, folder, case)
    speedUpChart(comparisonDataSet, separatorColumn, folder, case) 


def gflopsChart(gflopsDataSet : pd.DataFrame, separatorColumn: str, folder : str, case : str) :
    plt.figure(figsize = (15, 5), clear = True, layout = "tight")
    for sepValue in gflopsDataSet[separatorColumn].unique() :
        subDataSet = gflopsDataSet[(gflopsDataSet[separatorColumn] == sepValue)]
        plt.plot(subDataSet["K"], subDataSet["GFLOPS"], marker = "o", label = f"{separatorColumn} = {sepValue}")

    plt.xticks(gflopsDataSet["K"].unique(), rotation = 30)
    plt.legend(loc = "upper left")
    plt.grid()
    plt.xlabel("K")
    plt.ylabel("GFLOPS")
    plt.title(f"{folder} - GFLOPS Trend - {case} Case")
    plt.savefig(f"../{folder}/Charts/{folder}_GFLOPS_{case}_Chart.png", format = "png", transparent = True)
    plt.clf()

def speedUpChart(comparisonDataSet : pd.DataFrame, separatorColumn: str, folder : str, case : str) :
    plt.figure(figsize = (15, 5), clear = True, layout = "tight")
    for sepValue in comparisonDataSet[separatorColumn].unique() :
        subDataSet = comparisonDataSet[(comparisonDataSet[separatorColumn] == sepValue)]
        plt.plot(subDataSet["K"], subDataSet["SpeedUp"], marker = "o", label = f"{separatorColumn} = {sepValue}")

    plt.xticks(comparisonDataSet["K"].unique(), rotation = 30)
    plt.legend(loc = "upper left")
    plt.grid()
    plt.xlabel("K")
    plt.ylabel("SpeedUp")
    plt.title(f"{folder} - SpeedUp Trend - {case} Case")
    plt.savefig(f"../{folder}/Charts/{folder}_SpeedUp_{case}_Chart.png", format = "png", transparent = True)
    plt.clf()



def mpiAnalyze() :
    testsAnalyzer("MPI", "Square", "ProcessNum", 10 ** -9)
    testsAnalyzer("MPI", "Rect", "ProcessNum", 10 ** -9) 


def cudaAnalyze() :
    testsAnalyzer("CUDA", "Square", "KernelVersion", 10 ** -6)
    testsAnalyzer("CUDA", "Rect", "KernelVersion", 10 ** -6)


def main() :
    mpiAnalyze() 
    cudaAnalyze() 


if __name__ == "__main__" :
    main()