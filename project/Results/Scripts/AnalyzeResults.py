import pandas as pd
import matplotlib.pyplot as plt


def main() :
    mpiDataSet = pd.read_csv("../MPI/MPI_Test.csv")

    avgDataSet = mpiDataSet.groupby(["M", "N", "K", "ProcessNum"]).mean().reset_index()
    avgDataSet["GFLOPS"] = (2 * avgDataSet["M"] * avgDataSet["N"] * avgDataSet["K"] / avgDataSet["ParallelTime"]) * (10 ** -9)

    comparisonDataSet = avgDataSet[avgDataSet["SequentialTime"] > 0] 
    comparisonDataSet["SpeedUp"] = comparisonDataSet["SequentialTime"] / comparisonDataSet["ParallelTime"]

    gflopsDataSet = avgDataSet.drop(columns = ["SequentialTime", "RelativeError"])
    comparisonDataSet.to_csv("../MPI/MPI_Comparisons.csv", index = False)
    gflopsDataSet.to_csv("../MPI/MPI_GFLOPS.csv", index = False)

    plt.figure(figsize = (15, 5), clear = True, layout = "tight")
    for procNum in gflopsDataSet["ProcessNum"].unique() :
        subDataSet = gflopsDataSet[gflopsDataSet["ProcessNum"] == procNum]
        plt.plot(subDataSet["K"], subDataSet["GFLOPS"], marker = "o", label = f"Process Num = {procNum}")
    plt.xticks([x for x in range(0, 6250, 250)], rotation = 30)
    plt.legend()
    plt.grid()
    plt.xlabel("K")
    plt.ylabel("GFLOPS")
    plt.title("GFLOPS Trend")
    plt.savefig("../MPI/Charts/GFLOPS_Chart.png")
    plt.clf()

    return 


if __name__ == "__main__" :
    main()