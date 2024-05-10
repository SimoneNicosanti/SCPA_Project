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

    plt.plot(avgDataSet["K"], avgDataSet["ParallelTime"])
    plt.show()
    return 


if __name__ == "__main__" :
    main()