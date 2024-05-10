import pandas as pd
import matplotlib.pyplot as plt


def main() :
    mpiDataSet = pd.read_csv("../MPI_Test.csv")

    avgDataSet = mpiDataSet.groupby(["M", "N", "K", "ProcessNum"]).mean().reset_index()
    avgDataSet["GFLOPS"] = (2 * avgDataSet["M"] * avgDataSet["N"] * avgDataSet["K"] / avgDataSet["ParallelTime"]) * (10 ** -9)
    avgDataSet["SpeedUp"] = avgDataSet["SequentialTime"] / avgDataSet["ParallelTime"]
    avgDataSet = avgDataSet.sort_values(by = ["ProcessNum", "M", "N", "K"])
    avgDataSet.to_csv("../MPI_Avg.csv")
    return 


if __name__ == "__main__" :
    main()