import pandas as pd
import numpy


df = pd.read_csv("datasets.csv")

nr_of_datasets = df["dataset"].nunique()
print("Number of datasets:", nr_of_datasets)

print("Names of datasets:")
for name in df["dataset"].unique():
    print("-", name)
