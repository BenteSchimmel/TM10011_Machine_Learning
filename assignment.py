import pandas as pd
import numpy
import seaborn
import matplotlib.pyplot as plt
from scipy.stats import linregress


df = pd.read_csv("datasets.csv")

nr_of_datasets = df["dataset"].nunique()
# print("Number of datasets:", nr_of_datasets)

# print("Names of datasets:")
for name in df["dataset"].unique():
    print("-", name)

groepen = df.groupby("dataset")
kolommen = groepen[["x", "y"]]

stats = kolommen.agg(['count', 'mean', 'var', 'std'])
# print(stats)

# plt.figure()
# seaborn.violinplot(data=df, x="dataset", y="x")
# plt.title("Violin plot of X per dataset")
# plt.tight_layout()
# plt.show()

# plt.figure()
# seaborn.violinplot(data=df, x="dataset", y="y")
# plt.title("Violin plot of y per dataset")
# plt.tight_layout()
# plt.show()

correlatie = kolommen.corr("pearson")
covariance = kolommen.cov()
print(correlatie, covariance)

def regressie_stats(group):
    res = linregress(group["x"], group["y"])
    return pd.Series({
        "slope": res.slope,
        "intercept": res.intercept,
        "r_value": res.rvalue,
        "p_value": res.pvalue,
        "std_err": res.stderr
    })

# Toepassen per dataset
regressie_resultaten = df.groupby("dataset").apply(regressie_stats)

print("\nLineaire regressie per dataset:")
print(regressie_resultaten)

g = seaborn.FacetGrid(df, col="dataset", col_wrap=3, height=3)
g.map_dataframe(seaborn.scatterplot, x="x", y="y")

g.set_axis_labels("x", "y")
g.set_titles("Dataset: {col_name}")

plt.show()

seaborn.lmplot(data=df, x="x",y="y",col="dataset",col_wrap=3,height=3,)
plt.show()