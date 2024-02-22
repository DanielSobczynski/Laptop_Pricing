import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Downloading data to Pandas dataframe
filepath="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod2.csv"
df = pd.read_csv(filepath, header=0)
df= df.drop(df.columns[0:2], axis=1)
print(df.to_string())


# Checking correlation of 3 variables with laptop price
sns.regplot(x="CPU_frequency", y="Price", data=df)
plt.ylim(0,)
#plt.show()

sns.regplot(x="Screen_Size_inch", y="Price", data=df)
plt.ylim(0,)
#plt.show()

sns.regplot(x="Weight_pounds", y="Price", data=df)
plt.ylim(0,)
#plt.show()

for param in ["CPU_frequency", "Screen_Size_inch","Weight_pounds"]:
    print(f"Correlation of Price and {param} is ", df[[param,"Price"]].corr())

# Generating Box plots for the different variables
sns.boxplot(x="Category", y="Price", data=df)
#plt.show()

sns.boxplot(x="GPU", y="Price", data=df)
#plt.show()

sns.boxplot(x="OS", y="Price", data=df)
#plt.show()

sns.boxplot(x="CPU_core", y="Price", data=df)
#plt.show()

sns.boxplot(x="RAM_GB", y="Price", data=df)
#plt.show()

sns.boxplot(x="Storage_GB_SSD", y="Price", data=df)
#plt.show()

# Generating statistical descriptions
print((df.describe()).to_string())
print((df.describe(include=['object'])).to_string())

# Grouping variables
df_group = df[["GPU", "CPU_core", "Price"]]
group_test = df_group.groupby(["GPU", "CPU_core"], as_index=False).mean()
print(group_test)

group_pivot = group_test.pivot(index="GPU", columns="CPU_core")
print(group_pivot)

# Plotting pivots
fig, ax = plt.subplots()
im = ax.pcolor(group_pivot, cmap='RdBu')

# Labeling names
row_labels = group_pivot.columns.levels[1]
col_labels = group_pivot.index

# Moving ticks and labels to the center
ax.set_xticks(np.arange(group_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(group_pivot.shape[0]) + 0.5, minor=False)

# Inserting labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

fig.colorbar(im)
plt.show()

# Calculating Pearson correlation
for param in ['RAM_GB','CPU_frequency','Storage_GB_SSD','Screen_Size_inch','Weight_pounds','CPU_core','OS','GPU','Category']:
    pearson_coef, p_value = stats.pearsonr(df[param], df['Price'])
    print("The Pearson Correlation for ",param," is", pearson_coef, " with a P-value of P =", p_value)
