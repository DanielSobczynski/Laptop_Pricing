import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filepath = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod1.csv"

df = pd.read_csv(filepath, header=0)
df= df.drop(df.columns[0], axis=1)
print(df.info())
print(df.head())

# Evaluating the dataset for missing data
missing_data = df.isnull()
print(missing_data.head())

for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")

# Replacing "NaN" values of "Weight_kg" column with mean value
avg_weight=df['Weight_kg'].astype('float').mean(axis=0)
df["Weight_kg"].replace(np.nan, avg_weight, inplace=True)

#Replacing the missing values of 'Screen_Size_cm' by the most frequent one
common_screen_size = df['Screen_Size_cm'].value_counts().idxmax()
df["Screen_Size_cm"].replace(np.nan, common_screen_size, inplace=True)

# Fixing the data types
df[["Weight_kg","Screen_Size_cm"]] = df[["Weight_kg","Screen_Size_cm"]].astype("float")

# Data standardization (converting weight from kg to pounds)
df["Weight_kg"] = df["Weight_kg"]*2.205
df.rename(columns={'Weight_kg':'Weight_pounds'}, inplace=True)

# Data standardization (converting screen size from cm to inch)
df["Screen_Size_cm"] = df["Screen_Size_cm"]/2.54
df.rename(columns={'Screen_Size_cm':'Screen_Size_inch'}, inplace=True)

# Data Normalization
df['CPU_frequency'] = df['CPU_frequency']/df['CPU_frequency'].max()

# Data bining (for "Price" column)
bins = np.linspace(min(df["Price"]), max(df["Price"]), 4)
group_names = ['Low', 'Medium', 'High']
df['Price-binned'] = pd.cut(df['Price'], bins, labels=group_names, include_lowest=True)

# Plotting the graph
plt.bar(group_names, df["Price-binned"].value_counts())
plt.xlabel("Price")
plt.ylabel("count")
plt.title("Price bins")
plt.show()

# Setting indicator variables of Screen
dummy_variable_1 = pd.get_dummies(df["Screen"])
dummy_variable_1.rename(columns={'IPS Panel':'Screen-IPS_panel', 'Full HD':'Screen-Full_HD'}, inplace=True)
df = pd.concat([df, dummy_variable_1], axis=1)

# Dropping original column "Screen" from "df"
df.drop("Screen", axis = 1, inplace=True)

print(df.head())