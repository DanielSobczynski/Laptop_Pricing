from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures

filepath = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod2.csv'
df = pd.read_csv(filepath, header=0)

# print(df.head().to_string())
df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1, inplace=True)
print(df.head().to_string())

#Cross validation stage for improving the model

# Dividing the dataset into dependent and independent parameters
y_data = df['Price']
x_data = df.drop('Price',axis=1)

# Splitting the data set into training and testing subsets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10, random_state=1)
#print("number of test samples :", x_test.shape[0])
#print("number of training samples:",x_train.shape[0])

#  Creating linear regression model using "CPU_frequency" parameter and checking the R^2 value
lrm=LinearRegression()
lrm.fit(x_train[['CPU_frequency']], y_train)
print(lrm.score(x_test[['CPU_frequency']], y_test))
print(lrm.score(x_train[['CPU_frequency']], y_train))

# 4-fold cross validation on the model
Rcross = cross_val_score(lrm, x_data[['CPU_frequency']], y_data, cv=4)
print("The mean of the folds are", Rcross.mean(), "and the standard deviation is" , Rcross.std())

# Splitting the dataset into training and testing components
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.5, random_state=0)

# Identifying the point of overfitting the model on the parameter "CPU_frequency"
plrm=LinearRegression()
Rsqu_test = []
order = [1, 2, 3, 4, 5]
for n in order:
    plrm = PolynomialFeatures(degree=n)
    x_train_plrm = plrm.fit_transform(x_train[['CPU_frequency']])
    x_test_plrm = plrm.fit_transform(x_test[['CPU_frequency']])
    plrm.fit(x_train_plrm, y_train)
    Rsqu_test.append(plrm.score(x_test_plrm, y_test))

# Plotting the values of R^2 scores
plt.plot(order, Rsqu_test)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using Test Data')

# Creating a polynomial feature model with different parameters
pmlrm=PolynomialFeatures(degree=2)
x_train_pmlrm=pmlrm.fit_transform(x_train[['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core', 'OS', 'GPU', 'Category']])
x_test_pmlrm=pmlrm.fit_transform(x_test[['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core', 'OS', 'GPU', 'Category']])

# Creating a Ridge Regression model and evaluating it using values of the hyperparameter alpha ranging from 0.001 to 1 with increments of 0.001
Rsqu_test = []
Rsqu_train = []
Alpha = np.arange(0.001,1,0.001)
pbar = tqdm(Alpha)

for alpha in pbar:
    RidgeModel = Ridge(alpha=alpha)
    RidgeModel.fit(x_train_pmlrm, y_train)
    test_score, train_score = RidgeModel.score(x_test_pmlrm, y_test), RidgeModel.score(x_train_pmlrm, y_train)
    pbar.set_postfix({"Test Score": test_score, "Train Score": train_score})
    Rsqu_test.append(test_score)
    Rsqu_train.append(train_score)

# Plotting the R^2 values in the context of alpha values
plt.figure(figsize=(10, 6))
plt.plot(Alpha, Rsqu_test, label='validation data')
plt.plot(Alpha, Rsqu_train, 'r', label='training Data')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.ylim(0, 1)
plt.legend()
plt.show()

# Using GridSearchCV to identify the value of alpha for which the model performs best.
parameters1= [{'alpha': [0.0001,0.001,0.01, 0.1, 1, 10]}]
RR=Ridge()
Grid1 = GridSearchCV(RR, parameters1,cv=4)
Grid1.fit(x_train[['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core', 'OS', 'GPU', 'Category']], y_train)
BestRR=Grid1.best_estimator_
print(BestRR.score(x_test[['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core','OS','GPU','Category']], y_test))




