from statistics import correlation
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
print("Enter house data")
house=sklearn.datasets.load_boston()
print(house)
house_dataframe=pd.DataFrame(house.data,columns=house.feature_names)
print(house_dataframe)
house_dataframe.head()
print("Enter the price")
house_dataframe['price']=house.target
print(house_dataframe)
house_dataframe.head()
house_dataframe.isnull().sum()
print(house_dataframe.isnull().sum())
house_dataframe.describe()
print(house_dataframe.describe())
correlation=house_dataframe.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation,cbar=True,square=True, fmt='.1f',annot=True,annot_kws={'size':8},cmap='Blues') 
plt.show()
X=house_dataframe.drop(['price'],axis=1)
Y=house_dataframe['price']
print(X,Y)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)
print(X.shape,X_test.shape,X_train.shape,)
M=XGBRegressor()
M.fit(X_train,Y_train)
training_data_prediction=M.predict(X_train)
print(training_data_prediction)
s_1=metrics.r2_score(Y_train,training_data_prediction)
s_2=metrics.mean_absolute_error(Y_train,training_data_prediction)
print("R squred error :",s_1)
print("Mean absolute error :",s_2)
test_data_prediction=M.predict(X_test)
print(test_data_prediction)
s_3=metrics.r2_score(Y_test,test_data_prediction)
s_4=metrics.mean_absolute_error(Y_test,test_data_prediction)
print("R squred error :",s_3)
print("Mean absolute error :",s_4)
plt.scatter(Y_train,training_data_prediction)
plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.title("Actual price vs Predicted price")
plt.show()
