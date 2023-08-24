# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.


6.Compare the graphs and hence we obtained the linear regression for the given datas. 

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:DHANUMALYA D 
Register Number:212222230030  

```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()

df.tail()

X=df.iloc[:,:1].values
X

Y=df.iloc[:,1].values
Y

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

Y_pred

Y_test

plt.scatter(X_train,Y_train,color='purple')
plt.plot(X_train,regressor.predict(X_train),color='gold')
plt.title('Hours vs Scores(Training Set)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()

plt.scatter(X_test,Y_test,color='red')
plt.plot(X_test,regressor.predict(X_test),color='pink')
plt.title('Hours vs Scores(Test Set)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```
## Output:

![ml21](https://github.com/Dhanudhanaraj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119218812/e43d37f5-5c35-46a8-80f7-14643f5978d5)

![ml22](https://github.com/Dhanudhanaraj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119218812/42635a5b-b95e-4575-bd2e-e1c4f06623b5)

![ml23](https://github.com/Dhanudhanaraj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119218812/60ce699e-4b04-4747-82da-163795080db0)

![ml24](https://github.com/Dhanudhanaraj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119218812/1f7290a2-ea16-4278-92f0-c9f47839304d)


![ml25](https://github.com/Dhanudhanaraj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119218812/b8c3f32c-8791-403d-868a-4a8124221a34)


![ml26](https://github.com/Dhanudhanaraj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119218812/0562348b-b72c-489c-a1c8-099588e0fc02)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
