# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the required packages.
2.Import the dataset to operate on.
3.Split the dataset.
4.Predict the required output.
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by:Manojkumar.K
RegisterNumber:24900281 
*/

Program to implement the SVM For Spam Mail Detection..
Developed by: Naadira Sahar N
RegisterNumber: 212221220034 

print("Result Output:")
import chardet 
file='/content/spam.csv'
with open(file, 'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("/content/spam.csv",encoding='Windows-1252')

print("data head:")
data.head()

print("data info:")
data.info()

print("data isnull:")
data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

print("y_prediction  value:")
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)
y_pred

print("Accuracy Value:")
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy

```

## Output:
![SVM For Spam Mail Detection](sam.png)

![Screenshot 2024-12-25 210054](https://github.com/user-attachments/assets/db7cc8f9-17c7-4b17-a374-dec5de64e84b)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
