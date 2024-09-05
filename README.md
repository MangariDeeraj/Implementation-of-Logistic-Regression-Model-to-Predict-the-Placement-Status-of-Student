# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.

2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

3.Import LabelEncoder and encode the dataset.

4.Import LogisticRegression from sklearn and apply the model on the dataset.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.Apply new unknown values
## Program:
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

Developed by:MANGARI DEERAJ

RegisterNumber:  212223100031
*/
```
import pandas as pd
data=pd.read_csv('/content/Placement_Data (1).csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1) #removes the specified row or column.
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1


x=data1.iloc[:,:-1]
x

y=data1['status']
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size =0.2,random_sta

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='liblinear')# A library for large linear classification
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)


lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
PLACEMENT DATA:

![267522570-f40d603b-0f61-48da-b1e1-87c31d03bd70](https://github.com/user-attachments/assets/6916efa9-2ef3-45a4-b6e3-66428acb5a69)

SALARY DATA:
![267525779-49f917bf-bce0-408e-8f94-d0bc79548d15](https://github.com/user-attachments/assets/db43f248-05fe-42ed-a63c-3589a171e72f)

CHECKING THE NULL() FUNCTION:]

![267522991-a22dfd91-b344-4266-91a0-2cd757a3623e](https://github.com/user-attachments/assets/eff1238a-8a7a-46bc-a870-b4d68c331d70)

DATA DUPLICATE:

![267523098-861c970b-ffeb-4246-af60-913e8388dea4](https://github.com/user-attachments/assets/6e931416-213a-43c4-8e27-2920b912da10)

PRINT DATA:

![267523282-d30f3eca-3148-47df-85b8-5896fb036250](https://github.com/user-attachments/assets/22f2a719-dca4-445d-bfcb-2f67a6f727fa)

DATA_STATUS:

![267523389-2f05d218-3b79-4c0e-a69b-407d74f9711d](https://github.com/user-attachments/assets/6e791579-48c6-4a73-ae67-f4597de45563)

DATA_STATUS:

![267523572-cea77cd5-ea38-4687-9f1a-a42ce7776821](https://github.com/user-attachments/assets/0de0ba4a-596d-4d59-8418-bc4ded220ac2)

Y_PREDICTION ARRAY:

![267524139-b6773f22-9c30-43b5-a3db-ca652e7b0009](https://github.com/user-attachments/assets/49f983c5-766a-4f5b-b330-0e84dd711abf)

ACCURACY VALUE:

![267524226-0f00fd95-00ee-483c-b333-fc18162a438e](https://github.com/user-attachments/assets/8b7a5d13-daa7-4cc6-8915-0b5a016e4e0b)

CONFUSION ARRAY:

![267524290-54627a5d-0db9-440c-a5d5-3f4831edeb03](https://github.com/user-attachments/assets/645297f4-62c6-49c0-96b0-83ada5f7de96)

CLASSIFICATION REPORT:

![267524355-97e25d65-6436-41c0-9a41-64cd9c457972](https://github.com/user-attachments/assets/b69b6ca2-d8cd-427f-83a2-a34b24ed8681)

PREDICTION OF LR:

![267524441-b7be4b3d-54cd-4472-95cc-189b0f9248a0](https://github.com/user-attachments/assets/fe443926-45d4-4b16-8af0-8058db00f405)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
