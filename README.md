# Tabular Data Project: Heart Disease Prediction

In this project, we will apply basic machine learning methods to predict whether an individual is prone to heart disease or not, 
based on the Cleveland Heart Disease dataset from the UCI Machine Learning Repository. The Cleveland dataset comprises 14 features, 
including age, gender, chest-pain type, resting blood pressure, serum cholesterol level, fasting blood sugar, resting electrocardiogram results, 
maximum heart rate achieved, exercise-induced angina, ST depression induced by exercise relative to rest, peak exercise ST segment, the number of 
major vessels illuminated by fluoroscopy (ranging from 0 to 3), thalassemia presence, and heart disease diagnosis (0 representing no heart disease 
and 1, 2, 3, 4 representing varying degrees of heart disease). The Cleveland dataset consists of 303 samples, and these 14 features.
![image](https://github.com/Buitruongvi/Decision_Tree_and_Its_Variances_Project_btvir/assets/49474873/0544cc3a-393c-4955-b135-3e828979d728)

We will utilize various machine learning algorithms to predict whether a patient is susceptible to heart disease or not. These algorithms include naive Bayes, 
k-nearest neighbors (KNN), decision tree, random forest, AdaBoost, gradient boost, XGBoost, and support vector machine (SVM).

## Load data
```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns

# Bai tap 1
df = pd.read_csv('/content/cleveland.csv', header = None)
df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol',
              'fbs', 'restecg', 'thalach', 'exang',
              'oldpeak', 'slope', 'ca', 'thal', 'target']
df['target'] = df.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
df['thal'] = df.thal.fillna(df.thal.mean())
df['ca'] = df.ca.fillna(df.ca.mean())
```
## 1. KNN
Use the KNN algorithm to predict whether a patient is at risk of heart disease or not, using the following parameters: `n_neighbors=5`, `weights='uniform'`, `algorithm='auto'`, `leaf_size=30`, `p=2`, `metric='minkowski'`.
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn_classifier = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski')
knn_classifier.fit(X_train, y_train)

y_train_pred = knn_classifier.predict(X_train)
y_test_pred = knn_classifier.predict(X_test)

cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)
print()
accuracy_for_train = np.round((cm_train [0][0] + cm_train [1][1])/len(y_train) ,2)
accuracy_for_test = np.round ((cm_test [0][0] + cm_test [1][1])/len(y_test) ,2)
print('Accuracy for training set for KNeighborsClassifier = {}'.format(accuracy_for_train))
print('Accuracy for test set for KNeighborsClassifier = {}'.format(accuracy_for_test))
```
```
Accuracy for training set for KNeighborsClassifier = 0.76
Accuracy for test set for KNeighborsClassifier = 0.69
```
## 2. SVM
```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC

X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

SVM_classifier = SVC(kernel = 'rbf', random_state=42)
SVM_classifier.fit(X_train , y_train)

y_train_pred = SVM_classifier.predict(X_train)
y_test_pred = SVM_classifier.predict(X_test)

cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)
print()
accuracy_for_train = np.round((cm_train [0][0] + cm_train [1][1])/len(y_train) ,2)
accuracy_for_test = np.round ((cm_test [0][0] + cm_test [1][1])/len(y_test) ,2)
print('Accuracy for training set for SVMClassifier = {}'.format(accuracy_for_train))
print('Accuracy for test set for SVMClassifier = {}'.format(accuracy_for_test))
```
```
Accuracy for training set for SVMClassifier = 0.66
Accuracy for test set for SVMClassifier = 0.67
```
## 3. Naive Bayes
```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB

X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb_classifier = GaussianNB()
nb_classifier.fit(X_train , y_train)

y_train_pred = nb_classifier.predict(X_train)
y_test_pred = nb_classifier.predict(X_test)

cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)
print()
accuracy_for_train = np.round((cm_train [0][0] + cm_train [1][1])/len(y_train) ,2)
accuracy_for_test = np.round ((cm_test [0][0] + cm_test [1][1])/len(y_test) ,2)
print('Accuracy for training set for NBClassifier = {}'.format(accuracy_for_train))
print('Accuracy for test set for NBClassifier = {}'.format(accuracy_for_test))
```
```
Accuracy for training set for NBClassifier = 0.85
Accuracy for test set for NBClassifier = 0.84
```
...
## Ensemble Stacking
We will integrate them using stacking techniques.
![image](https://github.com/Buitruongvi/Decision_Tree_and_Its_Variances_Project_btvir/assets/49474873/d56690b9-2762-4283-b514-a8e5a5123d83)
```python
from sklearn.ensemble import StackingClassifier
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

base_classifiers = [
    ('Decision Tree', DecisionTreeClassifier(random_state=42)),
    ('Random Forest', RandomForestClassifier(random_state=42)),
    ('K-Nearest Neighbors', KNeighborsClassifier()),
    ('Gradient Boosting', GradientBoostingClassifier(random_state=42)),
    ('Support Vector Classifier', SVC(kernel='rbf', random_state=42)),
    ('AdaBoost', AdaBoostClassifier(random_state=42))
]

stacking_classifier = StackingClassifier(
    estimators=base_classifiers,
    final_estimator=XGBClassifier()  # You can change the final estimator as needed
)
stacking_classifier.fit(X_train, y_train)

y_train_pred = stacking_classifier.predict(X_train)
y_test_pred = stacking_classifier.predict(X_test)

cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)
print()
accuracy_for_train = np.round((cm_train [0][0] + cm_train [1][1])/len(y_train) ,2)
accuracy_for_test = np.round ((cm_test [0][0] + cm_test [1][1])/len(y_test) ,2)
print('Accuracy for training set for Stacking = {}'.format(accuracy_for_train))
print('Accuracy for test set for Stacking = {}'.format(accuracy_for_test))
```
```
Accuracy for training set for Stacking = 0.92
Accuracy for test set for Stacking = 0.9
```

# Reference
