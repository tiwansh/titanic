import pandas as pd
import numpy as nm

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


dataFrame = pd.read_csv('../input/train.csv', header = 0)

#print dataFrame.shape

#print dataFrame

X_train = dataFrame.drop(['PassengerId' , 'Survived', 'Cabin', 'Ticket', 'Embarked', 'Name'],axis=1)
X_train.Sex[X_train.Sex == 'female'] = 1
X_train.Sex[X_train.Sex == 'male'] = 0
X_train['Age'] = X_train['Age'].fillna(0)


#print dataFrame

#print type(X_train)


#Y_train = dataFrame.Survived
Y_train1 = pd.DataFrame(dataFrame,columns=['Survived'])

#print Y_train1.shape
#print type(Y_train1)

#print Y_train.shape
#print Y_train


#---------------------------------Now training sets are ready-----------------------------------------#



'''pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators = 100))

hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'], 'randomforestregressor__max_depth': [None, 5, 3, 1]}

clf = GridSearchCV(pipeline, hyperparameters, cv = 10)

clf.fit(X_train, Y_train)

prediction = clf.predict(X_train)
print accuracy_score(X_train, Y_train)
print r2_score(Y_train, prediction) '''

#Training using Logistic Regression
lr= DecisionTreeClassifier()
lr.fit(X_train, Y_train1)
predictions = lr.predict(X_train)
#print accuracy_score(Y_train1, predictions)
#print confusion_matrix(Y_train1, predictions)

#-----------------The model is trained--------------------#
#now create 

dataFrame2 = pd.read_csv('../input/test.csv',header = 0)

X_validation = dataFrame2.drop(['PassengerId', 'Cabin', 'Ticket', 'Embarked', 'Name'],axis=1)
X_validation.Sex[X_validation.Sex == 'female'] = 1
X_validation.Sex[X_validation.Sex == 'male'] = 0
X_validation = X_validation.fillna(0)
passId = dataFrame2.PassengerId
#Y_validation = dataFrame2.Survived
#Y_validation.fillna(0)

#print X_validation

#pred = clf.predict(X_validation)
#sLength= len(dataFrame2['PassengerId'])
pred = lr.predict(X_validation)

#print pred
predFrame = pd.DataFrame(data = pred, columns = ['Survived'])
passIdFrame = pd.DataFrame(data = passId, columns = ['PassengerId'])
predFrame.Survived = predFrame.Survived.round()
#print predFrame
#print passIdFrame

passIdFrame = passIdFrame.assign(Survived=predFrame.Survived)

#print passIdFrame

passIdFrame.to_csv('solution.csv', encoding= 'utf-8', index = False)


