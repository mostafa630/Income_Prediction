import numpy as np
import scipy as sp 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

#Accuracy measures
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

def   runIncomePrediction(ourData) :
    print(ourData)
    data = pd.read_csv("train_data.csv")
    test = pd.read_csv('test_data.csv')

    lbl = LabelEncoder()

    """""
    Checking the skewed data(most of the data was 0 in capital-gain and capital-loss)

        #count =(data['capital-gain']==0).sum()
        #print(count)

        #data.isin([" ?"]).sum()
        #count =(data['capital-loss']==0).sum()
        #print(count)

    #education_count = data.groupby('education-num').size().reset_index(name='count')
    #print(education_count)


    """""

    """Preprocessing"""

    #convert any question mark to null

    data=data.replace(' ?' , np.NaN)
    test=test.replace(' ?' , np.NaN)

    # solve problem of feature selection

    data.drop('fnlwgt',axis=1 ,inplace=True)
    data.drop('capital-gain',axis=1 ,inplace=True)
    data.drop('capital-loss',axis=1 ,inplace=True)
    data.drop('relationship',axis=1 ,inplace=True)
    data.drop('race',axis=1 ,inplace=True)

    test.drop('fnlwgt',axis=1 ,inplace=True)
    test.drop('capital-gain',axis=1 ,inplace=True)
    test.drop('capital-loss',axis=1 ,inplace=True)
    test.drop('relationship',axis=1 ,inplace=True)
    test.drop('race',axis=1,inplace=True)


    #solve problem of nulls

    #data.isnull().sum()          Checking how many nulls do we have

    data['workclass'].fillna(data['workclass'].mode()[0],inplace=True) 
    data['occupation'].fillna(data['occupation'].mode()[0],inplace=True) 
    data['native-country'].fillna(data['native-country'].mode()[0],inplace=True)

    test['workclass'].fillna(data['workclass'].mode()[0],inplace=True) 
    test['occupation'].fillna(data['occupation'].mode()[0],inplace=True) 
    test['native-country'].fillna(data['native-country'].mode()[0],inplace=True) 


    #solve problem of outliers

    # education num column
    edu_q1=9
    edu_q3=12
    edu_IQR=edu_q3-edu_q1
    minconvert=math.ceil(edu_q1-1.5*edu_IQR)
    maxconvert=math.floor(edu_q3+1.5*edu_IQR)
    data.loc[data['education-num']<minconvert,'education-num']=minconvert
    data.loc[data['education-num']>maxconvert,'education-num']=maxconvert

    #hours per week 
    hours_q1=40
    hours_q3=45
    hours_IQR=hours_q3-hours_q1
    minconvert2=math.ceil(hours_q1-1.5*hours_IQR)
    maxconvert2=math.floor(hours_q3+1.5*hours_IQR)
    data.loc[data['hours-per-week']<minconvert2,'hours-per-week']=minconvert2
    data.loc[data['hours-per-week']>maxconvert2,'hours-per-week']=maxconvert2


    # education num column
    edu_q1=9
    edu_q3=12
    edu_IQR=edu_q3-edu_q1
    minconvert=math.ceil(edu_q1-1.5*edu_IQR)
    maxconvert=math.floor(edu_q3+1.5*edu_IQR)
    test.loc[test['education-num']<minconvert,'education-num']=minconvert
    test.loc[test['education-num']>maxconvert,'education-num']=maxconvert

    #hours per week 
    hours_q1=40
    hours_q3=45
    hours_IQR=hours_q3-hours_q1
    minconvert2=math.ceil(hours_q1-1.5*hours_IQR)
    maxconvert2=math.floor(hours_q3+1.5*hours_IQR)
    test.loc[test['hours-per-week']<minconvert2,'hours-per-week']=minconvert2
    test.loc[test['hours-per-week']>maxconvert2,'hours-per-week']=maxconvert2

    "#############################################################################"

    #Feature Selection

    x_train = data.drop(['Income '],axis=1)
    y_train = data['Income ']

    x_test = test.drop(['Income '], axis=1)
    y_test = test['Income ']

    def Feature_Encoder(X,cols):
        for c in cols:
            X[c] = lbl.fit_transform(X[c])
        return X

    cols = ("workclass","education","marital-status","occupation","sex","native-country")

    x_train = Feature_Encoder(x_train,cols)
    y_train = lbl.fit_transform(y_train)

    x_test = Feature_Encoder(x_test,cols)
    y_test = lbl.fit_transform(y_test)

    #Scaling our features to be in range
    sc_x = StandardScaler()
    x_train = sc_x.fit_transform(x_train)
    x_test = sc_x.transform(x_test)
    

    """MODELLING"""
    
    #Logistic Regression
    classifier = LogisticRegression(random_state = 44)
    classifier.fit(x_train, y_train)
    print(x_test[1].reshape(1,-1))
    y_pred = classifier.predict(x_test[1].reshape(1,-1))
    ''''
    report = classification_report(y_test,y_pred)
    print(report,sep='\n')

    #Decision Tree
    from sklearn.model_selection import GridSearchCV

    # Step 2: Hyperparameter tuning
    param_grid = {
        'max_depth': [None, 3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 3, 5],
        'max_features': [None, 'sqrt', 'log2']
    }

    classifier = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(classifier, param_grid, cv=5)
    grid_search.fit(x_train, y_train)
    best_params = grid_search.best_params_

    print(best_params)

    # Step 3: Initialize and train the decision tree model
    classifier = DecisionTreeClassifier(**best_params)
    classifier.fit(x_train, y_train)

    # Step 4: Make predictions on the test set
    y_pred = classifier.predict(x_test)

    #Decision Tree without grid search
    # classifier = DecisionTreeClassifier(random_state = 44)
    # classifier = classifier.fit(x_train,y_train)
    # y_pred = classifier.predict(x_test)
    report = classification_report(y_test,y_pred)

    print(report,end='\n')
    '''
    '''
    #SVM Tree
    classifier = SVC(C=10.0, random_state=44, kernel='rbf')
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    report = classification_report(y_test,y_pred)
    
    print(report,sep='\n')
    '''
    # ourData=lbl.fit_transform(ourData)
    # print(ourData)
    
    #guipredict =classifier.predict(x_test[1])
    print(y_pred)
    guipredict=lbl.inverse_transform(y_pred)
    #print(x_test[1].reshape(1,-1))
    print(guipredict)
    
    return guipredict

