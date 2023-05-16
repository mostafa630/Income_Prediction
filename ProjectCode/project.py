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
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

#Accuracy measures
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import classification_report

#Models
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def runIncomePrediction(ourData):
    data = pd.read_csv("train_data.csv")
    test = pd.read_csv('test_data.csv')

    data=data.replace(' ?' , np.NaN) #convert any question mark to null
    test=test.replace(' ?' , np.NaN)

    """
    solve problem of feature selection
    """

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
    data['workclass'].fillna(data['workclass'].mode()[0],inplace=True) # test sucseesded
    data['occupation'].fillna(data['occupation'].mode()[0],inplace=True) # test sucseeded
    data['native-country'].fillna(data['native-country'].mode()[0],inplace=True) # test sucseeded



    sns.boxplot(data=data ,palette='rainbow' ,orient='h')#Before handling the outliers


    """
    solve problem of outliers
    """
    #age column
    age_q1=28
    age_q3=48
    age_IQR=age_q3-age_q1
    minconvert=math.ceil(age_q1-1.5*age_IQR)
    maxconvert=math.floor(age_q3+1.5*age_IQR)
    data.loc[data['age']<minconvert,'age']=minconvert
    data.loc[data['age']>maxconvert,'age']=maxconvert

    # education num column
    edu_q1=9
    edu_q3=12
    edu_IQR=edu_q3-edu_q1
    minconvert2=math.ceil(edu_q1-1.5*edu_IQR)
    maxconvert2=math.floor(edu_q3+1.5*edu_IQR)
    data.loc[data['education-num']<minconvert2,'education-num']=minconvert2
    data.loc[data['education-num']>maxconvert2,'education-num']=maxconvert2

    #hours per week 
    hours_q1=40
    hours_q3=45
    hours_IQR=hours_q3-hours_q1
    minconvert3=math.ceil(hours_q1-1.5*hours_IQR)
    maxconvert3=math.floor(hours_q3+1.5*hours_IQR)
    data.loc[data['hours-per-week']<minconvert3,'hours-per-week']=minconvert3
    data.loc[data['hours-per-week']>maxconvert3,'hours-per-week']=maxconvert3

    sns.boxplot(data=data ,palette='rainbow' ,orient='h')#After handling the outliers

    """Encoding categorical features to numerical ones"""
    cols = ("workclass","education","marital-status","occupation","sex","native-country")
    lbl = LabelEncoder()

    def Feature_Encoder(X,cols):
        for c in cols:
            X[c] = lbl.fit_transform(X[c])
        return X

    x_train = data.drop(['Income '],axis=1)
    y_train = data['Income ']

    x_train = Feature_Encoder(x_train,cols)
    y_train = lbl.fit_transform(y_train)


    x_test = test.drop(['Income '], axis=1)
    y_test = test['Income ']

    x_test = Feature_Encoder(x_test,cols)
    y_test = lbl.fit_transform(y_test)


    sc_x = StandardScaler()
    x_train = sc_x.fit_transform(x_train)
    x_test = sc_x.fit_transform(x_test)

    """Modelling"""
    if(len(ourData) == 0): 
        #Logistic Regression
        classifier = LogisticRegression(random_state = 44)
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)

        cm = confusion_matrix(y_test, y_pred)

        """
        true negatives 00
        false negatives is 10
        true positives is 11
        and false positives is 01
        """
        ConfusionMatrixDisplay(confusion_matrix=cm,
                                    display_labels=classifier.classes_).plot()

        report = classification_report(y_test,y_pred)

        print(report,sep='\n')

        #Logistic Regression
        from sklearn.model_selection import GridSearchCV

        # Decision Tree Hyperparameter tuning
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


        classifier = DecisionTreeClassifier(**best_params)
        classifier.fit(x_train, y_train)

        y_pred = classifier.predict(x_test)

        cm = confusion_matrix(y_test, y_pred)

        """
        true negatives 00
        false negatives is 10
        true positives is 11
        and false positives is 01
        """

        ConfusionMatrixDisplay(confusion_matrix=cm,
                                    display_labels=classifier.classes_).plot()
        report = classification_report(y_test,y_pred)

        print(report,end='\n')

        #SVM
        classifier = SVC(C=10.0, random_state=44, kernel='rbf')
        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)

        cm = confusion_matrix(y_test, y_pred)

        """
        true negatives 00
        false negatives is 10
        true positives is 11
        and false positives is 01
        """

        ConfusionMatrixDisplay(confusion_matrix=cm,
                                    display_labels=classifier.classes_).plot()
        report = classification_report(y_test,y_pred)

        print(report,sep='\n')

    #Random Forest
    classifier = RandomForestClassifier(n_estimators=100, max_depth=10)
    classifier.fit(x_train,y_train)
    print(x_test[1].reshape(1,-1))
    y_pred = classifier.predict(x_test[1].reshape(1,-1))

    #cm = confusion_matrix(y_test, y_pred)

    """
    true negatives 00
    false negatives is 10
    true positives is 11
    and false positives is 01
    """

    #ConfusionMatrixDisplay(confusion_matrix=cm,
                                #display_labels=classifier.classes_).plot()
    
    #ourData = np.array(ourData)

    #ourData = lbl.fit_transform(ourData)

    #ourData = ourData.reshape(-1,1)
    
    #scaled_data = sc_x.fit_transform(ourData)

    #scaled_data = scaled_data.reshape(1,-1)
    
    #y_pred = classifier.predict(scaled_data)

    #report = classification_report(y_test,y_pred)
    #print(report,sep='\n')

    gui_pred = lbl.inverse_transform(y_pred)
    print(gui_pred)
    return gui_pred

