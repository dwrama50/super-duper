# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 23:26:12 2021

@author: dwram
"""

import pandas as pd
import numpy as np
import os,sys
import seaborn as sns
from matplotlib import pyplot as plot
from sklearn.ensemble import RandomForestClassifier

dir = 'E:/loan_vidya/'

train_loan=pd.read_csv(os.path.join(dir,"train.csv"))
train_loan.describe()
train_loan.info()

train_loan.Dependents.replace('+3',3,inplace=True)
train_loan.Loan_Status.replace('N',0,inplace=True)
train_loan.Loan_Status.replace('Y',1,inplace=True)

train_loan.isnull().sum()

train_loan.Gender.fillna(train_loan.Gender.mode()[0],inplace=True)
train_loan.Married.fillna(train_loan.Married.mode()[0],inplace=True)
train_loan.Dependents.fillna(train_loan.Dependents.mode()[0],inplace=True)
train_loan.Self_Employed.fillna(train_loan.Self_Employed.mode()[0],inplace=True)
train_loan.Credit_History.fillna(train_loan.Credit_History.mode()[0],inplace=True)

train_loan.Loan_Amount_Term.value_counts()
train_loan.Loan_Amount_Term.fillna(train_loan.Loan_Amount_Term.mode()[0],inplace=True)

train_loan.LoanAmount.fillna(train_loan.LoanAmount.median(),inplace=True)


train_loan['TotalIncome']=train_loan['ApplicantIncome']+train_loan['CoapplicantIncome']
train_loan['EMI']=train_loan['LoanAmount']/train_loan['Loan_Amount_Term']

r=8.7/100
train_loan['EMI']=train_loan['LoanAmount']*r*1000*((1+r)*train_loan['Loan_Amount_Term']
/(12*((1+r)*train_loan['Loan_Amount_Term']-1)))


train_loan['LoanAmount_log']=np.log(train_loan['LoanAmount']*1000)
train_loan['ApplicantIncome_log']=np.log(train_loan['ApplicantIncome'])
train_loan['CoapplicantIncome_log']=np.log(train_loan['CoapplicantIncome'])
train_loan['TotalIncome_log']=np.log(train_loan['TotalIncome'])
train_loan['EMI_log']=np.log(train_loan['EMI'])
train_loan['BalanceIncome']=train_loan['TotalIncome']-(train_loan['EMI'])

train_loan['Balance Income_log']=train_loan['TotalIncome_log']-(train_loan['EMI_log'])



train_loan.to_csv(os.path.join(dir,"train_loan.csv"),index=False)

Lst=list(zip(train_loan['ApplicantIncome'],train_loan['CoapplicantIncome']))
HasCoapplicant=[]
for i in Lst:
    if i[1]!= 0:
        HasCoapplicant.append(1)
    else:
        HasCoapplicant.append(0)
        
train_loan['HasCoapplicant'] = HasCoapplicant
train_loan['Loan_Amount_Term_log']=np.log(train_loan['Loan_Amount_Term'])

train_loan1=pd.get_dummies(train_loan)


selct_variables2=['Gender_Male','Married_Yes','Dependents_0','Dependents_1',
                 'Education_Graduate',
                 'Self_Employed_No','TotalIncome_log',
                 'Balance Income_log','EMI_log','Loan_Amount_Term_log',
                 'Property_Area_Rural', 'Property_Area_Semiurban',
                 'Credit_History','HasCoapplicant','LoanAmount_log']
len(selct_variables2)

X=train_loan1[selct_variables2]
Y=train_loan1['Loan_Status']
#x_subm=test_data1[selct_variables2]
#train-test split

import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
 
>>> from sklearn.preprocessing import StandardScaler
>>> data = [[0, 0], [0, 0], [1, 1], [1, 1]]
>>> scaler = StandardScaler()
>>> print(scaler.fit(data))
StandardScaler()
>>> print(scaler.mean_)
[0.5 0.5]
>>> print(scaler.transform(data))
 
std_scaler = StandardScaler()
 
std_scaler.fit(train_loan1[selct_variables2])
X =pd.DataFrame( std_scaler.transform(X))
  
train_loan2.shape
train_loan2.to_csv(os.path.join(dir,"train_loanscaled.csv"),index=False)
X=train_loan2[selct_variables2]
Y=train_loan2['Loan_Status']
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


#from sklearn.ensemble import RandomForestClassifier
RFC_est = RandomForestClassifier(max_depth=2, random_state=0)
RFC_est.fit(X_train, y_train)
y_predrf= RFC_est.predict(X_test) 

y_pred_trainrf=RFC_est.predict(X_train)  
cmrf= confusion_matrix(y_test, y_predrf)  
cmrf_train= confusion_matrix(y_train, y_pred_trainrf)  
accuracy_score(y_train, y_pred_trainrf) 
accuracy_score(y_test, y_predrf) 
#82 77.2

#71 64
param_grid = {
    'bootstrap': [True],
    'max_depth': [4,5,6,7,8],
    'max_features': [2,3,4,5,6],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [10,11,12,13,15]
}
# Create a based model
rf_grid = RandomForestClassifier()
# Instantiate the grid search model
grid_rf = GridSearchCV(estimator =rf_grid, param_grid = param_grid, 
                          cv =10, n_jobs = -1, verbose = 2)
grid_rf.fit(X_train, y_train)
grid_rf.score(X_train, y_train)
grid_rf.score(X_test, y_test)
#71.9 65
#84 76
#84.3 76.6
#83.4 77.2
#84.3 76.6
#84.7 78.5
y_submrf=grid_rf.predict(x_subm)
submission = pd.DataFrame({'Loan_ID':test_data['Loan_ID'],'Loan_Status':y_submrf})
submission['Loan_Status'].replace(0,'N',inplace=True)
submission['Loan_Status'].replace(1,'Y',inplace=True)
submission.to_csv(os.path.join(dir,'submissionrf4.csv'),index=False)


submission.to_csv(os.path.join(dir,'submissionxg3.csv'),index=False)
print(grid_rf.cv_results_)
print(grid_rf.best_params_)
print(grid_rf.best_score_) #83.2 #83.9 #84.3
print(grid_rf.best_estimator_)

from sklearn.linear_model import PassiveAggressiveClassifier
model = PassiveAggressiveClassifier(C = 0.4, random_state = 5,fit_intercept=True,)
  
# Fitting model 
model.fit(X_train, y_train)
y_pred_trainpg=model.predict(X_train)  
accuracy_score(y_train, y_pred_trainpg) 

# Making prediction on test set
test_pred = model.predict(X_test)
accuracy_score(y_test, test_pred)
#82.3 76.6



test_loan=pd.read_csv(os.path.join(dir,"test.csv"))
test_loan.describe()
test_loan.info()

test_loan.Dependents.replace('+3',3,inplace=True)
#test_loan.Loan_Status.replace('N',0,inplace=True)
#test_loan.Loan_Status.replace('Y',1,inplace=True)

test_loan.isnull().sum()

test_loan.Gender.fillna(test_loan.Gender.mode()[0],inplace=True)
test_loan.Married.fillna(test_loan.Married.mode()[0],inplace=True)
test_loan.Dependents.fillna(test_loan.Dependents.mode()[0],inplace=True)
test_loan.Self_Employed.fillna(test_loan.Self_Employed.mode()[0],inplace=True)
test_loan.Credit_History.fillna(test_loan.Credit_History.mode()[0],inplace=True)

test_loan.Loan_Amount_Term.value_counts()
test_loan.Loan_Amount_Term.fillna(test_loan.Loan_Amount_Term.mode()[0],inplace=True)

test_loan.LoanAmount.fillna(test_loan.LoanAmount.median(),inplace=True)


test_loan['TotalIncome']=test_loan['ApplicantIncome']+test_loan['CoapplicantIncome']
test_loan['EMI']=test_loan['LoanAmount']/test_loan['Loan_Amount_Term']

r=8.7/100
test_loan['EMI']=test_loan['LoanAmount']*r*1000*((1+r)*test_loan['Loan_Amount_Term']
/(12*((1+r)*test_loan['Loan_Amount_Term']-1)))


test_loan['LoanAmount_log']=np.log(test_loan['LoanAmount']*1000)
test_loan['ApplicantIncome_log']=np.log(test_loan['ApplicantIncome'])
test_loan['CoapplicantIncome_log']=np.log(test_loan['CoapplicantIncome'])
test_loan['TotalIncome_log']=np.log(test_loan['TotalIncome'])
test_loan['EMI_log']=np.log(test_loan['EMI'])
test_loan['BalanceIncome']=test_loan['TotalIncome']-(test_loan['EMI'])

test_loan['Balance Income_log']=test_loan['TotalIncome_log']-(test_loan['EMI_log'])



test_loan.to_csv(os.path.join(dir,"test_loan.csv"),index=False)

Lst=list(zip(test_loan['ApplicantIncome'],test_loan['CoapplicantIncome']))
HasCoapplicant=[]
for i in Lst:
    if i[1]!= 0:
        HasCoapplicant.append(1)
    else:
        HasCoapplicant.append(0)
        
test_loan['HasCoapplicant'] = HasCoapplicant
test_loan['Loan_Amount_Term_log']=np.log(test_loan['Loan_Amount_Term'])

test_loan1=pd.get_dummies(test_loan)

x_subm=test_data1[selct_variables2]
x_subm =pd.DataFrame( std_scaler.transform(x_subm))
y_submrf=grid_rf.predict(x_subm)
submission = pd.DataFrame({'Loan_ID':test_data['Loan_ID'],'Loan_Status':y_submrf})
submission['Loan_Status'].replace(0,'N',inplace=True)
submission['Loan_Status'].replace(1,'Y',inplace=True)
submission.to_csv(os.path.join(dir,'submissionrf5.csv'),index=False)



from sklearn.linear_model import LogisticRegression
LR_Classifier=LogisticRegression()
LR_Classifier.fit(X_train, y_train) 
y_predlr= LR_Classifier.predict(X_test) 
LR_Classifier.predict_proba(X_test)


y_pred_trainlr=LR_Classifier.predict(X_train)  
cm= confusion_matrix(y_test, y_predlr)  
cm_train= confusion_matrix(y_train, y_pred_trainlr)  
accuracy_score(y_train, y_pred_trainlr) 
accuracy_score(y_test, y_predlr) 
y_subm=knn_classifier.predict(X_subm)
y_pre1=knn_classifier.predict(X)
#82 77
#81 77
grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge
logreg=LogisticRegression()
logreg_cv=GridSearchCV(logreg,grid,cv=10)
logreg_cv.fit(X_train,y_train)
logreg_cv.best_score_
y_predlrgr= logreg_cv.predict(X_test) 
y_pred_trainlrgr=logreg_cv.predict(X_train)  
accuracy_score(y_train, y_pred_trainlrgr) 
accuracy_score(y_test, y_predlrgr)