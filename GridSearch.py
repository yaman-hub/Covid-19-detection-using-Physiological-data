import pandas as pd
import numpy as np
from sklearn.ensemble import StackingClassifier , RandomForestClassifier , ExtraTreesClassifier , AdaBoostClassifier ,  HistGradientBoostingClassifier , GradientBoostingClassifier
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
from time import time
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, precision_score , recall_score
df=pd.read_csv("10-8-22.csv",header=0 )

HR='HR'
Steps='Steps'
# D_N='D_N'
RRinterval='RRinterval'
HR_Diff='HR_Diff'
label_sym='label_sym'
df.columns = [HR, Steps, RRinterval, HR_Diff,label_sym]
###X.columns = [HR, Steps, RRinterval, HR_Diff,D_N]
###Y.columns = [label]
#print(df.head())

df = df.sample(frac = 1)
X = df.iloc[:,0:4]
Y = df.iloc[:,4]

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=0)

# parameters = {'n_estimators': [50,80],
#     'learning_rate': [1.0, 0.5,0.01],
#               'max_depth': [6,8],
#               'min_samples_split': [500, 1000]
#               }
#
#
# GB_model = GradientBoostingClassifier(random_state = 123)
# GS_model = GridSearchCV(GB_model , parameters , cv = 5, n_jobs=1)
#
# GS_model.fit(X_train, y_train)
# from sklearn.metrics import confusion_matrix, accuracy_score
# y_pred = GS_model.predict(X_val)
#
# cm = confusion_matrix(y_val, y_pred)
# print(cm)
# print(accuracy_score(y_val, y_pred))
#
# from sklearn.metrics import classification_report
# print(classification_report(y_val, y_pred))
#
from sklearn.metrics import precision_recall_fscore_support
# print(precision_recall_fscore_support(y_val, y_pred))

print("************************")
#
# estimators = [200 , 300 ,400,500 ]#, 600 , 650 , 700 , 750 , 800 , 1000]
# for est in estimators:
#     xgb_t= XGBClassifier(booster='gbtree', objective='binary:logistic', max_depth= 10, learning_rate=0.1, gamma = 0.15, n_estimators=est, random_state=2, n_jobs=-1)
#     xgb_t.fit(X_train , y_train)
#     print(est)
#     print('XGBClassifier')
#
#     #print('XGBClassifier Results for '+ est + ' number of estimators: ')
#     y_pred = xgb_t.predict(X_val)
#     cm = confusion_matrix(y_val, y_pred)
#     print(cm)
#     print(accuracy_score(y_val, y_pred))
#     print(classification_report(y_val, y_pred))
#     print(precision_recall_fscore_support(y_val, y_pred))
#
#
#     GBModel= GradientBoostingClassifier( max_depth= 10, learning_rate=0.1, n_estimators=est, random_state=2)
#     GBModel.fit(X_train , y_train)
#     print('Gardient boosting: ')
#
#
#     y_pred_GB = GBModel.predict(X_val)
#     cm = confusion_matrix(y_val, y_pred_GB)
#     print(cm)
#     print(accuracy_score(y_val, y_pred_GB))
#     print(classification_report(y_val, y_pred_GB))
#     print(precision_recall_fscore_support(y_val, y_pred_GB))

xgb_t= XGBClassifier(booster='gbtree', objective='binary:logistic', max_depth= 10, learning_rate=0.1, gamma = 0.15, n_estimators=200, random_state=2, n_jobs=-1)
xgb_t.fit(X_train , y_train)


#print('XGBClassifier Results for '+ est + ' number of estimators: ')
y_pred = xgb_t.predict(X_val)
cm = confusion_matrix(y_val, y_pred)
print(cm)
print(accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))
print(precision_recall_fscore_support(y_val, y_pred))




print(" **************UOS Result **************")
df=pd.read_csv("8-8-22UOS.csv",header=0 )

HR='HR'
Steps='Steps'
# D_N='D_N'
RRinterval='RRinterval'
HR_Diff='HR_Diff'
label_sym='label_sym'
df.columns = [HR, Steps, RRinterval, HR_Diff,label_sym]
###X.columns = [HR, Steps, RRinterval, HR_Diff,D_N]
###Y.columns = [label]
#print(df.head())

df = df.sample(frac = 1)
X_UOS = df.iloc[:,0:4]
Y_UOS = df.iloc[:,4]




#print('XGBClassifier Results for '+ est + ' number of estimators: ')
y_pred = xgb_t.predict(X_UOS)
cm = confusion_matrix(Y_UOS, y_pred)
print(cm)
print(accuracy_score(Y_UOS, y_pred))
print(classification_report(Y_UOS, y_pred))
print(precision_recall_fscore_support(Y_UOS, y_pred))


