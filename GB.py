import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier

data=pd.read_csv('10-8-22.csv')

# separate our target and features variables
target = np.array(data['label_sym'])
# Remove the target from the features
# axis 1 refers to the columns
features= data.drop('label_sym', axis = 1)
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(features, target, test_size=0.30, random_state=8)

x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.50, random_state=8)

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score


def modelfit(alg, X_train, Y_train, performCV=True, printFeatureImportance=True, cv_folds=5):
    # Fit the algorithm on the data
    #alg.fit(dtrain[predictors], dtrain['Disbursed'])
    alg.fit(X_train, Y_train)
    #predictors= list(X_train.columns)

    # Predict training set:
    dtrain_predictions = alg.predict(X_train)
    dtrain_predprob = alg.predict_proba(X_train)[:, 1]

    # Perform cross-validation:
    if performCV:
        cv_score = cross_val_score(alg, X_train, Y_train, cv=cv_folds,
                                                    scoring='roc_auc')

    # Print model report:
    print ("\nModel Report")
    print ("Accuracy : %.4g" % accuracy_score(Y_train, dtrain_predictions))
    from sklearn.metrics import classification_report
    print(classification_report(Y_train, dtrain_predictions))

    from sklearn.metrics import precision_recall_fscore_support
    print(precision_recall_fscore_support(Y_train, dtrain_predictions))
    # print
    # "AUC Score (Train): %f" % roc_auc_score(Y_train, dtrain_predprob)

    if performCV:
        print
        "CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (
        np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score))

    # Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')


gbm0 = GradientBoostingClassifier(random_state=10)
modelfit(gbm0, x_train, y_train)





############## UOS data :
print('***************************************************************')
print('Result on UOS dataset: ')

data=pd.read_csv('8-8-22UOS.csv')

# separate our target and features variables
target = np.array(data['label_sym'])
# Remove the target from the features
# axis 1 refers to the columns
features= data.drop('label_sym', axis = 1)
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)





