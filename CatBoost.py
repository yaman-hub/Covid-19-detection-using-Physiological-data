from catboost import CatBoostClassifier, Pool
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, precision_score , recall_score

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

df=pd.read_csv("10-8-22.csv",header=0 )

HR ='HR'
Steps='Steps'
# D_N='D_N'
RRinterval='RRinterval'
HR_Diff='HR_Diff'
label_sym='label_sym'
df.columns = [HR, Steps, RRinterval, HR_Diff,label_sym]
###X.columns = [HR, Steps, RRinterval, HR_Diff,D_N]
###Y.columns = [label]
#print(df.head())
##print(df)
# scaler = StandardScaler()
# df_Scaled=scaler.fit(df)
#df_Scaled=df_Scaled.transform(df)
#df_Scaled=pd.DataFrame(df_Scaled)


# scalerMM = MinMaxScaler()
# df_minmax=scalerMM.fit(df)

# df = df.sample(frac = 1)
# X = df.iloc[:,0:4]
# Y = df.iloc[:,4]

df_Scaled = df.sample(frac = 1)
#print(df_Scaled)
X = df.iloc[:,0:4]
Y = df.iloc[:,4]

X=pd.DataFrame(X)
norm = StandardScaler().fit(X)

X_normalized=norm.transform(X)
X_normalized=pd.DataFrame(X_normalized)
X_normalized.to_csv('x normalized.csv')
X.to_csv('x normal.csv')

# df = df_minmax.sample(frac = 1)
# X = df_minmax.iloc[:,0:4]
# Y = df_minmax.iloc[:,4]


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=0)
X_val.to_csv('X validation_stanford.csv')
y_val.to_csv('Y actual.csv')


#print('First fig is for x vs actual')

# sns.scatterplot(x=X_val.Steps, y=y_val)
# plt.xlabel('Steps')
# plt.ylabel('Actual Output')
# plt.title('Steps vs Actual Output')
# plt.show()


# Classifiers
from catboost import CatBoostClassifier, Pool

# Model selection
from sklearn.model_selection import StratifiedKFold

# Metrics
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import make_scorer

best_params = {'bagging_temperature': 0.6,
              'border_count': 200,
              'depth': 8,
              'iterations': 350,
              'l2_leaf_reg': 30,
              'learning_rate': 0.30,
              'random_strength': 0.01,
              'scale_pos_weight': 0.48}

catb = CatBoostClassifier(#best_params,
                          loss_function='MultiClass',
                          eval_metric = 'AUC',
                          nan_mode='Min',
                          thread_count=2,
                          verbose = False)

catb.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = catb.predict(X_val)
y_pred=pd.DataFrame(y_pred)
y_pred.to_csv('Y prediction Stanford.csv')
cm = confusion_matrix(y_val, y_pred)
print('this is the CM for the Stanford dataset 85K')
print(cm)
accuracy_score(y_val, y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_val, y_pred))

from sklearn.metrics import precision_recall_fscore_support
print(precision_recall_fscore_support(y_val, y_pred))

cf_matrix = confusion_matrix(y_val, y_pred)
print(cf_matrix)

import seaborn as sns
sns.heatmap(cf_matrix, annot=True, fmt='g')
plt.show()



y_pred=y_pred.values.flatten()



# # print('second fig is for x vs prediction')
# sns.scatterplot(x=X_val.Steps, y=y_pred)
# plt.xlabel('Steps')
# plt.ylabel('Predicted Output')
# plt.title('Steps vs Predicted Output')
# plt.show()

#########UOS dataset

UOS=pd.read_csv('UOS-29-9-22.csv')
HR='HR'
Steps='Steps'
# D_N='D_N'
RRinterval='RRinterval'
HR_Diff = 'HR_Diff'
label_sym = 'label_sym'
UOS.columns = [HR, Steps, RRinterval, HR_Diff,label_sym]


# df_Scaled_UOS=df_Scaled_UOS.transform(df)
# df_Scaled_UOS=pd.DataFrame(df_Scaled_UOS)


df_Scaled_UOS = UOS.sample(frac = 1)
X_UOS = UOS.iloc[:,0:4]
Y_UOS = UOS.iloc[:,4]
norm_UOS = StandardScaler().fit(X_UOS)

X_UOS_norm=norm_UOS.transform(X_UOS)
X_UOS_norm=pd.DataFrame(X_UOS_norm)

Y_UOS.to_csv('UOS actual.csv')
X_UOS.to_csv('xUOS.csv')
# print(X_UOS)

print("*******************************************************************")
print("The reuslt on UOS dataset: ")
y_pred_UOS = catb.predict(X_UOS)
y_pred_UOS=pd.DataFrame(y_pred_UOS)
y_pred_UOS.to_csv('y_pred_UOS.csv')
cm_UOS = confusion_matrix(Y_UOS, y_pred_UOS)

accuracy_score(Y_UOS, y_pred_UOS)
print(classification_report(Y_UOS, y_pred_UOS))
print(precision_recall_fscore_support(Y_UOS, y_pred_UOS))

print('this is the CM for the UOS dataset 106K')
# cf_matrix_UOS = confusion_matrix(Y_UOS, y_pred_UOS)
print(cm_UOS)

import seaborn as sns
sns.heatmap(cm_UOS, annot=True, fmt='g')
plt.show()

y_pred_UOS=y_pred_UOS.values.flatten()


print('First fig is for xUOS vs actual UOS')

# sns.scatterplot(x=X_UOS.HR, y=Y_UOS)
# plt.xlabel('Steps')
# plt.ylabel('Actual Output')
# plt.title('Steps vs Actual Output')
# plt.show()
# #
# print('second fig is for xUOS vs predicted UOS')
#
# sns.scatterplot(x=X_UOS.HR, y=y_pred_UOS)
# plt.xlabel('Steps')
# plt.ylabel('Predicted Output')
# plt.title('Steps vs Predicted Output')
# plt.show()

#catb.save_model('Cat_Model')

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
np.random.seed(123)
import seaborn as sns
import matplotlib.pyplot as plt

import shap
print(f"Shap version used: {shap.__version__}")
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

## implement shaply
explainer = shap.Explainer(catb)
print('check XAI 1')
shap_values = explainer(X_val)
print('check XAI 2')

plt.title('Feature Importance using SHAP')
shap.plots.bar(shap_values, show=True, max_display=5)