import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import lime

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split




df=pd.read_csv("10-8-22.csv",header=0 )

HR ='HR'
Steps='Steps'
# D_N='D_N'
RRinterval='RRinterval'
HR_Diff='HR_Diff'
label_sym='label_sym'
df.columns = [HR, Steps, RRinterval, HR_Diff,label_sym]

X = df.iloc[:,0:4]
Y = df.iloc[:,4]

import shap
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=0)


print (Y)
model = RandomForestRegressor(n_estimators=20, max_depth=5)
print(model)
model.fit(X_train, y_train)

score =model.score(X_val, y_val)
print ('score is: ' + str(score))


## implement shaply
explainer = shap.Explainer(model)
print('check XAI 1')
shap_values = explainer(X_val)
print('check XAI 2')

plt.title('Feature Importance using SHAP')
shap.plots.bar(shap_values, show=True, max_display=12)
#
# plt.title('SHAP Heatmap Plot')
# shap.plots.heatmap(shap_values, max_display=12, show=False)
# plt.gcf().axes[-1].set_box_aspect(100)
# plt.ylabel('Features')
# plt.show()

### implement Lime
#
# import lime
# from lime import lime_tabular
#
# explainer = lime_tabular.LimeTabularExplainer(
#     training_data=np.array(X_train),
#     feature_names=X_train.columns,
#    class_names=[0, 1],
#     mode='classification'
# )
#
# exp = explainer.explain_instance(
#     data_row=X_val.iloc[1],
#     predict_fn=model.predict_proba
# )
#
# exp.show_in_notebook(show_table=True)
# exp.save_to_file('lime.html')