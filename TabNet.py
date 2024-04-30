import pandas as pd
import numpy as np

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

from numpy import isnan
# print total missing
print('Missing: %d' % sum(isnan(features).flatten()))


from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(features, target, test_size=0.30, random_state=8)

x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.50, random_state=8)

print("X train shape: ", x_train.shape)
print("X validation shape: ", x_val.shape)
print("X test shape: ", x_test.shape)
print("Y train shape: ", y_train.shape)
print("Y validation shape: ", y_val.shape)
print("Y test shape: ", y_test.shape)

import pytorch_tabnet
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# define the model
clf1_nopreproc = TabNetClassifier(optimizer_fn=torch.optim.Adam, n_d= 32, n_a=32, n_steps= 6,

                       optimizer_params=dict(lr=2e-2),
                       scheduler_params={"step_size":10, # how to use learning rate scheduler
                                         "gamma":0.9},
                       scheduler_fn=torch.optim.lr_scheduler.StepLR,
                       mask_type='entmax' # "sparsemax"

                       #n_independent=1-5,
                      )


# fit the model
clf1_nopreproc.fit(
    x_train,y_train,
    eval_set=[(x_train, y_train), (x_val, y_val)],
    eval_name=['train', 'valid'],
    eval_metric=['auc','accuracy'],
    max_epochs=80 , patience=50,
    batch_size=256, virtual_batch_size=128,
    num_workers=0,
    weights=1,
    drop_last=False
)
import matplotlib.pyplot as plt

# plot losses
plt.plot(clf1_nopreproc.history['loss'])

# plot accuracy
plt.plot(clf1_nopreproc.history['train_accuracy'])
plt.plot(clf1_nopreproc.history['valid_accuracy'])

# find and plot feature importance
# y_pred = clf1_nopreproc.predict(x_test)
# clf1_nopreproc.feature_importances_
# feat_importances = pd.Series(clf1_nopreproc.feature_importances_, index=feat.columns)
# feat_importances.nlargest(20).plot(kind='barh')


# determine best accuracy for test set
preds = clf1_nopreproc.predict(x_test)
test_acc = accuracy_score(preds, y_test)

# determine best accuracy for validation set
preds_valid = clf1_nopreproc.predict(x_val)
valid_acc = accuracy_score(preds_valid, y_val)
precision= precision_score(preds_valid, y_val)
recall= recall_score(preds_valid, y_val)

print(f"BEST ACCURACY SCORE ON VALIDATION SET : {valid_acc}")
print(f"BEST ACCURACY SCORE ON TEST SET : {test_acc}")
print(f"BEST precision SCORE ON val SET : {precision}")
print(f"BEST Recall SCORE ON val SET : {recall}")


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
preds_UOS = clf1_nopreproc.predict(features)
test_acc_UOS = accuracy_score(preds_UOS, target)
test_pre_UOS = precision_score(preds_UOS, target)
test_recall_UOS = recall_score(preds_UOS, target)

print(f"BEST ACCURACY SCORE ON UOS dataset : {test_acc_UOS}")
print(f"BEST precision SCORE ON UOS dataset : {test_pre_UOS}")
print(f"BEST Recall SCORE ON UOS dataset : {test_recall_UOS}")

