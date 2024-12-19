## main
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from imblearn.over_sampling import SMOTE
from PIL import Image
from matplotlib import image
import joblib

## skelarn -- preprocessing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_features.transformers import DataFrameSelector

## skelarn -- models
# from sklearn.neighbors import KNeighborsClassifier
#from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

## sklearn -- metrics
from sklearn.metrics import f1_score, confusion_matrix

TRAIN_PATH = os.path.join(os.getcwd(),'dataset' ,'dataset.csv')
df = pd.read_csv(TRAIN_PATH)

df.drop(columns=["RowNumber","CustomerId","Surname"], axis=1, inplace=True)

df.drop(index=df[df["Age"]>50].index.tolist(),axis=0,inplace=True)

X = df.drop(columns=['Exited'], axis=1)
y = df['Exited']

## Split to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=45, stratify=y)

num_cols = ['Age', 'CreditScore', 'Balance', 'EstimatedSalary']
categ_cols = ['Gender', 'Geography']

ready_cols = list(set(X_train.columns.tolist()) - set(num_cols) - set(categ_cols))

num_pipeline = Pipeline(steps=[
                        ('selector', DataFrameSelector(num_cols)),
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler())
                    ])


## For Categorical
categ_pipeline = Pipeline(steps=[
                        ('selector', DataFrameSelector(categ_cols)),
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('ohe', OneHotEncoder(drop='first', sparse_output=False))
                    ])


## For ready cols
ready_pipeline = Pipeline(steps=[
                        ('selector', DataFrameSelector(ready_cols)),
                        ('imputer', SimpleImputer(strategy='most_frequent'))
                    ])



## combine all
all_pipeline = FeatureUnion(transformer_list=[
                                    ('numerical', num_pipeline),
                                    ('categorical', categ_pipeline),
                                    ('ready', ready_pipeline)
                                ])

## apply
X_train_final = all_pipeline.fit_transform(X_train)
X_test_final = all_pipeline.transform(X_test)

## --------------------- Impalancing ---------------------------- ##

## 1. use algorithm without taking the effect of imbalancing

## 2. prepare class_weights for solving imbalance dataset
vals_count = 1 - (np.bincount(y_train) / len(y_train))
vals_count = vals_count / np.sum(vals_count)  ## normalizing


dict_weights = {}
for i in range(2):  ## 2 classes (0, 1)
    dict_weights[i] = vals_count[i]

## 3. Using SMOTE for over sampling
over = SMOTE(sampling_strategy=0.7)
X_train_resmapled, y_train_resampled = over.fit_resample(X_train_final, y_train)
with open('metrics.txt', 'w') as f:
    pass
def train_model(X_train, y_train, plot_name='', class_weight=None):
    """ A function to train model given the required train data """

    global clf_name

    clf = LogisticRegression(C = 1.5,penalty='l2',random_state=45,max_iter=1000, class_weight=class_weight)
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test_final)

    ## Using f1_score
    score_train = f1_score(y_train, y_pred_train)
    score_test = f1_score(y_test, y_pred_test)

    clf_name = clf.__class__.__name__

    ## Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True, cbar=False, fmt='.2f', cmap='Blues')
    plt.title(f'{plot_name}')
    plt.xticks(ticks=np.arange(2) + 0.5, labels=[False, True])
    plt.yticks(ticks=np.arange(2) + 0.5, labels=[False, True])

    ## Save the plot locally
    plt.savefig(f'{plot_name}.png', bbox_inches='tight', dpi=300)
    plt.close()

    with open('metrics.txt', 'a') as f:
        f.write(f'{clf_name} {plot_name}\n')
        f.write(f"F1-score of Training is: {score_train * 100:.2f} %\n")
        f.write(f"F1-Score of Validation is: {score_test * 100:.2f} %\n")
        f.write('----' * 10 + '\n')

    joblib.dump(clf,os.path.join(os.getcwd(),'models',f'{clf_name}-{plot_name}.pkl'))
    return
train_model(X_train_final,y_train,plot_name='without_imbalance',class_weight=None)
train_model(X_train_final,y_train,plot_name='with class weight',class_weight=dict_weights)
train_model(X_train_resmapled,y_train_resampled,plot_name='with SMOTE',class_weight=None)

#Combine all cms
confusion_matrix_path= [f'./without_imbalance.png', f'./with class weight.png', f'./with SMOTE.png']
plt.figure(figsize=(15, 5))
for i, path in enumerate(confusion_matrix_path, 1):
    img = Image.open(path)
    plt.subplot(1, len(confusion_matrix_path), i)
    plt.imshow(img)
    plt.axis('off')  # Disable axis for cleaner visualization

plt.suptitle(clf_name, fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f'conf_matrix.png', bbox_inches='tight', dpi=300)



