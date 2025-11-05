## Importing necessary libraries

# For scientific computation and processing array elements.
import numpy as np

# Importing pandas
import pandas as pd

# For plotting statstical visualization
import matplotlib.pyplot as plt
import seaborn as sns

# For pretty-printing tabular data
from tabulate import tabulate


# For handling class imbalance
from imblearn.over_sampling import SMOTE

# For Split dataset into train and test
from sklearn.model_selection import train_test_split


# For Scaliing dataset
from sklearn.preprocessing import MinMaxScaler

# Importing algorithams for building model
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

# Evaluation metrics
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc


# For building Artificial Neural Networks
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout
from keras import regularizers

#To ignore warnings
import warnings
warnings.filterwarnings('ignore')

from modeleval import *

# 데이터셋 불러오기
df=pd.read_csv(r"dataset/marketing_train.csv", sep =",")


# 범주형 변수 찾기
categorical_variables = [var for var in df.columns if df[var].dtype=='O']

# 숫자형 속성 찾기
numerical_variables=[var for var in df.columns if var not in categorical_variables]

# unknown 값을 결측치로 간주 (null 로 변경)
df = df.replace('unknown', np.nan)

# 결측 비율이 50% 초과하는 속성 제거
df.drop(columns='prev_outcome', inplace=True)

# 결측치를 최빈값으로 대체
df['education']=df['education'].fillna(df['education'].mode()[0])
df['job']=df['job'].fillna(df['job'].mode()[0])

# 분위수 범위를 통한 이상치 제거

# prev_call 제외한 이상치를 가진 속성 선택
outlier_var=['age', 'balance', 'call_duration', 'campaign']

# Capping dataset
for i in outlier_var:
    # Findling IQR
    Q1=df[i].quantile(0.25)
    Q3=df[i].quantile(0.75)
    IQR=Q3-Q1

    # Defining upper and lower limit
    lower_limit =df[i].quantile(0.25)-1.5*IQR
    upper_limit =df[i].quantile(0.75)+1.5*IQR

    # 이상치를 upper_limit 또는 lower_limit 값으로 조정
    df.loc[(df[i] > upper_limit),i] = upper_limit
    df.loc[(df[i] < lower_limit),i] = lower_limit

# describe 결과에서 unique 행만 추출하고 전치(transpose)
unique_summary = df[categorical_variables].describe(include='all').loc[['unique']].T

# 인덱스 초기화하고 보기 좋게
unique_summary.reset_index(inplace=True)
unique_summary.columns = ['속성', 'unique 개수']

## 라벨 인코딩

df['marital'] = df['marital'].map({'single':0,'married':1,'divorced':2})
df['education'] = df['education'].map({'secondary':0,'tertiary':1, 'primary':2})
df['default'] = df['default'].map({'yes':1,'no':0})
df['housing'] = df['housing'].map({'yes':1,'no':0})
df['loan'] = df['loan'].map({'yes':1,'no':0})
df['target'] = df['target'].map({'yes':1,'no':0})

## job 변수는 OneHot 인코딩
df = pd.get_dummies(df, columns=['job'], prefix=["job"], drop_first=True, dtype=bool)

# 종속변수 선택
dependent_variable = 'target'

# 종속변수를 제외한 모든 변수가 독립변수
independent_variables = list(set(df.columns.tolist()) - {dependent_variable})

# 독립변수 데이터 생성
X = df[independent_variables].copy()
# 종속변수 데이터 생성
y = df[dependent_variable].copy()

# Transforming data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_minmax = scaler.fit_transform(X)

# Synthetic Minority Oversampling Technique (SMOTE) 를 사용한 데이터 불균형 해소

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=0)

# fit predictor and target variable
x_smote, y_smote = smote.fit_resample(X,y)

# Train / Test 로 데이터 분리
from sklearn.model_selection import train_test_split, GridSearchCV
X_train, X_test, y_train, y_test= train_test_split(x_smote, y_smote, test_size=0.2, random_state=42)


## 인공신경망(ANN) 모델 구성


# 필요한 라이브러리 임포트
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout
from keras import regularizers

# 인공신경망 모델 초기화
classifier = Sequential()

# 입력층과 첫 번째 은닉층 추가
classifier.add(Dense(units = 51, kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation = 'relu', input_dim = 20))

# 두 번째 은닉층 추가
classifier.add(Dense(units = 51,kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001), activation = 'relu'))

# 출력층 추가 (이진 분류용)
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# 모델 학습 수행
adam = Adam(learning_rate=0.001)
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# 모델 학습 수행
classifier.fit(X_train, y_train, batch_size = 10, epochs = 20)

## 예측 수행
y_train_predict = classifier.predict(X_train)
y_train_pred = (y_train_predict > 0.5)

y_test_predict = classifier.predict(X_test)
y_test_pred = (y_test_predict > 0.5)
print('__'*45)

## 모델 평가
print('Training set evaluation result for ANN :\n')
cm_train_ann = confusion_matrix(y_train, y_train_pred)
accuracy_train_ann = accuracy_score(y_train, y_train_pred)
precision_train_ann = precision_score(y_train, y_train_pred)
recall_train_ann = recall_score(y_train, y_train_pred)
f1_train_ann = f1_score(y_train, y_train_pred)
roc_auc_score_train_ann = roc_auc_score(y_train, y_train_pred)
print("Confusion Matrix: \n", cm_train_ann)
print("Accuracy: ", accuracy_train_ann)
print("Precision: ", precision_train_ann)
print("Recall: ", recall_train_ann)
print("F1 Score: ", f1_train_ann)
print("roc_auc_score: ", roc_auc_score_train_ann)
print('\n-------------------------------\n')
print('Test set evaluation result for ANN :\n')
cm_test_ann = confusion_matrix(y_test, y_test_pred)
accuracy_test_ann = accuracy_score(y_test, y_test_pred)
precision_test_ann = precision_score(y_test, y_test_pred)
recall_test_ann = recall_score(y_test, y_test_pred)
f1_test_ann = f1_score(y_test, y_test_pred)
roc_auc_score_test_ann=roc_auc_score(y_test, y_test_pred)
print("Confusion Matrix: \n", cm_test_ann)
print("Accuracy: ", accuracy_test_ann)
print("Precision: ", precision_test_ann)
print("Recall: ", recall_test_ann)
print("F1 Score: ", f1_test_ann)
print("roc_auc_score: ", roc_auc_score_test_ann)
print('=='*45)

## 평가 지표 시각화
fig,axes = plt.subplots(nrows=2, ncols=2)
ax1 = sns.heatmap(cm_train_ann, annot=True, ax=axes[0,0], fmt='d')
ax1.set_title('Confusion Matrix for training set')
ax1.set_ylabel('True label')
ax1.set_xlabel('Predicted label')
ax2 = sns.heatmap(cm_test_ann, annot=True, ax=axes[0,1], fmt='d')
ax2.set_title('Confusion Matrix for test set')
ax2.set_ylabel('True label')
ax2.set_xlabel('Predicted label')
ax3 = sns.barplot(x=['Accuracy', 'Precision', 'Recall', 'F1','roc_auc_score'], y=[accuracy_train_ann, precision_train_ann, recall_train_ann, f1_train_ann, roc_auc_score_train_ann], ax=axes[1,0])
ax3.set_title('Evaluation Metrics for training set-ANN')
ax3.tick_params(axis='x', rotation=90)
ax4 = sns.barplot(x=['Accuracy', 'Precision', 'Recall', 'F1','roc_auc_score'], y=[accuracy_test_ann, precision_test_ann, recall_test_ann, f1_test_ann, roc_auc_score_test_ann], ax=axes[1,1])
ax4.set_title('Evaluation Metrics for test set-ANN')
ax4.tick_params(axis='x', rotation=90)
plt.tight_layout()
plt.show()
print('=='*45)

# Plot ROC curve for ANN
plot_roc_curve(y_test, y_test_pred)

# ANN 모델의 train_test_split 평가 지표 비교

import pandas as pd
import matplotlib.pyplot as plt

# 평가 지표 목록
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score','roc_auc_score']
ev_result = [accuracy_test_ann, precision_test_ann, recall_test_ann, f1_test_ann, roc_auc_score_test_ann]

# 데이터프레임 생성
ann_evaluation_df = pd.DataFrame({'Evaluation Metrics': metrics,
                   'Result': ev_result})
# 데이터프레임 출력
print('=='*45)
print(tabulate(ann_evaluation_df, headers='keys', tablefmt='grid'))
print('\n', '=='*45, '\n')

# 막대 그래프 생성
plt.figure(figsize=(7,7))
ax=ann_evaluation_df.plot.bar(x='Evaluation Metrics', rot=0)

# 제목 및 라벨 설정
ax.set_title("Comparing Evaluation Metrics of ANN")
ax.set_xlabel("Evaluation Metrics")
ax.set_ylabel("Accuracy Score")
ax.bar_label(ax.containers[0])
ax.legend(loc="lower right")

# 그래프 출력
plt.show()

print('=='*45)

# Gradient Boosting Machine 알고리즘을 환경에 import
from sklearn.ensemble import GradientBoostingClassifier
# Gradient Boosting Machine 모델을 학습 데이터에 학습시킴
classifier_gbm = GradientBoostingClassifier(max_leaf_nodes=10, random_state=0)

# XGBoost 알고리즘을 환경에 import
from xgboost import XGBClassifier
# XGBoost 모델을 학습 데이터에 학습시킴
classifier_xgb = XGBClassifier(max_leaf_nodes=10, random_state=0)
xgb = classification_model(X_train, X_test, y_train, y_test, classifier_gbm)

# XGBoost 분류기의 ROC 커브를 그림
y_pred = xgb['y_test_pred']
plot_roc_curve(y_test, y_pred)


# XGBoost 알고리즘을 환경에 import
from xgboost import XGBClassifier

## 교차검증을 사용하여 XGBoost 모델을 학습시킴

# 파라미터 딕셔너리 정의
param_grid = {'n_estimators':[50,80,100],
              'max_depth':[4,6,8],
              'min_samples_split':[50,100,150],
              'min_samples_leaf':[40,50]}
# XGBoost 분류기 인스턴스 생성
classifier_xgb = XGBClassifier(max_leaf_nodes=10, random_state=0)
# 모델 학습
xgb_cv = classification_CV_model(X_train, X_test, y_train, y_test, classifier_xgb, param_grid)

# XGBoost 분류기의 ROC 커브를 그림
y_pred = xgb_cv['y_test_pred']
plot_roc_curve(y_test, y_pred)


# XGBoost 모델의 train_test_split과 GridSearchCV 결과 비교

import pandas as pd
import matplotlib.pyplot as plt

# 지표 목록
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score','roc_auc_score']
train_test_split = [xgb['accuracy_test'], xgb['precision_test'], xgb['recall_test'], xgb['f1_test'], xgb['roc_auc_score_test']]
cross_validation = [xgb_cv['accuracy_test'], xgb_cv['precision_test'], xgb_cv['recall_test'], xgb_cv['f1_test'], xgb_cv['roc_auc_score_test']]

# 데이터프레임 생성
dt_evaluation_df = pd.DataFrame({'Evaluation Metrics': metrics,
                   'Train/Test Split': train_test_split,
                   'Cross-Validation': cross_validation})
# 데이터프레임 출력
print('=='*45)
print(tabulate(dt_evaluation_df, headers='keys', tablefmt='grid'))
print('\n', '=='*45, '\n')

# 막대그래프 생성
plt.figure(figsize=(7,7))
ax=dt_evaluation_df.plot.bar(x='Evaluation Metrics', rot=0)

# 그래프 제목 및 라벨 설정
ax.set_title("Comparing Evaluation Metrics of Train-Test Split vs. Cross-Validation for Decision Tree")
ax.set_xlabel("Evaluation Metrics")
ax.set_ylabel("Accuracy Score")
ax.legend(loc="lower right")

# 그래프 출력
plt.show()
print('=='*45)


# Data: XGBoost와 ANN만 포함
model = ['XGBoost', 'ANN']

Accuracy = [xgb_cv['accuracy_test'], accuracy_test_ann]
Precision = [xgb_cv['precision_test'], precision_test_ann]
Recall = [xgb_cv['recall_test'], recall_test_ann]
F1_score = [xgb_cv['f1_test'], f1_test_ann]
roc_auc_score = [xgb_cv['roc_auc_score_test'], roc_auc_score_test_ann]
confusion_matrix = [xgb_cv['cm_test'], cm_test_ann]

# Create a dataframe
models_evaluation_df = pd.DataFrame({
    'model': model,
    'Accuracy': Accuracy,
    'Precision': Precision,
    'Recall': Recall,
    'F1_score': F1_score,
    'roc_auc_score': roc_auc_score,
    'confusion matrix': confusion_matrix
})

# 결과 출력
models_evaluation_df

# 데이터셋 불러오기
df_prod=pd.read_csv(r"dataset/marketing_test_without_target.csv", sep =",")

# unknown 값을 결측치로 간주 (null 로 변경)
df_prod = df_prod.replace('unknown', np.nan)

# 결측 비율이 50% 초과하는 속성 제거
df_prod.drop(columns='prev_outcome', inplace=True)

# 결측치를 최빈값으로 대체
df_prod['education']=df_prod['education'].fillna(df_prod['education'].mode()[0])
df_prod['job']=df_prod['job'].fillna(df_prod['job'].mode()[0])

# 분위수 범위를 통한 이상치 제거

# prev_call 제외한 이상치를 가진 속성 선택
outlier_var=['age', 'balance', 'call_duration', 'campaign']

# Capping dataset
for i in outlier_var:
    # Findling IQR
    Q1=df_prod[i].quantile(0.25)
    Q3=df_prod[i].quantile(0.75)
    IQR=Q3-Q1

    # Defining upper and lower limit
    lower_limit =df_prod[i].quantile(0.25)-1.5*IQR
    upper_limit =df_prod[i].quantile(0.75)+1.5*IQR

    # 이상치를 upper_limit 또는 lower_limit 값으로 조정
    df_prod.loc[(df_prod[i] > upper_limit),i] = upper_limit
    df_prod.loc[(df_prod[i] < lower_limit),i] = lower_limit

# Addressing categorical variables from the dataset
categorical_variables=df_prod.describe(include=['object']).columns

## 라벨 인코딩

df_prod['marital'] = df_prod['marital'].map({'single':0,'married':1,'divorced':2})
df_prod['education'] = df_prod['education'].map({'secondary':0,'tertiary':1, 'primary':2})
df_prod['default'] = df_prod['default'].map({'yes':1,'no':0})
df_prod['housing'] = df_prod['housing'].map({'yes':1,'no':0})
df_prod['loan'] = df_prod['loan'].map({'yes':1,'no':0})

## job 변수는 OneHot 인코딩
df_prod = pd.get_dummies(df_prod, columns=['job'], prefix=["job"], drop_first=True, dtype=bool)

# Transforming data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_prod = scaler.fit_transform(df_prod)

# 최적 모델 객체 추출 (GridSearchCV 객체 안에서 최적 모델 꺼내기)
best_model = xgb_cv['model'].best_estimator_


# 예측 수행
y_pred_label = best_model.predict(X_prod)
y_pred_proba = best_model.predict_proba(X_prod)[:, 1]

# 예측 결과 저장
df_prod['predicted_label'] = y_pred_label
df_prod['predicted_proba'] = y_pred_proba

print(df_prod[['predicted_label', 'predicted_proba']].head())

# 예측 결과 포함된 데이터프레임을 CSV 파일로 저장
df_prod.to_csv("/content/drive/MyDrive/Colab Notebooks/results/prediction_result.csv",
               index=False, encoding='utf-8-sig')
