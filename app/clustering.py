# %% [markdown]
# # 라이브러리 불러오기

# %%
import os
from django.conf import settings
import numpy as np
import pandas as pd
from matplotlib import pyplot
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import  confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import statsmodels.api as sm

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import StratifiedKFold

import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
# 크롬 드라이버 자동 업데이트
# from webdriver_manager.chrome import ChromeDriverManager
import pytesseract
import cv2
from pdf2image import convert_from_path
import re

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import RobustScaler

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import VotingClassifier 

from sklearn.ensemble import AdaBoostClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from yellowbrick.cluster import KElbowVisualizer

from statsmodels.stats.outliers_influence import variance_inflation_factor

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def analysis(apt_name, docfile):
    # %% [markdown]
    # # 직접 수집한 엑셀 파일에서 데이터 분석에 활용할 부분 추출 

    # %%
    path= "C:\\Users\\hyjoo\\Project\\data_onair\\app\\0822data.csv"
    df = pd.read_csv(path)

    # %%
    df['시공일자_datetime'] = pd.to_datetime(df['시공일자'])
    df['시공일자_년도'] = df['시공일자_datetime'].dt.year

    # %%
    X = df[['전세가율','시공일자_년도','근저당정규화','가압류','소유권이전', '신탁', '임차권등기명령']]
    y = df['label']

    # %%
    X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.3 , random_state = 1, stratify = y)
    #split 함수를 통해 train set과 test set 분할 (test set의 비율 : 0.3)

    # %% [markdown]
    # # LogisticRegression()

    # %% [markdown]
    # ##  우선 로지스틱 단일 모델에 별다른 파라미터를 주지 않고, 다양한 성능지표를 확인   
    # ### - 추후에 GridsearchCV 사용 등을 통해 모델 컨트롤 진행

    # %%
    lr = LogisticRegression() # 로지스틱 함수 객체 생성 

    # %%
    lr.fit(X_train,y_train) # fit() 함수를 통한 훈련 
    y_pred = lr.predict(X_test) # 훈련을 통해 만들어진 모델에 test 데이터 대입 
    print(y_pred)




    # %%
    cm = confusion_matrix(y_test, y_pred) # confusion matrix 생성 
    print(cm)

    # %%


    # 다양한 성능지표 확인
    acc = accuracy_score(y_test,y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"정확도: {acc}")
    print(f"정밀도: {prec}")
    print(f"재현율: {rec}")
    print(f"F1 Score : {f1}")

    # %% [markdown]
    # # ROC curve & AUC SCORE
    # 
    # x 축 (FPR, 1-specificity)  = 실제 전세 사기가 아닌 것 중 전세 사기라고 예측한 정도 
    # 
    # y축 (TPR, recall) = 실제 전세 사기인 것 중 전세 사기라고 예측한 정도 
    # 
    # * data imbalanced 상황에서 ROC curve의 사용은 기만적이고 낙관적인 잘못된 해석을 초래할 수 있다는 점을 유의사항으로 둠.

    # %%


    # 모든 예측은 다수 클래스(0:사기 아님)으로 설정 
    ns_probs = [0 for _ in range(len(y_test))] 

    # 모델 피팅 
    model = LogisticRegression(solver='lbfgs')
    model.fit(X_train, y_train)

    # 위에서 피팅한 모델에 test 데이터 예측 (확률로 예측)
    lr_probs = model.predict_proba(X_test)

    # 1(사기) 클래스에 대한 확률만 추출 
    lr_probs = lr_probs[:, 1]

    # AUC SCORE 계산 
    ns_auc = roc_auc_score(y_test, ns_probs)
    lr_auc = roc_auc_score(y_test, lr_probs)

    # 점수 출력 
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic Regression: ROC AUC=%.3f' % (lr_auc))

    # ROC curve 계산
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)

    # ROC curve 그리기
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill') # No Skill 에 대해서는  ROC 커브는 대시 선 스타일로 작성 
    pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic') # Logistic Regression 예측에 대한 ROC 커브는 점으로 표시

    # 축 설정 
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')

    # 범례 표시
    pyplot.legend()

    # ROC Curve plot 보이기 
    # pyplot.show()

    # %% [markdown]
    # # Precision - Recall curve 
    # 
    # precision = 전세 사기라고 예측한 것 중에서 실제 사기 
    # 
    # TPR = recall = 실제 사기인 것 중 사기라고 예측한 정도 
    # 
    # used in class imbalance.

    # %%

    # 모델 피팅 
    model = LogisticRegression(solver='lbfgs')
    model.fit(X_train, y_train)


    # 위에서 피팅한 모델에 test 데이터 예측 (확률로 예측)
    lr_probs = model.predict_proba(X_test)
    # 1(사기) 클래스에 대한 확률만 추출 
    lr_probs = lr_probs[:, 1]
    # # 클래스 값 예측
    yhat = model.predict(X_test)


    # Logistic Regression 예측 확률에 대한 정밀도-재현율 곡선 계산
    lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)

    # Logistic Regression 예측에 대한 F1 스코어와 AUC 계산
    lr_f1, lr_auc = f1_score(y_test, yhat), auc(lr_recall, lr_precision)

    # 점수 출력
    print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))



    # 테스트 세트에서 양성 클래스 (클래스 1)의 비율을 계산하여 "No Skill" 예측의 기준값으로 사용
    no_skill = len(y_test[y_test==1]) / len(y_test)
    # No Skill 곡선을 대시 선 스타일로 그리기
    pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')

    #Logistic Regression 예측에 대한 정밀도-재현율 곡선을 점으로 표시
    pyplot.plot(lr_recall, lr_precision, marker='.', label='Logistic')

    # x축, y축 레이블 설정
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')

    # 범례 표시
    pyplot.legend()

    # 정밀도-재현율 곡선 보여주기  
    # pyplot.show()

    # %% [markdown]
    # # threshold 직접 설정
    # 
    # ### 도메인을 고려했을 때, 실제 사기를 사기라고 판단하는 것이 핵심이다. 
    # ### 따라서, 재현율을 포함한 지표가 주요 지표이며, 재현율 값이 좋지 않다면 다른 성능 지표에 크게 영향을 주지 않는 한 재현율을 높이는 것을 목표로 삼았다. 
    # ### 재현율을 높이기 위해 threshold 값을 default(0.5) 보다 낮게 주어 변화를 관찰할 예정이었으나, 만족하는 성능 지표에 따라 실제 사용은 하지 않았다. 
    # 
    # y_pred = lr.predict_proba(X_test)[:,1]   #. .iloc[:,[0]]
    # 
    # y_pred_series = pd.Series(y_pred)
    # 
    # def PRED(y,threshold):
    #     Y = y.copy()
    #     Y[Y > threshold] = 1
    #     Y[Y <= threshold] = 0
    #     return(Y.astype(int))
    # 
    # Y_pred = PRED(y_pred, 0.5)
    # 
    # Y_pred

    # %% [markdown]
    # # statsmodels 이용해서 통계 결과 확인
    # 
    # * statsmodels은 통계기반이고, sklearn은 머신러닝 기반이기에 불가피하게 statsmodels를 통해 p-value, 회귀 계수등 확인

    # %%

    # %%
    df_new = df[['전세가율','시공일자_년도','압류','근저당정규화','소유권이전','가압류','신탁', '임차권등기명령','label']]

    # %%
    pred_sm = sm.add_constant(df_new, has_constant = 'add')
    pred_sm.head()

    # %%
    feature_columns = list(pred_sm.columns.difference(["label"]))
    X = pred_sm[feature_columns]
    y = pred_sm['label']

    # %%
    X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.3 , random_state = 1, stratify = y)

    # %%
    model = sm.Logit(y_train,X_train)
    results = model.fit(method = "newton")

    results.summary()

    # %%
    results.params

    # %%
    np.exp(results.params)

    # %% [markdown]
    # # SMOTE

    # %%

    # %%
    # 원래 개수 : {0: 173 1: 125}
    # 적은 데이터 개수, 약간의 데이터 불균형을 해소하기 위해 -> 오버샘플링 진행 (단, 과적합을 피하기 위해 성능에 영향을 거의 주지 않는 개수에 한하여 진행)

    smote = SMOTE(sampling_strategy={0: 180, 1: 150},random_state=1) 
    X, y = smote.fit_resample(X, y)

    # %%
    # feature들과 label dataframe을 합침 
    df = pd.concat([X, y], axis=1)
    df








    # %% [markdown]
    # ## 오버샘플링 후 성능지표 확인 

    # %%
    X = df[['전세가율','시공일자_년도','근저당정규화','신탁','압류','가압류','임차권등기명령']]
    y = df['label']




    X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.3 , random_state = 1, stratify = y)

    lr = LogisticRegression()
    lr.fit(X_train,y_train)
    y_pred = lr.predict(X_test)
    y_pred

    cm = confusion_matrix(y_test, y_pred)
    cm

    acc = accuracy_score(y_test,y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"정확도: {acc}")
    print(f"정밀도: {prec}")
    print(f"재현율: {rec}")
    print(f"F1 Score : {f1}")

    # %% [markdown]
    # ## 다중공선성 확인

    # %%
    X.corr() # 상관계수 확인 

    # %%
    # 글꼴 파일 설치 필요 

    # %%
    # 위의 상관계수를 시각화 

    plt.rc("font", family = "Malgun Gothic")
    sns.set(font="Malgun Gothic",
    rc={"axes.unicode_minus":False}, style='white')

    sns.heatmap(X.corr(), annot = True)
    # plt.show()

    # %%
    sns.pairplot(X) # pairplot을 통해 변수간의 산점도 확인 
    # plt.show()

    # %%
    # vif 점수 확인 

    vif = pd.DataFrame()

    vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif["features"] = X.columns
    vif


    # %% [markdown]
    # ### vif score는 10이상이면 다중공선성이 있다고 해석한다. 
    # ### 다중공선성이 있다는 것은 회귀의 기본 가정을 위배하기 때문에 해결해줘야 하는 문제이다. 
    # ### 우리는 추후에 GridSearchCV에 Regularization 을 통해 이를 해결한다. 

    # %% [markdown]
    # # 교차검증
    # 

    # %%

    kf = KFold(n_splits = 5, shuffle = True, random_state = 1) # 5 -fold cross validation을 진행  
    n_iter = 0 # iteration 초기화

    # 각 Iteration에서 학습 데이터와 검증 데이터의 label 분포를 작성 

    for train_index, test_index in kf.split(X):
        n_iter +=1
        label_train = df['label'].iloc[train_index]
        label_test = df['label'].iloc[test_index]
        
        print('교차검증 : {} 번째'.format(n_iter))
        print('학습 데이터의 레이블 분포 :\n', label_train.value_counts())
        print('\n')
        print('검증 데이터의 레이블 분포 :\n', label_test.value_counts())
        print('\n')
        print('\n')

    # %% [markdown]
    # # stratifiedKfold
    # 
    # * stratifiedKfold 하게되면 각 iteration 마다 label이 균등하게 배분되어 학습하고 검증하기 때문에 보다 공정한 상황에서의 결과라고 해석할 수 있다. 특히, 특정 레이블 값이 특이하게 많거나 적어서 값의 분포가 치우치는 상황에 활용된다. 
    # * 우리의 경우 데이터 수집과 smote를 통해 data imbalance를 어느정도 해결하였지만, 약간의 imbalance가 남아있어 stratifiedKfold를 사용하여 조금이나마 더 보완하고자 한다.    

    # %%

    # %%
    skf = StratifiedKFold(n_splits=5, random_state=1, shuffle = True ) # 5 -StratifiedKfold를 진행  

    cnt_iter = 0 # iteration 횟수 초기화 

    # 각 Iteration에서 학습 데이터와 검증 데이터의 label 분포를 작성 
    for train_index, test_index in skf.split(X,y):
        cnt_iter += 1
        label_train = df['label'].iloc[train_index]
        label_test = df['label'].iloc[test_index]
        print('교차검증 : {} 번째'.format(cnt_iter))

        print('학습 데이터의 레이블 분포 :\n', label_train.value_counts())
        print('\n')
        print('검증 데이터의 레이블 분포 :\n', label_test.value_counts())
        print('\n')
        print('\n')

    # %% [markdown]
    # # GridsearchCV

    # %% [markdown]
    # * GridsearchCV를 통해 직접 파라미터 후보를 설정하고 최적의 파라미터를 찾아 성능을 높인다. 
    # * 단순히 성능을 높이는 것을 넘어, l2 regularization 등을 주어 이전에 발생한 vif 문제등을 해결한다.

    # %% [markdown]
    # ### LogisticRegression - GridsearchCV

    # %%

    # 로지스틱 객체 생성 
    lreg = LogisticRegression()

    # 파라미터 후보
    param_lreg = {'C': [0.001, 0.01, 0.1, 1, 10, 30, 50, 100],
                'penalty' :['l1','l2']
                }

    # StratifiedKFold 
    skf = StratifiedKFold(n_splits=5, random_state=1, shuffle = True  )  

    # iteration 마다 다른 지표를 관찰하면서 파라미터를 결정 및 점수 확인   
    scores = ['roc_auc','recall', 'f1']






    for score in scores:
        grid_search = GridSearchCV(estimator = lreg, param_grid = param_lreg, scoring = '%s' % score, cv=skf, refit='roc_auc') # roc 점수를 기준 삼아 모델을 refit
        grid_search.fit(X_train, y_train) # train 데이터로 피팅 
        print('Lreg 파라미터:' , grid_search.best_params_) # 각 iteration에서 최적의 파라미터 출력 
        print('Lreg 최고점수: {:.4f}'.format(grid_search.best_score_)) # 각 iteration에서 가장 높은 점수 기록

    y_pred = grid_search.predict(X_test) # test 데이터로 예측 
    print("\n")
    # 성능 지표 확인 
    print(f"실제값과 예측값 정확도 auc: {roc_auc_score(y_test, y_pred)}")
    print(f"실제값과 예측값 정확도 recall: {recall_score(y_test, y_pred)}")
    print(f"실제값과 예측값 정확도 f1: {f1_score(y_test, y_pred)}")


    # %% [markdown]
    # # K-Nearest Neighbor1
    # - minmaxscaler 를 이용하여 scaling 진행 
    # - minmaxscaler의 경우 우리의 domain에 맞지 않은 특성을 가지며, 성능 지표에서도 다시 증명됨 

    # %%

    # %%
    # knn 객체 생성 
    knn = KNeighborsClassifier()

    # knn에서 설정 가능한 파라미터 파악 
    estimator = knn
    estimator.get_params().keys()  

    # %%
    # scaling
    # knn은 거리에 기반한 classifier이기에 사전적으로 scaling 작업 수행 


    # MinMaxScaler 객체 생성 
    scaler = MinMaxScaler()

    # %%


    # KNN 객체 생성 
    knn = KNeighborsClassifier()


    # 파라미터 후보
    knears_params = {"n_neighbors": list(range(1,20,1)),
                    "weights": ["uniform", "distance"],
                    "metric" : ['euclidean', 'manhattan', 'minkowski'],
                    "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute']}


    skf = StratifiedKFold(n_splits=5, random_state=1, shuffle = True  )

    # 위의 GridsearchCV 과정과 동일 
    scores = ['roc_auc','recall', 'f1']

    X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.3 , random_state = 1, stratify = y)

    # 미리 생성한 scaling 객체를 통해 scaling 
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for score in scores:
        grid_search = GridSearchCV(estimator = knn, param_grid = knears_params, scoring = '%s' % score, cv=skf, refit='roc_auc')
        grid_search.fit(X_train_scaled, y_train)
        print('KNN 파라미터:' , grid_search.best_params_)
        print('KNN 최고점수: {:.4f}'.format(grid_search.best_score_)) #최고 점수 ex) recall 최고점수 , f1최고점수

    y_pred = grid_search.predict(X_test)
    print("\n")
    print(f"실제값과 예측값 정확도 auc score: {roc_auc_score(y_test, y_pred)}")
    print(f"실제값과 예측값 정확도 recall: {recall_score(y_test, y_pred)}")
    print(f"실제값과 예측값 정확도 f1: {f1_score(y_test, y_pred)}")


    # %% [markdown]
    # # K-Nearest Neighbor2
    # - robust 스케일링
    # - 이상치의 영향을 최소화 할 수 있기에 우리 domain에 적절하다고 판단하여 사용 

    # %%

    scaler = RobustScaler()

    # %%



    knn2 = KNeighborsClassifier()

    # 파라미터 후보
    knears_params = {"n_neighbors": list(range(1,20,1)),
                    "weights": ["uniform", "distance"],
                    "metric" : ['euclidean', 'manhattan', 'minkowski'],
                    "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute']}


    skf = StratifiedKFold(n_splits=5, random_state=1, shuffle = True  )
    scores = ['recall', 'f1']


    X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.3 , random_state = 1, stratify = y)

    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)



    for score in scores:
        grid_search = GridSearchCV(estimator = knn2, param_grid = knears_params, scoring = '%s_macro' % score, cv=skf, refit='roc_auc')
        grid_search.fit(X_train_scaled, y_train)
        print('KNN 파라미터:' , grid_search.best_params_)
        print('KNN 예측 정확도: {:.4f}'.format(grid_search.best_score_)) #최고 점수 ex) recall 최고점수 , f1최고점수

    y_pred = grid_search.predict(X_test)
    print("\n")
    print(f"실제값과 예측값 정확도 auc score: {roc_auc_score(y_test, y_pred)}")
    print(f"실제값과 예측값 정확도 recall: {recall_score(y_test, y_pred)}")
    print(f"실제값과 예측값 정확도 f1: {f1_score(y_test, y_pred)}")

    # %% [markdown]
    # # SVC
    # 
    # * C: 오차허용을 조절하는 매개변수, 작은 c값은 오차허용을 늘림. 하지만 C값이 너무 크면 모델이 복잡
    # * break_ties: bool(TRUE, FALSE로 끝, default =FALSE)
    # * gamma ('scale', 'auto', float): 커널함수에 영향을 주는 매개변수, 데이터 포인트의 영향 범위를 조절함.
    # * shrinking(bool, default = True): 서포트 벡터 개수를 줄이는 최적화 기법

    # %%

    # %%
    # 파라미터 확인 

    svc = SVC()
    estimator = svc
    estimator.get_params().keys()   

    # %%

    #객체 생성 
    # decision_function_shape: default = ovr(일대다) <-> ovo (일대일) 
    # probability(bool, default = False) : return을 predict_proba로 클래스 확률 예측 

    svc = SVC(decision_function_shape = 'ovo',probability=True)


    # 파라미터 후보
    svc_params = {'C': [0.001, 0.01, 0.1, 1, 10, 30, 50, 100],
                'break_ties': [True, False],
                'gamma': ['scale', 'auto'],
                'shrinking' : [True, False]}


    skf = StratifiedKFold(n_splits=5, random_state=1, shuffle = True  )
    # 그리드 서치 진행
    scores = ['recall', 'f1']

    X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.3 , random_state = 1, stratify = y)

    for score in scores:
        grid_search = GridSearchCV(estimator = svc, param_grid = svc_params, scoring = '%s_macro' % score, cv=skf, refit='roc_auc')
        grid_search.fit(X_train, y_train)
        print('SVC 파라미터:' , grid_search.best_params_)
        print('SVC 최고점수: {:.4f}'.format(grid_search.best_score_)) #최고 점수 ex) recall 최고점수 , f1최고점수

    y_pred = grid_search.predict(X_test)
    print("\n")
    print(f"실제값과 예측값 정확도 auc score: {roc_auc_score(y_test, y_pred)}")
    print(f"실제값과 예측값 정확도 recall: {recall_score(y_test, y_pred)}")
    print(f"실제값과 예측값 정확도 f1: {f1_score(y_test, y_pred)}")

    # %% [markdown]
    # # DecisionTreeClassifier
    # 
    # * ccp_alpha (default=0): 최소비용 복잡도 가지치기
    # * criterion : 분할 기준 => gini, entropy
    # * max_features: 각 분할에서의 최대 특성의 수, 'auto', 'sqrt', 'log2' or None
    # * min_impurity_decrease: 불순도가 감소하는 최소 양, 불순도 감소가 이 값보다 큰 경우에만 분할 수행
    # * splitter : best - 최선의 분할, random - 무작위분할

    # %%

    # %%
    tree = DecisionTreeClassifier()
    estimator = tree
    estimator.get_params().keys()   #파라미터 알 수 있음

    # %%


    # 객체 생성 
    tree = DecisionTreeClassifier()


    # 파라미터 후보

    tree_params = {'ccp_alpha': [0, 1],
                'criterion': ['gini', 'entropy'],
                'max_features': ['auto', 'sqrt', 'log2', None],
                'min_impurity_decrease' : [0, 0.05, 0.1, 0.15],
                'splitter' : ['random', 'best']}



    skf = StratifiedKFold(n_splits=5, random_state=1, shuffle = True  )

    # 그리드 서치 진행
    scores = ['recall', 'f1']

    X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.3 , random_state = 1, stratify = y)

    for score in scores:
        grid_search = GridSearchCV(estimator = tree, param_grid = tree_params, scoring = '%s_macro' % score, cv=skf, refit='roc_auc')
        grid_search.fit(X_train, y_train)
        print('decisiontree 파라미터:' , grid_search.best_params_)
        print('decisiontree 예측 정확도: {:.4f}'.format(grid_search.best_score_)) #최고 점수 ex) recall 최고점수 , f1최고점수

    y_pred = grid_search.predict(X_test)
    print("\n")
    print(f"실제값과 예측값 정확도 auc score: {roc_auc_score(y_test, y_pred)}")
    print(f"실제값과 예측값 정확도 recall: {recall_score(y_test, y_pred)}")
    print(f"실제값과 예측값 정확도 f1: {f1_score(y_test, y_pred)}")

    # %% [markdown]
    # # 보팅 - 소프트보팅
    # * 다양한 Classifier를 결합하여 하나의 classifier에 overfitting 되지 않도록 하였으며, 성능 또한 단일 모델을 사용했을 때보다 주요 지표로 삼은 것들에서 값이 우수하여 사용하였음    

    # %%

    # %%
    # 앞 서, GridsearchCV 로 만든 개별 모델들을 소프트 보팅으로 결합 및 객체 생성 
    svot_clf = VotingClassifier (estimators = [('LR', lreg),('KNN', knn2), ('svc',svc),('dt' ,tree)], voting = 'soft')

    # 모델 피팅 
    svot_clf.fit(X_train, y_train)

    # %%
    # 여러 성능지표 확인 
    pred_svot = svot_clf.predict(X_test)
    print(roc_auc_score(y_test, pred_svot))
    print(f1_score(y_test, pred_svot))
    print(recall_score(y_test, pred_svot))
    print(accuracy_score(y_test, pred_svot))

    # %% [markdown]
    # # 새로운 데이터 만들기 - 아직 거래가 일어나지 않은 새로운 데이터에 대한 예측
    # ## ( with 크롤링 & OCR, Optical Character Recognition) 

    # %% [markdown]
    # ##   1. 크롤링 : 네이버 부동산을 통해 전세가율, 시공일자 등 모델링에 사용할  값 가져오기 

    # %%



    # %%
    #매매가와 전세가를 가지고 오기 위한 딕셔너리

    item=dict()

    # 입력받을 때 출력할 멘트 
    query_txt = apt_name

    # 네이버 부동산 url 
    url = "https:/land.naver.com"


    options = Options() # 옵션 설정 
    options.add_argument("--headless") # 창 최대화
    options.add_experimental_option('detach', True) # 브라우저 세션을 분리해 백그라운드에서 실행 (브라우저 꺼짐 방지) 

    #웹 브라우저 제어
    driver = webdriver.Chrome(options = options)

    #위에서 지정한 url로 브라우저 열기
    driver.get(url)

    time.sleep(1)


    #검색창을 element 변수로 지정
    element = driver.find_element(By.ID, "queryInputHeader")
    element.send_keys(query_txt)

    # 검색어를 입력한 후 엔터키(\n) 입력을 통해 검색을 실행
    element.send_keys("\n")
    time.sleep(2)

    ##########

    # 첫 번째 목록 클릭
    driver.find_element(By.CSS_SELECTOR, "#ct > div.map_wrap > div.search_panel > div.list_contents > div > div > div:nth-child(2) > div > a").click()
    time.sleep(1)

    # 전체 거래방식 클릭
    driver.find_element(By.CSS_SELECTOR, "#complexOverviewList > div.list_contents > div.list_fixed > div.list_filter > div > div:nth-child(1) > button").click()
    time.sleep(1)

    item['아파트명'] = driver.find_element(By.CSS_SELECTOR, "#complexTitle").text

    # 전체거래방식 중 '전세' 클릭
    driver.find_element(By.CSS_SELECTOR, "#complexOverviewList > div.list_contents > div.list_fixed > div.list_filter > div > div:nth-child(1) > div > div > ul > li:nth-child(3) > label").click()
    time.sleep(1)
    # 전세로 거래방식 조정 후, X표 클릭 후 닫기
    driver.find_element(By.CSS_SELECTOR, "#complexOverviewList > div.list_contents > div.list_fixed > div.list_filter > div > div:nth-child(1) > div > button > i").click()

    # 매물 누르기
    매물_목록_2번째_열 = driver.find_element(By.CSS_SELECTOR, "#articleListArea > div:nth-child(2)")
    try:
        # 네이버에서 보기 버튼이 있으면 그 부분 누르기
        네이버에서_보기_버튼 = 매물_목록_2번째_열.find_element(By.CSS_SELECTOR, 'div.label_area > a')
        네이버에서_보기_버튼.click()
    except Exception as e:
        # 없으면 그냥 매물 버튼 누르기
        driver.find_element(By.CSS_SELECTOR, "#articleListArea > div:nth-child(2)").click()

    time.sleep(2)

    #시세/실거래가 클릭
    driver.execute_script("arguments[0].click();", driver.find_element(By.CSS_SELECTOR, "#detailTab2"))

    #매매 버튼 누르기
    driver.find_element(By.ID, "marketPriceTab1").click()

    time.sleep(2)

    ##매매가 가지고 오기
    item['매매가(억)'] = driver.find_element(By.CSS_SELECTOR, "#tabpanel1 > div:nth-child(6) > table > tbody > tr.type_emphasis > td:nth-child(3)").text
    print("매매가 가져오기 성공")

    #전세 버튼 누르기
    driver.find_element(By.ID, "marketPriceTab2").click()

    time.sleep(2)


    item['전세가(억)'] = driver.find_element(By.CSS_SELECTOR, "#tabpanel1 > div:nth-child(6) > table > tbody > tr.type_emphasis > td:nth-child(3)").text
    print("전세가 가져오기 성공")

    ##시공일자 가지고 오기
    item['시공일자'] = driver.find_element(By.CSS_SELECTOR, "#summaryInfo > dl > dd:nth-child(6)").text
    print("시공일자 가져오기 성공")


    time.sleep(1)
    driver.quit()

    print("끝")

    # %%
    # 가져온 값 딕셔너리 형태로 출력 
    print(item)

    # %%
    # 데이터 프레임으로 변경 
    df = pd.DataFrame(item,index = [0] )
    df

    # %%
    #한글로 얻어진 매매가와 전세가 숫자로 바꾸기

    df['전세가(억)'] = df['전세가(억)'].str.replace("억", "").str.replace(",", "").astype(float)
    df['매매가(억)'] = df['매매가(억)'].str.replace("억", "").str.replace(",", "").astype(float)

    #전세가와 매매가 억원 단위로 표현하기
    df['전세가(억)'] = df['전세가(억)']/10000
    df['매매가(억)'] = df['매매가(억)']/10000


    df.insert(3, '전세가율', df['전세가(억)']/df['매매가(억)'])

    # %%
    # 날짜 데이터 전처리
    df['시공일자_datetime'] = pd.to_datetime(df['시공일자'])
    df['시공일자_년도'] = df['시공일자_datetime'].dt.year
    df

    # %% [markdown]
    # ##   2. OCR(Optical Character Recognition) : 등기부등본에서 '압류'등의 키워드 추출과 근저당 금액 추출 

    # %%
    # OCR을 위한 라이브러리 설치 
    # ! pip install pdf2image
    # ! pip install opencv-python
    # ! pip install pytesseract

    # %%
    # 필요한 라이브러리 임포트


    # Tesseract 실행 경로 지정
    pytesseract.pytesseract.tesseract_cmd = r"c:\\Program Files\\Tesseract-OCR\\tesseract.exe"

    # PDF 파일 경로 및 Poppler 실행 경로 지정
    pdf_filename = '삼계서희스타힐스_108동_1504호.pdf'
    pdf_path = f'./media/uploads/{pdf_filename}'
    poppler_path = r'C:\Program Files\poppler-23.08.0\Library\bin'



    # PDF를 이미지로 변환
    images = convert_from_path(pdf_path, poppler_path=poppler_path)

    # 검색할 키워드 목록 정의
    keywords = ['가압류', '압류', '신탁', '소유권이전', '임차권등기명령', '근저당']

    # 키워드와 해당 키워드가 마지막으로 발견된 페이지 번호를 저장할 딕셔너리 초기화
    keyword_last_page = {keyword: None for keyword in keywords}
    # 키워드와 해당 키워드의 금액을 저장할 딕셔너리 초기화
    keyword_amounts = {keyword: None for keyword in keywords}
    # 주소와 아파트 이름 초기화
    address = None
    apartment_name = None

    # PDF의 각 페이지에 대해 처리 시작
    for page_num, image in enumerate(images, start=1):
        # 이미지를 NumPy 배열로 변환
        image_np = np.array(image, dtype=np.uint8)
        
        # 이미지에서 텍스트 추출
        tessdata_dir_config = '--tessdata-dir "<tesseract_data_path>"'
        text = pytesseract.image_to_string(image_np, lang='kor')

        # 주소 추출을 위한 정규식 패턴 설정
        address_pattern = re.compile(r'(?<=\[집합건물\]\s)(.*?)(?=\s제\d+층)')
        address_match = address_pattern.search(text)
        if address_match:
            address = address_match.group(1).strip()
        
        # 아파트 이름 추출을 위한 정규식 패턴 설정
        apartment_name_pattern = re.compile(r'(?<=층\s)(.*?)(?=\s\d+호)')
        apartment_name_match = apartment_name_pattern.search(text)
        if apartment_name_match:
            apartment_name = apartment_name_match.group(1).strip()
        
        # 각 키워드의 발견 여부 초기화
        keyword_found = {keyword: 0 for keyword in keywords}

        # 키워드 검색 시작
        for keyword in keywords:
            keyword_indices = [m.start() for m in re.finditer(keyword, text)]
            if keyword_indices:
                keyword_last_page[keyword] = page_num
                amount_found = False
                keyword_found[keyword] = 1

                # '근저당' 키워드의 경우 금액 추출 시도
                if keyword == '근저당':
                    for index in keyword_indices:
                        lines = text[index:].split('\n')
                        for line in lines:
                            # 금액 추출을 위한 정규식 패턴 설정
                            amount_pattern = re.compile(r'금\s*([0-9,]+)\s*원')
                            match = amount_pattern.search(line)
                            if match:
                                amount = match.group(1)
                                # 추출된 금액을 원하는 형식으로 변환
                                amount = float(amount.replace(',', '')) / 100000000
                                keyword_amounts[keyword] = amount
                                amount_found = True
                                break
                        if amount_found:
                            break

    # ...

    # PDF에서 추출한 데이터 결과 출력
    print("주소:", address)
    print("각 키워드 존재 여부 및 금액")
    for keyword in keywords:
        last_page = keyword_last_page[keyword]
        amount = keyword_amounts[keyword]
        found = keyword_found[keyword]
        
        if last_page is not None:
            if amount is not None:
                if keyword == '근저당':
                    formatted_amount = f"{amount:.3f}"
                    print(f" '{keyword}' : {formatted_amount}")
                else:
                    print(f" '{keyword}' : {amount}")
            else:
                amount = 0 if keyword == '근저당' else '0'  # '근저당'에는 0, 나머지에는 '0'으로 설정
                print(f" '{keyword}' : {amount}")
        else:
            print(f" '{keyword}' : 0")

    # ...

    # 등기부 등본 정보 데이터프레임 생성
    result_df = pd.DataFrame({
        '가압류': [keyword_found['가압류']],
        '압류': [keyword_found['압류']],
        '신탁': [keyword_found['신탁']],
        '소유권이전': [keyword_found['소유권이전']],
        '임차권등기명령': [keyword_found['임차권등기명령']],
        '근저당': [keyword_amounts['근저당'] if keyword_amounts['근저당'] is not None else 0],  # '근저당'에 값이 없으면 0으로 설정
        '지역': [address]
    })

    # 필요한 숫자만 남기는 함수
    def keep_only_numbers(value):
        if isinstance(value, str):
            return re.sub(r'[^0-9]', '', value)
        return value

    # 데이터 프레임의각 열에 대해 숫자만 남기도록 처리
    for col in result_df.columns:
        result_df[col] = result_df[col].apply(keep_only_numbers)

    # %%
    # 앞 서 크롤링 한 데이터와 등본에서 추출한 데이터 결합 
    final_df = pd.concat([df, result_df], axis=1)
    final_df['근저당정규화'] = (final_df['전세가(억)'] + final_df['근저당']) / final_df['매매가(억)']
    final_df = final_df[['전세가율','시공일자_년도','근저당정규화','신탁','압류','가압류','임차권등기명령']]
    final_df

    # %%
    # 기존의 X_test 데이터 새로운 매물 데이터와 결합하기 위해 맞춰주기  
    X_test  = pd.DataFrame(X_test)
    X_test.columns = ['전세가율','시공일자_년도','근저당정규화','신탁','압류','가압류','임차권등기명령']
    X_test

    # %%
    # 기존의 X_test 데이터 새로운 매물 데이터와 결합  
    X_test = pd.concat([X_test, final_df], axis = 0, ignore_index=True)
    X_test

    # %%
    # test 데이터에 대한 예측(predict_proba() 함수를 통해 '확률값' 으로 표현 , 우리의 목표는 전세사기일 확률이 ~~%이다. 라고 출력하는 것) 
    pred_svot = svot_clf.predict_proba(X_test)

    print(pred_svot)


    # %%
    # 크롤링과 ocr을 통해 얻어낸 새로운 매물 (마지막 행) 데이터에 대한 예측
    last = pred_svot[-1,1]
    print(f"사기일 확률은 : {last: .2%} 입니다")
    percent = f"{last:.2%}"

    # %% [markdown]
    # # 정규화 

    # %%
    # 정규화를 위한 라이브러리 호출 

    # %%
    # 거리 기반인 클러스터링을 위한 Scaling 작업
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_test)
    X_scaled = pd.DataFrame(X_scaled)
    X_scaled.columns = ['전세가율','시공일자_년도','근저당정규화','신탁','압류','가압류','임차권등기명령']

    # %% [markdown]
    # # 클러스터링 k 개수 선정 

    # %%

    # %%
    # k 개수를 정하기 위한 Elbow Method 관련 라이브러리 호출  

    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(1,10)) # 객체에 k가 1부터 10까지 상황 부여 
    visualizer.fit(X_scaled) # 이전에 스케일 한 데이터를 피팅 


    # %% [markdown]
    # ## label == 1 (전세 사기) & 사용자로부터 입력받은 새로운 매물에 대해 클러스터링 
    # ### (전세 사기를 클러스터링을 통해 유형화 했을 때 입력 받은 새로운 매물은 어디에 해당할 것인가) 

    # %%
    pred_svot = pd.DataFrame(svot_clf.predict(X_test)) # 소프트 보팅으로 예측한 결과를 dataframe 화 
    pred_svot.columns = ['label'] # 예측한 결과에 'label' column 명 부여  
    pred_svot  



    # %%
    # 크롤링, OCR 로 수집한 값들과 그에 대한 라벨링을 하나의 행으로 만들기  ( = 새로운 데이터 라고 지칭하겠습니다.)
    X_scaled = X_scaled.merge(pred_svot, how='left', left_index=True, right_index=True)  
    pred = X_scaled.iloc[-1]
    pred_df = pd.DataFrame(pred)
    pred_reshaped = pred_df.T
    pred_reshaped

    # %%
    # scaling 마친 기존의 test 데이터중 label 값이 1( = 사기라고 예측) 과 새로운 데이터를 행 결합 
    X_scaled = X_scaled[X_scaled['label'] == 1]
    X_scaled_pred = pd.concat([X_scaled,pred_reshaped],axis = 0)
    X_scaled_pred

    # %%
    # k - means 클러스터링 

    model = KMeans(n_clusters = 3)  
    model.fit(X_scaled_pred)

    # 클러스터링 결과를 'cluster' 열에 입력 
    X_scaled_pred['cluster'] =model.fit_predict(X_scaled_pred) 
    X_scaled_pred['cluster']
    X_scaled_pred

    # %%
    # 클러스터링 시각화 

    # variables = list(X_scaled_pred.columns)
    # index_to_highlight = len(X_scaled_pred) - 1 # 새로운 데이터에 대한 인덱싱 (이해를 돕고자 상수로 표현)
    #
    # for i in range(len(variables)):
    #     for j in range(i+1, len(variables)):
    #         plt.figure(figsize=(30, 18))
    #         plt.scatter(X_scaled_pred[variables[i]], X_scaled_pred[variables[j]], c='gray', alpha=0.5)
    #
    #         # 축 설정
    #         plt.xlabel(variables[i], fontsize=24)
    #         plt.ylabel(variables[j], fontsize=24)
    #
    #         # 제목 설정
    #         plt.title(f'{variables[i]} & {variables[j]}', fontsize=30)
    #
    #         # 각 데이터포인트의 인덱스 표시
    #         for idx, (x, y) in enumerate(zip(X_scaled_pred[variables[i]], X_scaled_pred[variables[j]])):
    #             if idx == index_to_highlight:  # 지정한 인덱스(새로운 데이터 (from 크롤링, ocr ))만 표시
    #                 plt.annotate(str(idx), (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=20, color='red')
    #
    #
    #         # K-means 클러스터링 결과 그래프로 보여주기
    #         plt.scatter(X_scaled_pred[variables[i]], X_scaled_pred[variables[j]], c = X_scaled_pred['cluster'], cmap='viridis', marker='o')
    #         plt.colorbar(label='Cluster Label')
    #         plt.rc("font", family = "Malgun Gothic")
    #         # plt.show()


    # 변수 설정
    x_variable = '전세가율'
    y_variable = '근저당정규화'
    z_variable = '시공일자_년도'

    # 3D 클러스터링 시각화
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    index_to_highlight = len(X_scaled_pred) - 1
    # 데이터 포인트 그리기
    scatter = ax.scatter(X_scaled_pred[x_variable], X_scaled_pred[y_variable], X_scaled_pred[z_variable],
                         c=X_scaled_pred['cluster'], cmap='viridis', marker='o')

    # 각 데이터포인트의 인덱스 표시
    for idx, (x, y, z) in enumerate(
            zip(X_scaled_pred[x_variable], X_scaled_pred[y_variable], X_scaled_pred[z_variable])):
        if idx == index_to_highlight:
            ax.text(x, y, z, str(idx), color='red', fontsize=10, backgroundcolor='white')

    # 축 레이블 설정
    ax.set_xlabel(x_variable)
    ax.set_ylabel(y_variable)
    ax.set_zlabel(z_variable)

    # 컬러바 추가
    cb = plt.colorbar(scatter, ax=ax, label='Cluster Label')
    cb.set_ticks(range(len(set(X_scaled_pred['cluster']))))
    cb.set_ticklabels(range(len(set(X_scaled_pred['cluster']))))


    save_folder = '../media/result/'

    # 이미지 파일명 및 경로
    image_file = '3Dresult.png'
    save_path = os.path.join(save_folder, image_file)

    # 이미지 저장
    plt.savefig(save_path)

    plt.show()

    return percent
