import pandas as pd

df=pd.read_csv('./day06/wine.csv')

data=df[['alcohol','sugar','pH']] #와인들의 속성 3개
target=df['class'] #1.화이트와인, 0: 레드와인
print(data) 
print(target)

from sklearn.model_selection import train_test_split

train_input,test_input,train_target,test_target=train_test_split(data,target,random_state=42)

#[2] 결정 트리(분류 모델)
from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier(random_state=42)
dt.fit(train_input,train_target)
print(dt.score(test_input,test_target)) #0.8516923076923076

#[3] 교차 검증
from sklearn.model_selection import cross_validate
# cross_validate(학습모델,학습세트,정답세트)
scores=cross_validate(dt,train_input,train_target)

#교차검증:전체 데이터를 N등분 (폴드) 하여 돌아가면서 검증을 한다. #기본값은 5등분

#즉 데이터를 여러 조각으로 나누어 학습하는 방법이다.
print(scores) #{'fit_time': array([0.00571513, 0.00613046, 0.00546026, 0.00541782, 0.00571179]), 'score_time': array([0.00144958, 0.00156426, 0.00174069, 0.0015595 , 0.00165915]), 'test_score': array([0.85128205, 0.84820513, 0.8788501 , 0.85112936, 0.84394251])}

import numpy as np
print(np.mean(scores['test_score'])) #5등분 학습의 평균 검증 점수 #0.8546818301479492



#[3]
from sklearn.model_selection import StratifiedKFold 
#분류 모델에서 #데이터를 N등분하여 교차 검증을 수행한다.
# StratifiedKFold(n_splits=N등분,shuffle=)
splits=StratifiedKFold(n_splits=10,shuffle=True,random_state=42)
scores=cross_validate(dt,train_input,train_target,cv=splits)
print(scores)
print(np.mean(scores['test_score'])) #0.8585800484734237 #약간의 증가



#[4] 그리드 서치,최적의 파라미터(변수/학습에 필요한 설정값)
from sklearn.model_selection import GridSearchCV

#(1) 여러개 최소 불순도 설정 ,불순도란?  0에 가까울수록 예측값이 명확하다. #0.5에 가까울수록 예측값이 애매하다.
#임의의 최소 불순도를 넣어서 리스트로 구성
params={'min_impurity_decrease':[0.0001,0.0002,0.0003,0.0004,0.0005]}

#(2)
#GridSearchCV(DecisionTreeClassifier(트리모델),{파리미터들})
#n_jobs=-1 컴퓨터내 모든 cpu 코어(연산의 흐름 단위)를 사용하여 병렬 연산 즉 cpu 최대 효윻
gs=GridSearchCV(DecisionTreeClassifier(random_state=42),params,n_jobs=-1)
#(3) 그리드 서치 학습
gs.fit(train_input,train_target) #기본값으로 교차 검증 5번
dt=gs.best_estimator_                  #최적의 파라미터로 학습 결과 
print(dt.score(test_input,test_target)) #0.8670769230769231
print(gs.best_score_) #0.8731517927657558
print(gs.cv_results_ ) #기본값으로 교차검증 5가 적용된다.



#[5]다중 파라미터
params={
  'min_impurity_decrease':np.arange(0.0001,0.001,0.0001), #0.0001 씩 증가
  'max_depth':range(5,20,1), #
  #노드 분할시 최저 샘플수
  'min_samples_split':range(2,100,10),
  #리프노트
  'min_samples_leaf':range(1,100,10)
}


gs=GridSearchCV(DecisionTreeClassifier(random_state=42),params,n_jobs=-1,cv=5) #대략 13000 가지 조합으로 학습
gs.fit(train_input,train_target)
print(gs.best_params_)
print(gs.best_score_) #0.8756162796819881



#[6]랜덤 서치
#조합 수가 많아지면 연산량이 많아져서 서버에 부하가 발생 할 수 있다.
#랜덤서치란? 고정된 값이 아니라 '확률 분포 함수' 를 제공하여 무작위로 숫자를 뽑아 학습한다.
from sklearn.model_selection import RandomizedSearchCV
#대략  13000개 조합에서 100개만 무작위로 추출 #교차검증 5 =>500번 학습
rs=RandomizedSearchCV(DecisionTreeClassifier(random_state=42,),params,n_jobs=-1,n_iter=100,cv=5)
rs.fit(train_input,train_target)
print(rs.best_params_)
print(rs.best_score_) #0.8710992470910336 #학습속도는 빨라졌지만 정확도가 조금 낮아짐

#최고의 파라미터로 예측한다.
dt=rs.best_estimator_
print(dt.predict())
