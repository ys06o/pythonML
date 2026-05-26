import pandas as pd

df=pd.read_csv('./day06/wine.csv')

data=df[['alcohol','sugar','pH']] #와인들의 속성 3개
target=df['class'] #1.화이트와인, 0: 레드와인


from sklearn.model_selection import train_test_split

train_input,test_input,train_target,test_target=train_test_split(data,target,random_state=42)

#트리의 앙상블(ensemble): 학습한 모델에서 오답들을 서로 상쇄하고 정답을 강화 하여 예측 정확도를 높여 과대적합을 방지하는 방법
#여러가지 방법이 존재

#[2] 렌덤포레스트
#결정트리는 전체 특성('alcocho,sugar,pH) 가장 영향력 있는 특성으로 예측 결정하는 방법
#(한쪽 특성엠나 과대적합 발생 할 수도있다.)
#랜덤포레스트 모든 특성을 사용한다.
#부트스트랩 샘플링: 전체 훈련데이터 중에서 무작위로 샘플 선정한다.
# 무작위 특성":전체 특성 중에서 무작위로 샘플 선정한다.
#즉] 모든 특성들을 사용하여 다양한 트리를 구성한다.

#obb: 무작위 이긴하지만, 중복 허용 선정시 1번도 선정 안된 자료들을 평가용으로 사용
#obb_score=True
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(oob_score=True , n_jobs=-1 , random_state=42 )


#교차검증 기본값 5
from sklearn.model_selection import cross_validate
scores=cross_validate(rf,train_input,train_target,n_jobs=-1)
print(scores)

import numpy as np
print(np.mean(scores['test_score'])) #0.8914208392565683

#특성 중요도
rf.fit(train_input,train_target)
print(rf.feature_importances_) #[0.23155241 0.49706658 0.27138101] #즉] 결정트리 보다 조금 더 골고루 분산 되었다.

#분류 모델중에서는 로지스틱회귀모델 vs 트리모델(앙상블)


#[3]엑스트라 트리 모델
#랜덤포레스트 중복허용한 무작위 샘플 추출
#전체 데이터를 학습
# 모든 트리가 전체 샘플 자료를 학습한다.
# 노드 분리할 때 무작위 노드 분할: 예:sugar 특성을 무작위로 1.4 기준으로 분리한다.  #무작위라서 오답 발생
#예시 '나이' 특성에  20세~60세가 존재하는 경우 노드분할 예시
#  Tree(노드1)에서 무작위로 나이 특성을 29세 이상 조건을 만든다.(수학적인 계산 x) 빠르다
#  Tree(노드2)에서 무작위로 나이 특성을 50세 이상 조건을 만든다.
# 즉 노드마다 서로 다른 기준점을 분할하여 다양성 확보
#계산식이 없어서 허술한 방법이지만 학습수와  방대한 양으로 오차를 극복할 수 있다.
from sklearn.ensemble import ExtraTreesClassifier
et=ExtraTreesClassifier(n_jobs=-1,random_state=42) #모델 생성

scores=cross_validate(et,train_input,train_target,n_jobs=-1)
print(scores)
print(np.mean(scores['test_score'])) #0.8903937240035804

#특성 중요도
et.fit(train_input,train_target)
print(et.feature_importances_) #[0.20702369 0.51313261 0.2798437 ]

#[4] 그레이디언트 부스팅
#랜덤 포레스트: 중복을 허용한 무작위 샘플/특성 선정하여 학습
#엑스트라 트리: 무작위로 노드를 분할 기준을 선정하여
from sklearn.ensemble import GradientBoostingClassifier
gb=GradientBoostingClassifier(random_state=42) #객체 생성
scores=cross_validate(gb,train_input,train_target,n_jobs=-1)
print(scores)
print(np.mean(scores['test_score'])) #0.8715107671247301
#특성 중요도
gb.fit(train_input,train_target)
print(gb.feature_importances_) #[0.12517641 0.73300095 0.14182264]
# sugar 에 좀 더 집중된 결과


#히스트그램 기반 그레이디언트 부스팅

#특성 정량화: 연속적인 구간을 256개 구간으로 나누어서 단순화 한다.
#분할 기준: 자식노드를 만들때 256개 구간 기준으로 분할 한다. <빠르다>

#예:
#180~181 까지 하나의 구간으로 묶어서 계산한다.
#미세한 소수점 오차는 과감하게 버린다. 메모리 절약과 속도 향상 한다.

from sklearn.ensemble import HistGradientBoostingClassifier
hgb=HistGradientBoostingClassifier(random_state=42)
scores=cross_validate(hgb,train_input,train_target,n_jobs=-1)
print(scores)
print(np.mean(scores['test_score'])) #0.8805410414363187

#앙상블(앞전 계산에 사용된 오차/결과를 다음/전체에 정확도를 향상 하는데 상쇄 하는 방법)

#그레이디언트 부스팅:부모노드에서 오차를 자식노드에게 전달하는 모델 학습이다.
#히스토그램 그레이디언트 부스팅: 연속된 샘플들을 구간(256)개 만들어서 모델 학습,전처리 시간이 부족하거나 학습 속도를 개선할 때


#외부 라이브러리 앙상블

# 1.
#pip install  xgboot (캐글 대회에서 나온 알고리즘)
from xgboost import XGBClassifier

#모델 생성
xgb=XGBClassifier(tree_method='hist',random_state=42)
scores=cross_validate(xgb,train_input,train_target,n_jobs=-1)
print("xgb=",np.mean(scores['test_score'])) #xgb= 0.8834147317432738

#2. pip install lightgbm(ms 회사 에서 나온 알고리즘)
from lightgbm import LGBMClassifier
lgb=LGBMClassifier(random_state=42)
scores=cross_validate(lgb,train_input,train_target,n_jobs=-1)
print("lgb=",np.mean(scores['test_score'])) #lgb= 0.8846461327857632

#3.pip install catboost

from catboost import CatBoostClassifier
cat=CatBoostClassifier(random_state=42)
scores=cross_validate(cat,train_input,train_target,n_jobs=-1)
print("cat=",np.mean(scores['test_score'])) #cat= 0.8809519296582952





























































































































































