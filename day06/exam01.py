import pandas as pd

df=pd.read_csv('./day06/wine.csv')

data=df[['alcohol','sugar','pH']] #와인들의 속성 3개
target=df['class'] #1.화이트와인, 0: 레드와인
print(data) 
print(target)

from sklearn.model_selection import train_test_split

train_input,test_input,train_target,test_target=train_test_split(data,target,random_state=42)


#[2] 결정트리(분류 모델)
from sklearn.tree import DecisionTreeClassifier # 의사결정 트리 분류 모델

#모델 객체 생성
dt=DecisionTreeClassifier()

#모델 학습
dt.fit(train_input,train_target) #모델 학습

#모델 정확도 측정
print(dt.score(train_input,train_target)) #모델 정확도  0.9973316912972086
print(dt.score(test_input,test_target))#모델 정확도  0.8516923076923076

#모델 예측
print(dt.predict(test_input[:5])) #5개만 예측 [1. 0. 1. 1. 1.]

#[3] 결정 트리 시각화
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree #트리 시각화할때 사용
plot_tree(dt,max_depth=3,feature_names=['alcohol','sugar','pH'],class_names=['Red Wine','White Wine'],filled=True) #plot_tree(트리모델,max_depth=가지개수)
#max_depth=뿌리의 최대개수, 트리의 최대 깊이,과대적합 억제 #가지수가 너무 많아지면 과대 적합 발생
plt.show()

#트리:전체적인 구조
#노드:사각형 상자 하나하나가 노드 #가장 위에 있는 노드를 루트노드

#노드 속성:value[예측타갯수] #[85,2097] 0으로 예측하는 수가 85개, 1으로 예측하는 수가 2097개 
# -gini= 불순도 0.075 #0으로 가까울수록 정확하다 숫자가 높을 수록 부정적인 수치 #0.5 가까울수록 혼란하다

# sugar 특성= sugar<=4.15 보다 작으면  true(왼쪽노드로 이동),false(오른쪽노드로 이동)

#[4] 특성 중요도
print(dt.feature_importances_) #각 특성이 트리 모델에 얼마나 중요한 역할 하는지 수치
# [0.23320804 0.51815804 0.24863392]
print(dt.feature_importances_[0]) #alcohol #sugar #pH

#[5]최소한의 불순도(gini) 설정 ,최적의 파라미터
dt=DecisionTreeClassifier(random_state=42, min_impurity_decrease= 0.0005)
dt.fit(train_input,train_target) #학습
print(dt.score(train_input,train_target)) #0.8975779967159278 과대적합 최소화
print(dt.score(test_input,test_target)) #0.8590769230769231

