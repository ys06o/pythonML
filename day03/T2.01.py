# =========================================================
# 머신러닝 기초 개념 정리
# =========================================================

# [1] 모델(Model)
# 데이터를 보고 규칙을 배우는 프로그램
# 예) 물고기의 길이와 무게를 보고 종류를 맞춤

# ---------------------------------------------------------

# [2] 학습(Learning)
# 데이터를 보면서 규칙(패턴)을 찾는 과정
# 예) "길고 무거우면 도미일 가능성이 높구나!"

# ---------------------------------------------------------

# [3] 예측(Predict)
# 학습한 규칙으로 새로운 데이터의 결과를 맞추는 과정
# 예) 새 물고기가 도미인지 빙어인지 예측

# ---------------------------------------------------------

# [4] 특성(Feature)
# 모델에 입력되는 정보(데이터)
# 예) 길이, 무게, 색깔

# ---------------------------------------------------------

# [5] 타깃(Target)
# 모델이 맞춰야 하는 정답 데이터
# 예) 물고기 종류(도미, 빙어 등)

# =========================================================
# K-NN (K 최근접 이웃 알고리즘)
# =========================================================

# [6] K-NN 모델
# 가까운 데이터들을 참고해서 결과를 예측하는 방식
# 예) 주변에 도미가 많으면 도미로 판단

# ---------------------------------------------------------

# [7] 하이퍼파라미터 k
# 몇 개의 가까운 이웃을 참고할지 정하는 값
# 예)
# k=3 -> 가까운 3개 참고
# k=5 -> 가까운 5개 참고

# k값이 너무 작으면 -> 데이터 하나에 너무 민감함
# k값이 너무 크면 -> 엉뚱한 데이터까지 참고함

# =========================================================
# KNN 종류
# =========================================================

# [8] KNeighborsClassifier()
# 분류 모델
# 종류(Category)를 맞추는 모델
# 예) 도미/빙어 구분

# ---------------------------------------------------------

# [9] KNeighborsRegressor()
# 회귀 모델
# 숫자(Value)를 예측하는 모델
# 예) 집값, 점수 예측

# =========================================================
# 데이터 전처리
# =========================================================

# [10] 표준화(스케일링, Scaling)
# 데이터 크기를 비슷하게 맞추는 작업
# 이유:
# 숫자 차이가 크면 큰 값이 더 중요하게 인식될 수 있음

# ---------------------------------------------------------

# [11] StandardScaler
# 평균을 0, 표준편차를 1로 맞추는 대표적인 표준화 방법

# ---------------------------------------------------------

# [12] transform()
# 데이터를 실제로 변환하는 함수
# 예)
# scaler.transform(data)

# =========================================================
# 모델 학습 문제점
# =========================================================

# [13] 과소적합(Underfitting)
# 모델이 너무 단순해서 학습이 부족한 상태
# 예) 공부를 너무 안 한 상태

# 특징:
# - 학습 데이터도 잘 못 맞춤
# - 테스트 데이터도 잘 못 맞춤

# ---------------------------------------------------------

# [14] 과대적합(Overfitting)
# 모델이 데이터를 너무 외워버린 상태
# 예) 답만 외우고 응용은 못하는 상태

# 특징:
# - 학습 데이터는 매우 잘 맞춤
# - 새로운 데이터는 잘 못 맞춤


#[1]숭어의 '길이','무게'
import pandas as pd

df = pd.read_csv('./day03/Fish.csv')

fish_data = df[df['Species'].isin(['Perch'])]

perch_length = fish_data['Length2'].values
perch_weight = fish_data['Weight'].values

print(perch_length)
print(perch_weight)

# [2] 훈련 세트와 테스트 세트 분리
from sklearn.model_selection import train_test_split
train_input , test_input , train_target , test_target = train_test_split(perch_length,perch_weight,test_size=0.2,random_state=42)


#[3]학습 하기전에 사이킷런 모델들을 2차원 배열만 가능 
train_input=train_input.reshape(-1 , 1) #reshape(행개수,열개수):#-1 행은 자동 열은 1개
test_input=test_input.reshape(-1 , 1)

#[4]k-최근접 이웃 회귀 모델 훈련
from sklearn.neighbors import KNeighborsRegressor
knr=KNeighborsRegressor() #모델 객체 생성
knr.fit(train_input,train_target)
print(knr.score(test_input,test_target)) #모델 평가 0.9932626838364674

#[5]임의의 값으로 예측하기 ,임의의 물고기 길이 50를 넘어서 무게 예측
print(knr.predict([[50]]) ) #[1010.]
print(knr.predict([[100]])) #[1010.]

#문제점:K-최근접 이웃의 문제점은 주변 이웃의 평균으로 예측하기 때문에 최댓값을 벗어나면 항상 동일한 예측값이 나온다. 
#즉] 소규모 또는 간단한 예측 프로그램에서만 사용된다.



#[1] (단순)선형회귀 모델 #1차 방정식
from sklearn.linear_model import LinearRegression #선형회귀 모델
lr=LinearRegression() #모델 객체 생성
lr.fit(train_input,train_target) #모델 학습
print(lr.score(test_input,test_target)) #모델 평가
print(lr.predict([[50]])) #[1238.3175398] 길이가 50일때 무게 예측
print(lr.predict([[100]])) #[3191.00026354] 길이가 100일때 무게 예측

#직선 y=w(가중치)x(특성)+절편
#즉 무게=가중치*길이+절편
print('lr.coef',lr.coef_) #[39.05365447] #직선의 기울기(특성의 가중치)
print(lr.intercept_) #-714.3651839448922 #편향 x가 0일때  y의 값
#실 자료들은 물고기의 길이가 1씩 증가할때 무게가 꼭 비례 증가 하지않는다.
#초반에는 길이에 따라 무게가 3배 증가 하다가 중/후반에는 무게가 2/1배 증가 할 수 있다.
# 기울기 공식:x와y의 편차 곱의 합/x의 편차 제곱함
#[2] 시각화
import matplotlib.pyplot as plt
plt.scatter(train_input,train_target)
plt.xlabel('Length')
plt.ylabel('Weight')
plt.scatter(50,1238) #길이가 50일때 무게는 1238정도가 될것이다.
plt.scatter(100,3191)
#회귀선 그리기
plt.plot([15,100],lr.predict([[15],[100]])) #15최저 길이 #0 길이의 시작점 ,100 길이의 끝점
plt.show()


#(다항) 선형회귀 모델 #2차 방정식
import numpy as np
train_poly=np.column_stack((train_input**2,train_input)) #+더하기 , **제곱 #[길이제곱,길이]
print(train_poly)
lr=LinearRegression()
lr.fit(train_poly,train_target) #다항으로 학습

print(lr.predict([[50**2,50]]))

#여러개 예측
point=np.arange(15,50) #예측하고 싶은 범위
point_poly=np.column_stack((point**2,point))
#시각화
plt.scatter(train_input,train_target)
plt.plot(point,lr.predict(point_poly))
plt.show()
test_poly=np.column_stack((test_input**2,test_target))
print(lr.score(test_poly,test_target))

