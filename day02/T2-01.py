#[1] fish csv 불러오기
import pandas as pd
df=pd.read_csv('./day02/fish.csv')

#[2] Perch(농어)만 추출
perch=df[df['Species'].isin(['Perch'])]
print(perch)
perch.info() #56마리
#농어의 길이/무게 만 추출
perch_length=perch['Length2'].values
perch_weight=perch['Weight'].values
print(perch_length,perch_weight)

#[3] 농어 길이에 따른 무게 예측하기
import matplotlib.pyplot as plt
plt.scatter(perch_length,perch_weight)
plt.xlabel('Length')
plt.ylabel('Weight')
plt.show()

#[4] 학습 모델 만들기, (1) 학습용과 테스트용 분리를 한다.
#왜? 모델평가에 사용된다.
from sklearn.model_selection import train_test_split
#train_test_split(학습자료,정답자료,test_size=테스트 자료 비율,random_state=랜덤 시드값)
#random_state=분리할떄 사용되는 난수값 랜덤 시드값이 같으면 같은 방식으로 분리된다.
train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, test_size=0.3, random_state=42)

print(train_input.shape) #(39,) ,총 56개 중에 학습용 7에 해당하는 개수가 39개
print(test_input.shape) #(17,) ,총 56개 중에 테스트용 3에 해당하는 개수가 17개

#(2)자료형식(모양) 구성 사이킷런 모델 학습,대부분 2차원 을 사용한다.
import numpy as np
array=np.array([1,2,3,4])
print(array.shape) #shape는 배열의 모양을 알려준다. (4,) 1차원 배열

array2=np.array([[1,2],[3,4]])
print(array2.shape) #shape는 배열의 모양을 알려준다. (2, 2) 2차원 배열

print(train_input.shape) #(39,) 1차원 배열-->사이킷런 모델들은 1차원 배열 학습이 불가능하다.
print(train_input) #1차원으로 구성된 '농어'의 길이
#1차원->2차원
train_input=train_input.reshape(-1,1) #reshape(행,열) -1은 행의 개수를 자동으로 계산하라는 의미 자료 갯수 만큼
print(train_input)
print(train_input.shape) #(39, 1) 2차원 배열  
#train_target=train_target.reshape(-1,1) #(39, 1) 2차원 배열
#print(train_target.shape) #(39, 1) 2차원 배열
test_input=test_input.reshape(-1,1) #(17, 1) 2차원 배열
print(test_input.shape) #(17, 1) 2차원 배열
test_target=test_target.reshape(-1,1) #(17, 1) 2차원 배열
print(test_target.shape) #(17, 1) 2차원 배열

#[5] 모델 학습
from sklearn.neighbors import KNeighborsRegressor #k 최근접이웃 모델 찾기
from sklearn.neighbors import KNeighborsClassifier #k 최근접이웃 회귀
knr=KNeighborsRegressor() #모델 객체 생성
knr.fit(train_input,train_target) #모델 학습 #길이에 따른 무게 학습
#[6] 모델 평가
print(knr.score(test_input,test_target)) #모델 평가 #0.992809406101064
#회귀모델에서는 결정계수라고 한다. 1에 가까울수록 좋은 모델이다.

#[7] 임의의 값으로 모델 예측하기
print(test_input) #[8.4 18, 27.5] #길이
print(knr.predict(test_input)) #[61.4 78. 248.] #길이 8.4cm인 농어의 무게는 61.4g, 길이 18cm인 농어의 무게는 78g, 길이 27.5cm인 농어의 무게는 100g으로 예측


#[6]k최근접이웃 회귀는 이웃의 평균으로 예측 한다. 하이파라미터 k의 값을 조절
# 임의의 길이 생성 ,임의의 물고기 길이 5~45cm 사이의 45개 데이터 생성
x=np.arange(5,45).reshape(-1,1) #5~45 사이의 45개 데이터 생성
print(x)
knr=KNeighborsRegressor()
for k in [1,3,5,10]: #이웃 개수를 4가지(1,3,5,10)로 조절하면서 모델 학습
    knr.n_neighbors=k #이웃 개수 설정
    knr.fit(train_input,train_target) #모델 학습
    print(knr.score(test_input, test_target)) #총 4번의 학습 평가
    pred=knr.predict(x) #임의의 값으로 예측하기
    print(pred) #총 45개의 물고기 길이에 따른 무게 예측값
    plt.scatter(train_input,train_target) #학습용 데이터 시각화
    plt.plot(x,pred) #예측값 시각화 #plot(선차트이면서 회귀(예측)선으로 많이쓰임) #x=길이 ,pred=예측값(무게)
    plt.title(f"k={k}") #그래프 제목 설정
    plt.show()


#k는 이웃개수 뜻한다. k최근접 회귀는 이웃의 평균으로 예측한다.
#k가 1일때  0.9918926744767643 #특정한 자료에 튀는 데이터까지(노이즈/이상치)까지 과대 적합 훈련
#k가 3일때  0.9766857219041255
#k가 5일때  0.9929281790592219
#k가 10일때 0.9742254836937329 #많은 자료에 둔감하고 단순화 된 자료까지 적용 될 수 있으므로 예측이 망가 질 수 있다. 과소 적합 훈련

# k가 5일때 가장 균형적인 추세를 보인다. 회귀선이 너무 꺾이거나, 완만한 일직선이 되지 않고 적절한 곡선을 보여준다. k가 1일때는 과대적합, k가 10일때는 과소적합이 나타난다. k값을 조절하면서 모델의 성능을 평가하는 것이 중요하다.
#결론: 머신러닝에서는 가장 최적의 파라미터 값을 찾는 과정을 튜닝이라고 한다.

