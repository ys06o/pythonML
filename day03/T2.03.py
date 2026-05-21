import pandas as pd
fish_data=pd.read_csv('./day03/Fish.csv')


#[1] 숭어의 '길이' '높이' '두께'(3가지 특성) 무게 1가지 타겟
perch=fish_data[fish_data['Species']=='Perch']

perch_full=perch[['Length2','Height','Width']]
perch_weight=perch['Weight'].values

#[2] 훈련 세트 와 테스트 세트를 분리
from sklearn.model_selection import train_test_split

train_input,test_input,train_target,test_target=train_test_split(perch_full,perch_weight,test_size=0.2,random_state=42)

#[3] 특성 공학 다항 특성 제공
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures()
poly.fit([[2]])
print(poly.transform([[2]]))


#예제 2
poly=PolynomialFeatures(include_bias=False)
poly.fit([[2,3]])
print(poly.transform([[2,3]]))

#적용
poly=PolynomialFeatures(include_bias=False) #다항 특성 객체 생성
poly.fit(train_input) #학습할 특성들을 대입한다.
train_poly=poly.transform(train_input) #학습한 특성들을 변환한다.
test_poly=poly.transform(test_input) #학습한 특성들을 변환한다.
print(train_poly) #3가지 특성--> 9가지 특성으로 변환됨


#[4] 다항 회귀
from sklearn.linear_model import LinearRegression # 회귀 모델
lr=LinearRegression() #회귀 모델 객체 생성
lr.fit(train_poly,train_target) #회귀 모델 학습

#[5] 평가
print(lr.score(train_poly,train_target)) #훈련 세트의 결정계수 R^2\

#계수란? 기울기와 가중치 뜻한다. #즉 어떠한 예측 결과에 얼마나 중요한 비중을 차지 하는지
#결정계수란? 0~1 사이의 값으로 표현되는 모델의 성능 지표로, 1에 가까울수록 모델이 데이터를 잘 설명한다는 것을 의미한다.
# 결정게수 k-nn 모델은  계산식 근접한 이웃을 이용한 계산식 이므로
  #타깃의 총 변동량 =SS_TOT=sum((실제값-평균값)**2)
  
#타깃의 총 변동량=SS_TOT=sum((실제값-평균값)**2)
#타깃의 오차 변동량=SS_RES=sum((실제값-예측값)**2)
#1(100%)-(ss_res/ss_tot)=R^2


#[6]과대 적합 확인
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly = PolynomialFeatures(degree=5, include_bias=False)

poly.fit(train_input)

train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)

print(train_poly.shape)  # (40, 21)

lr = LinearRegression()
lr.fit(train_poly, train_target)   

print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))

train_scaled = train_poly / np.sqrt(np.sum(train_poly ** 2, axis=0))

#릿지/라쏘 회귀들은 과적합된 자료들을 자동으로 제거 해준다.

#[8]릿지 회귀 : 가중치 줄여가면서 완전한 선 만들기 목적
from sklearn.linear_model import Ridge
ridge=Ridge() #릿지 회귀 객체 생성
ridge.fit(train_scaled,train_target) #릿지 회귀 모델 학습

alpha_list=[0.001,0.01,0.1,1,10,100]

for alpth in alpha_list:
    ridge=Ridge(alpha=alpth)
    ridge.fit(train_scaled,train_target)
    print(alpth,ridge.score(train_scaled,test_target))

    print(ridge.score(train_scaled,train_target))
    print(ridge.score(test_scaled,test_target))
#[9] 라쏘 회귀: 서로 특성 간의 관계없는 특성들을 제거하는게 목적
from sklearn.linear_model import lasso

#하.. 왤케 어렵냐

  

