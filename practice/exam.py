# PythonML Practice 8: SGDClassifier 기반 취업 여부 분류 예측
# 데이터 출처: https://www.kaggle.com/datasets/shambhurajejagadale/student-performance-prediction-dataset
# [1] 데이터 전처리 및 독립/종속 변수 분할
# 제시된 학생 데이터셋에서 취업 여부를 나타내는 범주형 문자열 데이터인 placement_status (Placed, Not Placed)를 종속변수(타깃)로 설정하시오.
# 타깃을 제외한 study_hours, attendance 등 총 6개의 수치형 특성을 독립변수로 지정하고, 전체 데이터를 학습 세트와 검증 세트 비율 8:2로 정확하게 분리하시오.
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('./practice/student_dataset_10000_rows.csv')

student_target = df['placement_status']
student_input = df[['study_hours', 'attendance', 'sleep_hours', 'internet_usage', 'assignments_completed', 'previous_score']]

train_input, test_input, train_target, test_target = train_test_split(student_input, student_target, test_size=0.2, random_state=42)

# [2] 특성 다항 확장 및 경사하강법 필수 스케일링
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=3,include_bias=False)
poly.fit(train_input)

train_poly=poly.transform(train_input)
test_poly=poly.transform(test_input)

ss=StandardScaler()
ss.fit(train_poly)

train_scaled=ss.transform(train_poly)
test_scaled=ss.transform(test_poly)




# [3] 로그손실 또는 힌지손실 기반의 규제 하이퍼파라미터

from sklearn.linear_model import SGDClassifier
sc=SGDClassifier(loss='log_loss',random_state=42,
max_iter=10,tol=None,alpha=0.0001)
sc.fit(train_scaled,train_target)
print(sc.score(test_scaled,test_target))
print(sc.predict(test_scaled[:3]))


#점진적 핛브
sc.partial_fit(train_scaled,train_target)
print(sc.score(test_scaled,test_target))

sc=SGDClassifier(loss='log_loss',random_state=42)

train_score=[]
test_score=[]

import numpy as np 
classes = np.unique( train_target )

for i in range( 0 , 150 ) :  # 150번 반복 
    sc.partial_fit( train_scaled , train_target , classes= classes )   # 1학습 
    train_score.append( sc.score( train_scaled , train_target  ) )
    test_score.append( sc.score( test_scaled , test_target ) )
    
    
import matplotlib.pyplot as plt
plt.plot( train_score  )    # 학습용 정확도 점수 
plt.plot( test_score )      # 테스트용 정확도 점수 
plt.show( )



sc=SGDClassifier(loss='log_loss',max_iter=100,random_state=42,alpha=0.0001)
sc.fit(train_scaled,train_target)


print( sc.score( test_scaled , test_target ) )


# [4] 최고 정확도(Score) 선정 , *0.90 이상 찾기* 

# [5] 신규 학생 데이터 취업 예측 (추론 함수 구현 및 검증)

# 구현된 모델에 아래 두 가지 샘플 데이터를 입력하여  취업 성공(Placed)과 실패(Not Placed)를 올바르게 분류해내는지 최종 검증하시오.
#  - 샘플 A : study_hours=9, attendance=95, sleep_hours=7, internet_usage=2, assignments_completed=18, previous_score=85
#  - 샘플 B : study_hours=2, attendance=60, sleep_hours=5, internet_usage=9, assignments_completed=4, previous_score=50


sample=[[9,95,7,2,18,85],[2,60,5,9,4,50]]


train_poly=poly.transform(sample)

train_scaled=ss.transform(train_poly)
print(sc.predict(train_scaled))

