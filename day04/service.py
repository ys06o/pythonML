from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import pandas as pd


class Service:
    def __init__(self):
        self.model = None
        self.poly = None
        self.scaler = None

    def 학습요청(self, car_list):
        # 1. df 만들기
        df = pd.DataFrame(car_list)
        train_data = df[["평균연비", "누적주행거리키로", "출고후경과월수", "사고감가건수", "소유자변경횟수"]]
        target_data = df["매매가격만원"].values

        train_input, test_input, train_target, test_target = train_test_split(
            train_data, target_data, test_size=0.2, random_state=42
        )

        # 2. 최적 모델 찾기
        optimization = []

        for degree in [1, 2, 3, 4, 5]:
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            poly.fit(train_input)
            train_poly = poly.transform(train_input)
            test_poly = poly.transform(test_input)

            lr = LinearRegression()
            lr.fit(train_poly, train_target)
            r2 = lr.score(test_poly, test_target)
            optimization.append({"r2": r2, "model": lr, "poly": poly, "scaler": None})

            ss = StandardScaler()
            ss.fit(train_poly)
            train_scaled = ss.transform(train_poly)
            test_scaled = ss.transform(test_poly)

            for alpha in [0.01, 0.1, 1, 10, 100]:
                ridge = Ridge(alpha=alpha)
                ridge.fit(train_scaled, train_target)
                r2 = ridge.score(test_scaled, test_target)
                optimization.append({"r2": r2, "model": ridge, "poly": poly, "scaler": ss})

                lasso = Lasso(alpha=alpha)
                lasso.fit(train_scaled, train_target)
                r2 = lasso.score(test_scaled, test_target)
                optimization.append({"r2": r2, "model": lasso, "poly": poly, "scaler": ss})

        # 3. 최적 모델 저장
        best = max(optimization, key=lambda x: x["r2"])
        self.model = best["model"]
        self.poly = best["poly"]
        self.scaler = best["scaler"]

        print(f"최적 모델: {self.model}, 결정계수: {best['r2']}")
        return True

    def 예측요청(self, car):
        if self.model is None:
            return "모델이 없습니다."
        data = [[
            car["평균연비"],
            car["누적주행거리키로"],
            car["출고후경과월수"],
            car["사고감가건수"],
            car["소유자변경횟수"],
        ]]

        data_poly = self.poly.transform(data)

        # 스케일링 적용 (사용한 모델만)
        if self.scaler is not None:
            data_poly = self.scaler.transform(data_poly)

        predict = self.model.predict(data_poly)
        print(predict)
        return float(predict[0])


service = Service()