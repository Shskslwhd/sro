# sro
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# Генерация данных
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Инициализация алгоритмов
linear_model = LinearRegression()
random_forest_model = RandomForestRegressor(n_estimators=10, random_state=42)
# Обучение алгоритмов
linear_model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)
# Предсказание
linear_predictions = linear_model.predict(X_test)
forest_predictions = random_forest_model.predict(X_test)
# Квазилинейная композиция
quasilinear_combination = 0.7 * linear_predictions + 0.3 * forest_predictions
# Оценка производительности
linear_mse = mean_squared_error(y_test, linear_predictions)
forest_mse = mean_squared_error(y_test, forest_predictions)
quasilinear_mse = mean_squared_error(y_test, quasilinear_combination)
print(f"Linear Model MSE: {linear_mse}")
print(f"Random Forest Model MSE: {forest_mse}")
print(f"Quasilinear Combination MSE: {quasilinear_mse}")
