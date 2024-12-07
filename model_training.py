import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv('heart.csv')

# Выбор признаков (включаем все важные столбцы)
X = data[['age', 'sex', 'bmi', 'chol', 'bp_sys', 'bp_dia', 
          'p_wave', 'heart_rate', 'qrs_duration', 'qt_interval', 'rr_interval']]
# Целевая переменная для классификации (наличие заболевания)
y_classification = data['disease']
# Целевая переменная для регрессии (продолжительность жизни)
y_regression = data['life']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_class_train, y_class_test = train_test_split(
    X, y_classification, test_size=0.2, random_state=42
)
_, _, y_reg_train, y_reg_test = train_test_split(
    X, y_regression, test_size=0.2, random_state=42
)

# Нормализация признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Модель классификации (наличие заболевания)
classification_model = RandomForestClassifier(n_estimators=100, random_state=42)
classification_model.fit(X_train_scaled, y_class_train)

# Предсказание для классификации
y_class_pred = classification_model.predict(X_test_scaled)
classification_accuracy = accuracy_score(y_class_test, y_class_pred)
print(f'Classification Model Accuracy: {classification_accuracy:.2f}')

# Модель регрессии (продолжительность жизни)
regression_model = RandomForestRegressor(n_estimators=100, random_state=42)
regression_model.fit(X_train_scaled, y_reg_train)

# Предсказание для регрессии
y_reg_pred = regression_model.predict(X_test_scaled)
regression_mae = mean_absolute_error(y_reg_test, y_reg_pred)
print(f'Regression Model MAE: {regression_mae:.2f}')

# Генерация данных для графика (пример одного пациента)
ages = np.arange(30, 80, 1)  # Возраст от 30 до 80 лет
risks = []
for age in ages:
    temp_data = np.array([[age, 1, 25.5, 200, 120, 80, 0.1, 75, 0.08, 0.4, 0.6]])  # Пример данных
    temp_scaled = scaler.transform(temp_data)
    risk = classification_model.predict_proba(temp_scaled)[0][1]  # Вероятность наличия заболевания
    risks.append(risk)

# Построение и сохранение графика
plt.figure(figsize=(8, 5))
plt.plot(ages, risks, marker='o', color='red', label='Риск сердечно-сосудистого заболевания')
plt.title('Увеличение риска сердечно-сосудистых заболеваний с возрастом')
plt.xlabel('Возраст')
plt.ylabel('Вероятность риска')
plt.legend()
plt.grid(True)

# Сохранение графика в папку static
plt.savefig('static/graph.png')
plt.close()

# Сохранение моделей и масштабатора
joblib.dump(classification_model, 'heart_disease_classifier.pkl')
joblib.dump(regression_model, 'life_expectancy_regressor.pkl')
joblib.dump(scaler, 'scaler.pkl')
