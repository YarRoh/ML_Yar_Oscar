from flask import Flask, request, render_template
import joblib
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

# Загрузка моделей и масштабатора
classifier_model = joblib.load('heart_disease_classifier.pkl')
regressor_model = joblib.load('life_expectancy_regressor.pkl')
scaler = joblib.load('scaler.pkl')

# Инициализация приложения Flask
app = Flask(__name__)

# Главная страница (форма ввода данных)
@app.route('/')
def index():
    return render_template('form.html')

# Обработка POST-запроса с данными
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Получение данных из формы
        age = float(request.form.get('age', 0))
        gender = float(request.form.get('gender', 0))  # 0 - Женщина, 1 - Мужчина
        bmi = float(request.form.get('bmi', 0))
        cholesterol = float(request.form.get('cholesterol', 0))
        bp_sys = float(request.form.get('bp_sys', 0))
        bp_dia = float(request.form.get('bp_dia', 0))
        p_wave = float(request.form.get('p_wave', 0.1))  # По умолчанию 0.1
        heart_rate = float(request.form.get('heart_rate', 0))
        qrs_duration = float(request.form.get('qrs_duration', 0))
        qt_interval = float(request.form.get('qt_interval', 0))
        rr_interval = float(request.form.get('rr_interval', 0))

        # Подготовка данных для модели
        data = np.array([[age, gender, bmi, cholesterol, bp_sys, bp_dia,
                          p_wave, heart_rate, qrs_duration, qt_interval, rr_interval]])
        data_scaled = scaler.transform(data)

        # Прогноз классификации (наличие заболевания)
        disease_prediction = classifier_model.predict(data_scaled)
        disease_result = 'Высокий риск сердечных заболеваний' if disease_prediction[0] == 1 else 'Низкий риск сердечных заболеваний'

        # Прогноз продолжительности жизни
        life_expectancy_prediction = regressor_model.predict(data_scaled)[0]

        # Генерация данных для графика
        ages = np.arange(age, life_expectancy_prediction + 1, 1)  # Возраст от текущего до прогнозируемого
        risks = []
        for a in ages:
            temp_data = np.array([[a, gender, bmi, cholesterol, bp_sys, bp_dia,
                                   p_wave, heart_rate, qrs_duration, qt_interval, rr_interval]])
            temp_scaled = scaler.transform(temp_data)
            risk = classifier_model.predict_proba(temp_scaled)[0][1]  # Вероятность наличия заболевания
            risks.append(risk)

        # Построение графика
        plt.figure(figsize=(8, 5))
        plt.plot(ages, risks, marker='o', color='red', label='Риск сердечно-сосудистого заболевания')
        plt.title('Увеличение риска сердечно-сосудистых заболеваний с возрастом')
        plt.xlabel('Возраст')
        plt.ylabel('Риск заболевания')
        plt.legend()
        plt.grid(True)

        # Сохранение графика в буфер
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        graph_url = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()

        # Возврат результата
        return render_template(
            'result.html',
            disease_result=disease_result,
            life_expectancy=f'{life_expectancy_prediction:.2f} лет',
            graph_url=graph_url
        )
    except Exception as e:
        # Логирование и обработка ошибок
        print(f"Ошибка: {str(e)}")
        return f"Произошла ошибка: {str(e)}", 400

# Запуск приложения
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
