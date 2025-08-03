import pickle
import tensorflow as tf

print("🔄 Загружаем модель...")
model = tf.keras.models.load_model("email_model.h5")
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

print("✅ Модель и векторизатор загружены!")

def predict_email(text):
    X = vectorizer.transform([text])
    pred = model.predict(X)
    class_idx = pred.argmax(axis=1)[0]
    confidence = pred[0][class_idx]
    if class_idx == 1:
        return f"📌 ВАЖНОЕ письмо (достоверность {confidence:.2f})"
    else:
        return f"✉️ НЕважное письмо (достоверность {confidence:.2f})"

samples = [
    "Ваш код подтверждения: 12345",
    "Скидки на одежду в вашем городе!",
    "Квитанция об оплате заказа №123",
    "Тренировка на 20 минут для здоровья"
]

print("\n=== Результаты теста ===")
for s in samples:
    result = predict_email(s)
    print(f"\nТекст: {s}\n→ {result}")
