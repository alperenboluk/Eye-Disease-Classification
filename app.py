from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

app = Flask(__name__)
CORS(app)

try:
    print("🧠 3 model yükleniyor...")
    models = [
        load_model("ensemble_model_1.keras", compile=False),
        load_model("ensemble_model_2.keras", compile=False),
        load_model("ensemble_model_3.keras", compile=False)
    ]
    print("✅ Tüm modeller yüklendi.")

    print("🔖 Sınıf etiketleri yükleniyor...")
    class_labels = joblib.load("ensemble_class_labels.pkl")  # dict: {0: 'cataract', ...}
    print("✅ Etiketler yüklendi.")
except Exception as e:
    print("❌ Model/etiket yüklenemedi:", e)

@app.route("/predict", methods=["POST"])
def predict():
    print("📥 İstek alındı.")
    if 'image' not in request.files:
        return jsonify({'error': "No image uploaded"}), 400

    file = request.files['image']
    print(f"📄 Dosya: {file.filename}")

    try:
        img = Image.open(file).convert("RGB")
        img = img.resize((224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        print("🤖 Ensemble tahmini yapılıyor...")
        preds = [model.predict(img_array)[0] for model in models]
        final_pred = np.mean(preds, axis=0)

        class_index = np.argmax(final_pred)
        class_name = class_labels[class_index]
        confidence = float(final_pred[class_index])

        # Tüm sınıflar ve olasılıkları
        all_probs = {
            class_labels[i]: float(final_pred[i])
            for i in range(len(final_pred))
        }

        print(f"✅ Tahmin: {class_name} ({confidence:.2f})")

        return jsonify({
            'class': class_name,
            'confidence': confidence,
            'all_probs': all_probs  # yeni eklendi
        })

    except Exception as e:
        print("❌ Tahmin hatası:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
