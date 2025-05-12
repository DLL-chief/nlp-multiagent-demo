# Продакшен версия.
# Дев-версия собрана в гугл-колабе. См. README

import os
from flask import Flask, request, jsonify
from detoxify import Detoxify
import torch

MODEL_TYPE = os.environ.get("MODEL_TYPE", "multilingual")  
TOXICITY_THRESHOLD = float(os.environ.get("TOXICITY_THRESHOLD", 0.2))  # По умолчанию порог 0.2

# Оптимизация под CPU / GPU (если есть)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Загрузка Detoxify модель {MODEL_TYPE} на {device}...")
detox = Detoxify(MODEL_TYPE, device=device)  # Подъем на доступном устройстве

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        text = data.get('text', '')
        if not isinstance(text, str) or not text.strip():
            return jsonify({"is_toxic": None, "score": None, "error": "Текст не предоставлен"}), 400

        # Минимизация нагрузки — no_grad
        with torch.no_grad():
            result = detox.predict(text)
        score = float(result.get('toxicity', 0))
        is_toxic = score > TOXICITY_THRESHOLD

        return jsonify({
            "is_toxic": is_toxic,
            "score": score,
            "error": ""
        }), 200

    except Exception as e:
        return jsonify({
            "is_toxic": None,
            "score": None,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    # Для Docker — лучше host="0.0.0.0"
    app.run(host="0.0.0.0", port=5001, debug=False)
