"""
Продакшен Flask API for text generation (LLM).
Учебный Colab-демо сервис и быстрый старт — см. README.md
"""

import os
from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import torch

# ---- Конфиг ----
HUGGINGFACE_TOKEN = os.environ.get("HF_TOKEN", "ВАШ_HF_API_TOKEN")  # Лучше передавать через переменную среды!
MODEL_NAME = os.environ.get("MODEL_NAME", "google/gemma-3-4b-it")

login(token=HUGGINGFACE_TOKEN)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
)

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json(force=True)
        prompt = data.get('prompt', None)
        max_new_tokens = int(data.get('max_new_tokens', 50))
        temperature = float(data.get('temperature', 0.7))

        if not prompt or not isinstance(prompt, str):
            return jsonify({
                "generated_text": None,
                "error": "Некорректный или пустой prompt"
            }), 400

        with torch.no_grad():
            result = pipe(
                prompt,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=1
            )
        text = result[0]["generated_text"]

        return jsonify({
            "generated_text": text,
            "error": ""
        }), 200

    except Exception as e:
        return jsonify({
            "generated_text": None,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
