Быстрый старт через Colab доступен тут: [colab-link](https://colab.research.google.com/drive/1HfTpGhQywIUXeG0_2LbhyG8xz15PnT6T?usp=sharing).

API
POST /predict
Тело:
{ "text": "Пример текста" }
Ответ:
{ "is_toxic": false, "score": 0.01, "error": "" }

Пример с curl (использован ngrok адрес для колаба)
curl -X POST "https://6eb9-34-73-17-254.ngrok-free.app/predict" -H "Content-Type: application/json" -d "{\"text\": \"Что за долбанная хрень?\"}"
{"error":"","is_toxic":true,"score":0.9735844135284424}
