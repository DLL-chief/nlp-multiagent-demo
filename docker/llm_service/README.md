Быстрый старт (Colab)
Смотрите демо-версию здесь: [Colab demo link](https://colab.research.google.com/drive/1yqrM_lKUaAtxoo_cWdUy8sBxTh0XRJu9?usp=sharing)

API
POST /generate
Тело запроса:

{ "prompt": "Напиши приветствие на японском", "max_new_tokens": 20 }
Ответ:

{ "generated_text": "こんにちは！はじめまして。どうぞよろしくお願いいたします。", "error": "" }

Пример (c ngrok для колаба)

curl -X POST -H "Content-Type: application/json" \
-d "{\"prompt\": \"Привет, как дела?\", \"max_new_tokens\": 50}" \

{"error":"","generated_text":"\u041f\u0440\u0438\u0432\u0435\u0442, \u043a\u0430\u043a \u0434\u0435\u043b\u0430?\n\n\u0423 \u043c\u0435\u043d\u044f \u0442\u0443\u0442 \u0432\u043e\u0437\u043d\u0438\u043a\u043b\u0430 \u0437\u0430\u0434\u0430\u0447\u0430 - \u043d\u0443\u0436\u043d\u043e \u043d\u0430\u043f\u0438\u0441\u0430\u0442\u044c \u043f\u0440\u043e\u0433\u0440\u0430\u043c\u043c\u0443, \u043a\u043e\u0442\u043e\u0440\u0430\u044f \u0431\u0443\u0434\u0435\u0442 \u0430\u043d\u0430\u043b\u0438\u0437\u0438\u0440\u043e\u0432\u0430\u0442\u044c \u0442\u0435\u043a\u0441\u0442\u043e\u0432\u044b\u0439 \u0444\u0430\u0439\u043b \u0438 \u043f\u043e\u0434\u0441\u0447\u0438\u0442\u044b\u0432\u0430\u0442\u044c \u043a\u043e\u043b\u0438\u0447\u0435\u0441\u0442\u0432\u043e \u0441\u043b\u043e\u0432, \u0432 \u043a\u043e\u0442\u043e\u0440\u044b\u0445 \u0435\u0441\u0442\u044c \u0445\u043e\u0442\u044f \u0431\u044b \u043e\u0434\u043d\u0430 \u0437\u0430\u0433\u043b\u0430\u0432\u043d\u0430\u044f \u0431\u0443\u043a\u0432\u0430.\n\n**\u0412\u0445\u043e\u0434\u043d\u044b\u0435 \u0434\u0430\u043d\u043d\u044b\u0435:**\n*   \u0422\u0435\u043a\u0441\u0442\u043e"}
