# Импортируем необходимые библиотеки
!pip install transformers sentence-transformers  # Выполняется в случае, если библиотеки не установлены

from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# Определение текстов для анализа
text1 = "Введите текст первого поста здесь."
text2 = "Введите текст второго поста здесь."

# 1. Zero-Shot Classification
print("1. Метод Zero-Shot Classification")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Формируем гипотезу
hypothesis = "Эти два поста гармонируют."

# Прогоняем через Zero-Shot классификатор
result = classifier(f"{text1} {text2}", candidate_labels=["истина", "ложь"])
zero_shot_label = 1 if result["labels"][0] == "истина" else 0
print(f"Zero-Shot Prediction: {'Гармонируют' if zero_shot_label else 'Не гармонируют'}")
print("Детали:", result)
print("\n" + "="*50 + "\n")

# 2. Семантическое Сходство с Sentence-BERT
print("2. Метод Семантического Сходства")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Преобразование текстов в эмбеддинги
embedding1 = model.encode(text1, convert_to_tensor=True)
embedding2 = model.encode(text2, convert_to_tensor=True)

# Вычисление косинусного сходства
similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
similarity_label = 1 if similarity > 0.8 else 0
print(f"Cosine Similarity Prediction: {'Гармонируют' if similarity_label else 'Не гармонируют'}")
print(f"Сходство: {similarity:.2f}")
print("\n" + "="*50 + "\n")

# Выводим общий результат
print("Итог:")
print(f"Zero-Shot Prediction: {'Гармонируют' if zero_shot_label else 'Не гармонируют'}")
print(f"Cosine Similarity Prediction: {'Гармонируют' if similarity_label else 'Не гармонируют'}")
