# Установка необходимых пакетов
!pip install sentence-transformers torch

from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader

# 1. Подготовка данных для Fine-Tuning
# Пример данных - тройки (anchor, positive, negative)
train_examples = [
    InputExample(texts=["Текст поста 1", "Гармонирующий текст поста 1", "Не гармонирующий текст поста 1"]),
    InputExample(texts=["Текст поста 2", "Гармонирующий текст поста 2", "Не гармонирующий текст поста 2"]),
    # Добавьте больше примеров из тренировочного набора
]

# Создание DataLoader для загрузки данных
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Используем Triplet Loss для fine-tuning
model = SentenceTransformer('all-MiniLM-L6-v2')
train_loss = losses.TripletLoss(model=model, distance_metric=losses.TripletDistanceMetric.COSINE, margin=0.3)

# 2. Fine-Tuning модели
num_epochs = 4
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=int(len(train_dataloader) * num_epochs * 0.1),
    show_progress_bar=True
)

# 3. Сохранение дообученной модели
model.save("fine-tuned-triplet-model")

# 4. Использование дообученной модели для предсказания
# Загрузка модели и проверка косинусного расстояния между текстами
model = SentenceTransformer('fine-tuned-triplet-model')

# Пример предсказания: текстовые эмбеддинги и косинусное сходство
embedding1 = model.encode("Текст поста 1", convert_to_tensor=True)
embedding2 = model.encode("Гармонирующий текст поста 1", convert_to_tensor=True)
embedding3 = model.encode("Не гармонирующий текст поста 1", convert_to_tensor=True)

# Вычисление косинусного сходства
similarity_positive = util.pytorch_cos_sim(embedding1, embedding2).item()
similarity_negative = util.pytorch_cos_sim(embedding1, embedding3).item()

print(f"Cosine Similarity (Гармонирующий): {similarity_positive:.2f}")
print(f"Cosine Similarity (Не гармонирующий): {similarity_negative:.2f}")
