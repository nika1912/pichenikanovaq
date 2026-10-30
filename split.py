import os
import shutil
import random

# --- НАСТРОЙКИ ---
SOURCE_DIR = "itgo"
TARGET_DIR = "."
SPLIT_RATIO = 0.8       # 80% train, 20% val

# Создаем структуру папок
for folder in ["images/train", "images/val", "labels/train", "labels/val"]:
    os.makedirs(os.path.join(TARGET_DIR, folder), exist_ok=True)

# Ищем все картинки
files = os.listdir(SOURCE_DIR)
images = [f for f in files if f.lower().endswith(('.jpeg', '.jpg', '.png'))]

# Перемешиваем
random.shuffle(images)

# Считаем, сколько пойдет в train
train_count = int(len(images) * SPLIT_RATIO)
train_images = images[:train_count]
val_images = images[train_count:]

print("Раскидываю по папкам...")

# --- ОБРАБОТКА TRAIN ---
for img_name in train_images:
    # 1. Копируем картинку в images/train
    src_img = os.path.join(SOURCE_DIR, img_name)
    dst_img = os.path.join(TARGET_DIR, "images", "train", img_name)
    shutil.copy(src_img, dst_img)

    # 2. Ищем такой же файл, но с .txt, и копируем в labels/train
    txt_name = os.path.splitext(img_name)[0] + ".txt"
    src_txt = os.path.join(SOURCE_DIR, txt_name)
    
    # Проверяем, существует ли txt файл, прежде чем копировать
    if os.path.exists(src_txt):
        dst_txt = os.path.join(TARGET_DIR, "labels", "train", txt_name)
        shutil.copy(src_txt, dst_txt)
    else:
        print(f"Внимание: для {img_name} нет файла разметки {txt_name}")

# --- ОБРАБОТКА VAL ---
# (Добавляем эту часть, так как на скриншоте код обрывается)
for img_name in val_images:
    # 1. Копируем картинку в images/val
    src_img = os.path.join(SOURCE_DIR, img_name)
    dst_img = os.path.join(TARGET_DIR, "images", "val", img_name)
    shutil.copy(src_img, dst_img)

    # 2. Ищем такой же файл, но с .txt, и копируем в labels/val
    txt_name = os.path.splitext(img_name)[0] + ".txt"
    src_txt = os.path.join(SOURCE_DIR, txt_name)
    
    if os.path.exists(src_txt):
        dst_txt = os.path.join(TARGET_DIR, "labels", "val", txt_name)
        shutil.copy(src_txt, dst_txt)

print("Готово! Обработано изображений:", len(images))
print(f"Train: {len(train_images)} | Val: {len(val_images)}")