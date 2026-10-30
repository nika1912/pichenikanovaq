import pyautogui
import cv2
import numpy as np
import onnxruntime as ort
import time
from pynput import keyboard

# ================= НАСТРОЙКИ =================
MODEL_PATH = 'bestV1.onnx'
IMG_SIZE = 640

# Пороги уверенности по классам (можно крутить отдельно)
CONF_GAME        = 0.15
CONF_UPGRADE     = 0.08     # мягкий порог — чтобы ловить даже слабые детекции
CONF_NOT_RECLAMA = 0.15

IOU_THRESHOLD = 0.4
COOKIE_CLICKS = 10
CLICK_DELAY = 0.05
UPGRADE_CLICK_DELAY = 0.5   # было 0.3 — увеличено
# =============================================

stop_bot = False
bot_running = False


def on_press(key):
    global stop_bot, bot_running
    try:
        if key == keyboard.Key.space:
            if bot_running:
                print("\n[!] ПРОБЕЛ — останавливаю бота...")
                stop_bot = True
            else:
                print("\n[!] ПРОБЕЛ — запускаю бота...")
                stop_bot = False
    except AttributeError:
        pass


listener = keyboard.Listener(on_press=on_press)
listener.start()

pyautogui.FAILSAFE = True

print("Загрузка ONNX модели...")
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# !!! Порядок и имена должны совпадать с data.yaml !!!
CLASS_NAMES = ['game', 'upgrade', 'not reclama']

# Проверка: выведите порядок классов, чтобы сверить с data.yaml
print("Порядок классов:", CLASS_NAMES)


def preprocess(image, img_size=IMG_SIZE):
    img = cv2.resize(image, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
    return img


def postprocess(outputs, orig_w, orig_h, img_size=IMG_SIZE):
    """
    Возвращает ВСЕ боксы по классам без фильтрации по порогу:
    {class_name: [ {conf, x, y}, ... ]}.
    Порог применяется позже, отдельно для каждого класса.
    """
    predictions = np.squeeze(outputs[0]).T
    found = {name: [] for name in CLASS_NAMES}
    boxes, scores, class_ids = [], [], []

    for pred in predictions:
        classes_scores = pred[4:]
        class_id = int(np.argmax(classes_scores))
        max_score = float(classes_scores[class_id])

        if max_score <= 0.01:   # отбрасываем совсем мусор
            continue

        x_center, y_center, w, h = pred[0], pred[1], pred[2], pred[3]

        x1 = int((x_center - w / 2) * orig_w / img_size)
        y1 = int((y_center - h / 2) * orig_h / img_size)
        x2 = int((x_center + w / 2) * orig_w / img_size)
        y2 = int((y_center + h / 2) * orig_h / img_size)

        x1 = max(0, min(x1, orig_w - 1))
        y1 = max(0, min(y1, orig_h - 1))
        x2 = max(0, min(x2, orig_w - 1))
        y2 = max(0, min(y2, orig_h - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(max_score)
        class_ids.append(class_id)

    if not boxes:
        return found

    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.01, IOU_THRESHOLD)
    if len(indices) == 0:
        return found

    indices = np.array(indices).flatten()

    for i in indices:
        x, y, w, h = boxes[i]
        cls_name = CLASS_NAMES[class_ids[i]]
        conf = scores[i]
        cx, cy = x + w / 2, y + h / 2
        found[cls_name].append({'conf': conf, 'x': cx, 'y': cy})

    for name in found:
        found[name].sort(key=lambda d: d['conf'], reverse=True)

    return found


def click(x, y):
    pyautogui.moveTo(int(x), int(y), duration=0)
    time.sleep(0.05)            # небольшая пауза после наведения
    pyautogui.click()
    time.sleep(0.05)            # пауза после клика


def find_and_click():
    global stop_bot
    if stop_bot:
        return False

    screenshot = pyautogui.screenshot()
    screenshot_np = np.array(screenshot)
    orig_h, orig_w = screenshot_np.shape[:2]
    screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)

    input_tensor= preprocess(screenshot_bgr)
    outputs = session.run([output_name], {input_name: input_tensor})
    found = postprocess(outputs, orig_w, orig_h)

    # Фильтрация по индивидуальным порогам + отладка
    upgrades = [u for u in found['upgrade']     if u['conf'] >= CONF_UPGRADE]
    reclamas = [r for r in found['not reclama'] if r['conf'] >= CONF_NOT_RECLAMA]
    games    = [g for g in found['game']        if g['conf'] >= CONF_GAME]

    # Отладка — показываем всё, что выше порога
    for u in upgrades:
        print(f"[Вижу] upgrade      | conf={u['conf']:.2f} | X={u['x']:.0f} Y={u['y']:.0f}")
    for r in reclamas:
        print(f"[Вижу] not reclama  | conf={r['conf']:.2f} | X={r['x']:.0f} Y={r['y']:.0f}")
    for g in games:
        print(f"[Вижу] game         | conf={g['conf']:.2f} | X={g['x']:.0f} Y={g['y']:.0f}")

    # --- 1. УЛУЧШЕНИЯ: покупаем ВСЕ найденные ---
    if upgrades:
        print(f">>> Найдено улучшений: {len(upgrades)}")
        for up in upgrades:
            if stop_bot:
                return False
            click(up['x'], up['y'])
            print(f"    >>> Клик по улучшению (conf={up['conf']:.2f})")
            time.sleep(UPGRADE_CLICK_DELAY)
        return True

    # --- 2. NOT RECLAMA ---
    if reclamas:
        best = reclamas[0]
        click(best['x'], best['y'])
        print(">>> Клик по NOT RECLAMA")
        time.sleep(0.5)
        return True

    # --- 3. ПЕЧЕНЬКА ---
    if games:
        best = games[0]
        pyautogui.moveTo(int(best['x']), int(best['y']), duration=0)
        for _ in range(COOKIE_CLICKS):
            if stop_bot:
                return False
            pyautogui.click()
            time.sleep(CLICK_DELAY)
        print(f">>> {COOKIE_CLICKS} кликов по ПЕЧЕНЬКЕ")
        return True

    return False


def bot_loop():
    while not stop_bot:
        try:
            if not find_and_click():
                time.sleep(0.3)
        except pyautogui.FailSafeException:
            print("\nОстановлено (мышь в углу).")
            return
        except Exception as e:
            print(f"Ошибка: {e}")
            time.sleep(1)


def main():
    global stop_bot, bot_running

    print("==========================================")
    print("ПРОБЕЛ — старт / стоп.  Ctrl+C — выход.")
    print("==========================================")
    print("Бот в режиме ожидания. Нажмите ПРОБЕЛ для запуска.")

    try:
        while True:
            if not stop_bot and not bot_running:
                bot_running = True
                print("\n>>> Бот ЗАПУЩЕН")
                bot_loop()
                bot_running = False
                print("\n>>> Бот ОСТАНОВЛЕН. ПРОБЕЛ — запустить снова.")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nВыход (Ctrl+C).")
    finally:
        listener.stop()
        print("Программа завершена.")


if __name__ == "__main__":
    main()