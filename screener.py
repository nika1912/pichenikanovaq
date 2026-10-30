import pyautogui
import time 
from datetime import datetime
import os

save_dir = "screensshots"
os.makedirs(save_dir, exist_ok=True)

print("Начинаю делать скриншоты каждые 5 секунд...")
print("Нажмите Ctrl+C для остановки\n")


count = 0

try:
    while True:
        screenshot = pyautogui.screenshot()

        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        filename = f"{save_dir}/screenshot_{timestamp}.png"

        screenshot.save(filename)
        count += 1
        print(f"[{count}] Сохранено: {filename}")

        time.sleep(5)


except KeyboardInterrupt:
    print(f"\nОстановлено. Всего сделано скриншотов: {count}")