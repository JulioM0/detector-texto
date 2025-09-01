import cv2
import easyocr
import threading
import tkinter as tk
from tkinter import scrolledtext
import time
import numpy as np

reader = easyocr.Reader(['es'], gpu=False, verbose=False)
dummy = np.zeros((100, 100, 3), dtype=np.uint8)
reader.readtext(dummy, detail=0)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 540)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

results = []
frame_to_process = None
lock = threading.Lock()
conteo_textos = {}
textos_en_pantalla = set()
last_read_time = 0
scan_interval = 0.5  # segundos entre lecturas

# funcion ocr optimizada
def ocr_worker():
    global frame_to_process, results, last_read_time
    while True:
        if frame_to_process is not None and (time.time() - last_read_time) > scan_interval:
            with lock:
                frame_copy = frame_to_process.copy()

            frame_small = cv2.resize(frame_copy, (320, 240))
            gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # OCR
            results = reader.readtext(
                thresh,
                detail=1,
                paragraph=False,
                text_threshold=0.6,
                low_text=0.4,
                link_threshold=0.5,
                allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz áéíóúüñÁÉÍÓÚÜÑ.,!?-()[]{}:;'
            )
            last_read_time = time.time()
            frame_to_process = None

threading.Thread(target=ocr_worker, daemon=True).start()

#funcion que actualiza camara y consola
def update_frame():
    global frame_to_process, textos_en_pantalla
    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    if frame_to_process is None:
        with lock:
            frame_to_process = frame.copy()

    textos_actuales = set()

    for item in results:
        if len(item) >= 2:
            bbox, text = item[0], item[1]
            prob = item[2] if len(item) == 3 else 1.0
        else:
            continue

        if prob < 0.8:
            continue

        textos_actuales.add(text)

        if text not in textos_en_pantalla:
            conteo_textos[text] = conteo_textos.get(text, 0) + 1
            console_text.insert(tk.END, f"Valores detectados: {text} - Conteo: {conteo_textos[text]}\n")

        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(frame, text, (top_left[0], top_left[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    textos_en_pantalla = textos_actuales

    cv2.imshow("Detector de texto", frame)

    if cv2.waitKey(1) == 27:  # ESC para salir
        cap.release()
        cv2.destroyAllWindows()
        root.destroy()
        return

    root.after(10, update_frame)

root = tk.Tk()
root.title("Resultados de la lectura")
root.geometry("600x400")

button_frame = tk.Frame(root)
button_frame.pack(pady=5)

#boton de limpiar consola
def clear_console():
    console_text.delete(1.0, tk.END)

clear_button = tk.Button(button_frame, text="Limpiar Consola", command=clear_console)
clear_button.pack(side=tk.LEFT, padx=5)

console_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=20)
console_text.pack(padx=10, pady=10)

update_frame()
root.mainloop()