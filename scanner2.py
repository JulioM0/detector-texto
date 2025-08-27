import cv2
import easyocr
import threading
import tkinter as tk
from tkinter import scrolledtext

# configuracion de la camara y medidas
reader = easyocr.Reader(['es'], gpu=False, verbose=False)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# variables globales 
results = []
frame_to_process = None
lock = threading.Lock()
conteo_textos = {}
textos_en_pantalla = set()

# funcion ocr 
def ocr_worker():
    global frame_to_process, results
    while True:
        if frame_to_process is not None:
            with lock:
                frame_copy = frame_to_process.copy()
            gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)  
            gray = cv2.GaussianBlur(gray, (3,3), 0)  
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            results = reader.readtext(thresh, detail=1, paragraph=True, text_threshold=0.6,  low_text=0.4,  link_threshold=0.5, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            frame_to_process = None

threading.Thread(target=ocr_worker, daemon=True).start()

#funcion que actualiza camara y consola
def update_frame():
    global frame_to_process, resultados_texto, textos_en_pantalla
    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    if frame_to_process is None:
        with lock:
            frame_to_process = frame.copy()
    textos_actuales = set()

    for item in results:
        if len(item) == 3:
            bbox, text, prob = item
        elif len(item) == 2:
            bbox, text = item
            prob = 1.0 
        if prob < 0.85:
            continue
        textos_actuales.add(text)
        if text not in textos_en_pantalla:
            conteo_textos[text] = conteo_textos.get(text, 0) + 1
            console_text.insert(tk.END, f"{text}: {conteo_textos[text]}\n")

    # Dibujar en pantalla
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(frame, text, (top_left[0], top_left[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    textos_en_pantalla = textos_actuales

    cv2.imshow("Detector de texto", frame)

    if cv2.waitKey(1) == 27:  #ESC para salir
        cap.release()
        cv2.destroyAllWindows()
        root.destroy()
        return

    root.after(10, update_frame)

root = tk.Tk()
root.title("Resultados de la lectura")

console_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=20)
console_text.pack(padx=10, pady=10)

update_frame()
root.mainloop()
