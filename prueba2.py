import cv2
import easyocr
import time
import threading

reader = easyocr.Reader(["en"], gpu=True)  #ponemos el idioma y el uso del gpu
video = cv2.VideoCapture(0, cv2.CAP_DSHOW) #activamos la camara

#tamaño de ventana
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#variables globales
conteo_textos = {}     
ocr_results = []
process_every = 10
frame_count = 0
lock = threading.Lock()
textos_activos = set()

def ocr_worker(img):
    global ocr_results
    results = reader.readtext(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)              
    blur = cv2.GaussianBlur(gray, (3, 3), 0)                 
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  
    results = reader.readtext(thresh, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.")
    with lock:
        ocr_results = results

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % process_every == 0 and threading.active_count() == 1:
        threading.Thread(target=ocr_worker, args=(frame.copy(),)).start()
    textos_en_frame = set()

    with lock:
        for res in ocr_results:
            text_detectado = res[1]
            textos_en_frame.add(text_detectado)

            pt0 = (int(res[0][0][0]), int(res[0][0][1]))
            pt2 = (int(res[0][2][0]), int(res[0][2][1]))
            cv2.rectangle(frame, pt0, pt2, (0, 255, 0), 2)
            mostrar_texto = f"{text_detectado} ({conteo_textos.get(text_detectado, 0)})"
            cv2.putText(frame, mostrar_texto, (pt0[0], pt0[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    for t in textos_en_frame:
        if t not in textos_activos:
            conteo_textos[t] = conteo_textos.get(t, 0) + 1

    textos_activos = textos_en_frame.copy()
    end = time.time()
    fps = 1 / max((end - time.time()), 1e-6)
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Detector", frame)
    if cv2.waitKey(1) & 0xFF == 27:  
        break

video.release()
cv2.destroyAllWindows()

print("\nConteo final de textos detectados:")
for texto, cantidad in conteo_textos.items():
    print(f"{texto}: {cantidad} veces")
