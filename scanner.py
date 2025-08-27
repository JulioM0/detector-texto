import cv2
import easyocr
import threading

reader = easyocr.Reader(['en', 'es'], gpu=True)

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

text_detected = set()
frame_to_process = None
ocr_results = []
lock = threading.Lock()

def ocr_thread():
    global frame_to_process, ocr_results
    while True:
        if frame_to_process is not None:
            lock.acquire()
            frame = frame_to_process.copy()
            frame_to_process = None
            lock.release()

            results = reader.readtext(frame)
            lock.acquire()
            ocr_results = results
            lock.release()

            for (_, text, _) in results:
                if text not in text_detected:
                    print(f"Texto detectado: {text}")
                    text_detected.add(text)

threading.Thread(target=ocr_thread, daemon=True).start()

while True:
    ret, frame = video.read()
    if not ret:
        break

    if frame_to_process is None:
        lock.acquire()
        frame_to_process = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        lock.release()
    lock.acquire()
    for (bbox, text, _) in ocr_results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(int(i*2) for i in top_left)
        bottom_right = tuple(int(i*2) for i in bottom_right)
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(frame, text, (top_left[0], top_left[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    lock.release()

    cv2.imshow("Detector de Texto", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
