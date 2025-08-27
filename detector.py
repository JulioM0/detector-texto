import cv2
import easyocr
import time

reader = easyocr.Reader(["en"], gpu=True)
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
start = time.time()
while True:
    ret, frame = video.read()
    if ret == False : break

    result = reader.readtext(frame)
    for res in result:
        texto = res
        pt0 = int(res[0][0][0]), int(res[0][0][1])
        pt2 = int(res[0][2][0]), int(res[0][2][1])
        cv2.rectangle(frame, pt0, pt2, (0, 255, 0), 2)
        cv2.putText(frame, res[1], (10, 40), 1, 2.4, (166,56,242), 4)

    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 : break
end = time.time()
fps = 1 / (end - start)
video.release()
cv2.destroyAllWindows()