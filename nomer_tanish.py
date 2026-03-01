import cv2
from ultralytics import YOLO
import easyocr
import serial
import numpy as np

# 20 ta ruxsatli nomer array
allowed_plates = np.array([
    "01A123BC","10B777AA","20C456DF","30D890GH","40E321IJ",
    "50F654KL","60G987MN","70H147OP","80I258QR","90J369ST",
    "01K741UV","10L852WX","20M963YZ","30N159AB","40O753CD",
    "50P486EF","60Q951GH","70R357IJ","80S642KL","90T804MN"
])

# YOLO model yuklash
model = YOLO("yolov8n.pt")  # internetdan yuklaydi test uchun
model.predict(source="https://ultralytics.com/images/bus.jpg", show=True)

# OCR
reader = easyocr.Reader(['en'])

# Arduino ulanish (shlagbaum uchun) manda bomaganu uchun shu ko'rinishda cmd
class DummyArduino:
    def write(self, cmd):
        print(f"Arduino command: {cmd}")

arduino = DummyArduino()

# Kamera ochish
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO bilan nomer aniqlash
    results = model(frame)

    for r in results:
        boxes = r.boxes.xyxy

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            # Bounding box chizish
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            # Nomer qismini crop qilish
            plate_img = frame[y1:y2, x1:x2]

            # OCR
            ocr_result = reader.readtext(plate_img)

            for (_, text, prob) in ocr_result:
                plate = text.replace(" ", "")

                # Tekshirish
                if plate in allowed_plates:
                    print("Ruxsat bor:", plate)
                    # Shlagbaumni ochish
                    arduino.write(b'OPEN')  
                    cv2.putText(frame, f"{plate} - OPEN", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                else:
                    print("Ruxsat yo‘q:", plate)
                    cv2.putText(frame, f"{plate} - BLOCKED", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    # Ekranda ko‘rsatish
    cv2.imshow("License Plate Detection", frame)

    # ESC bosilsa chiqadi
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

















