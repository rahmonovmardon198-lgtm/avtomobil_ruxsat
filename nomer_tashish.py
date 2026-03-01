from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode

# Biz ruhsat bergan nomerli avtolar
allowed_plates = [
    "01A123BC","10B777AA","20C456DF","30D890GH","40E321IJ",
    "50F654KL","60G987MN","70H147OP","80I258QR","90J369ST",
]

# Shlagbaum ochish uchun
class DummyArduino:
    def write(self, cmd):
        print(f"Arduino command: {cmd}")

arduino = DummyArduino()

# JS Kamera avtolarni nomerni aniqlash uchun (colab uchun ideal)
def take_photo(filename='photo.jpg', quality=0.8):
    js = Javascript('''
        async function takePhoto(quality) {
          const div = document.createElement('div');
          const capture = document.createElement('button');
          capture.textContent = '📸 Capture';
          div.appendChild(capture);
          
          const video = document.createElement('video');
          const stream = await navigator.mediaDevices.getUserMedia({video: true});
          document.body.appendChild(div);
          div.appendChild(video);
          video.srcObject = stream;
          await video.play();
          
          await new Promise((resolve) => capture.onclick = resolve);
          
          const canvas = document.createElement('canvas');
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          canvas.getContext('2d').drawImage(video, 0, 0);
          
          stream.getVideoTracks()[0].stop();
          div.remove();
          return canvas.toDataURL('image/jpeg', quality);
        }
    ''')
    display(js)
    data = eval_js('takePhoto({})'.format(quality))
    binary = b64decode(data.split(',')[1])
    with open(filename,'wb') as f:
        f.write(binary)
    return filename

# kamera qaytargan rasm kerakli qismni oladi
filename = take_photo()
img = cv2.imread(filename)

# Colab uchun tayyor model (HuggingFace)
!wget -O best.pt "https://huggingface.co/koushim/yolov8-license-plate-detection/resolve/main/best.pt"
model = YOLO("best.pt")

# tili ingliz alifbosida
reader = easyocr.Reader(['en'])


# olingan natijani tekshirish
results = model(img)

for r in results:
    for box in r.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        plate_img = img[y1:y2, x1:x2]

        # OCR
        ocr_result = reader.readtext(plate_img)
        for (_, text, prob) in ocr_result:
            plate = text.replace(" ","")
            print("Detected:", plate)

            if plate in allowed_plates:
                print("✅ RUXSAT BOR")
                color = (0,255,0)
                arduino.write(b'OPEN')
            else:
                print("❌ RUXSAT YO‘Q")
                color = (0,0,255)

            # Bounding box + text
            cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
            cv2.putText(img, plate, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

# natijani ko‘rsatish
cv2_imshow(img)

















