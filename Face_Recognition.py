import cv2
import time
class VideoCamera(object):
def __init__(self):
# Using OpenCV to capture from device 0. If you have trouble capturing
# from a webcam, comment the line below out and use a video file
# instead.
self.video = cv2.VideoCapture(0)
# If you decide to use video.mp4, you must have this file in the folder
# as the main.py.
# self.video = cv2.VideoCapture('output.mp4')

def __del__(self):
self.video.release()

def get_frame(self):
success, image = self.video.read()
ret, jpeg = cv2.imencode('.jpg', image)
# We are using Motion JPEG, but OpenCV defaults to capture raw images,
# so we must encode it into JPEG in order to correctly display the
# video stream.
faceCascade = cv2.CascadeClassifier('C:\\Users\\HADES\\Documents\\OCV\\opencv-4.0.1\\data\\haarcascades\\haarcascade_frontalface_alt.xml')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30),flags = cv2.CASCADE_SCALE_IMAGE)
if len(faces)>0:
# for (x1, y1, x2, y2) in faces:
# face = cv2.resize(image[y1:y2,x1:x2],(150,150),interpolation = cv2.INTER_AREA)
cv2.imwrite('images/image'+str(time.time())+'.jpg',image)
return jpeg.tobytes()
from flask import Flask, render_template, Response
from camera import VideoCamera

app = Flask(__name__)

@app.route('/')
def index():
return render_template('index.html')

def gen(camera):
while True:
frame = camera.get_frame()
yield (b'--frame\r\n'
b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
return Response(gen(VideoCamera()),
mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
app.run(host='0.0.0.0', debug=True)

#importation des bibliotheques
import face_recognition

import pickle

#   D’abord en doit collecter les images
Anouar_image = face_recognition.load_image_file("teachers/Anouar_Ben_Maasoud.png")
#   Faire un modèle, nom pour chaque visage
Anouar_face_encoding = face_recognition.face_encodings(Dhafer_image)[0]

known_face_encodings = [
Anouar_face_encoding
]
known_face_names = [
"Mr Anouar Ben Maasoud"
]

# Sauvgrade des objets:
with open('faces.pkl', 'wb') as f:
pickle.dump([Anouar_face_encoding, known_face_encodings, known_face_names], f)

# importation des biblotheques
import face_recognition
from gpiozero import LED
from time import sleep
import cv2
import numpy as np
import pickle
import yagmail
#preparation de email
yag = yagmail.SMTP("willsmithplaysfootball@gmail.com","2019ocvmagic")
contents = ['Hello ',
'You can find a file attached.', name+'image.jpg']
#source de caputre de video
video_capture = cv2.VideoCapture(0)
name = ''
# importation des objets:
with open('faces.pkl', 'rb') as f:
Anouar_face_encoding, Souhail_face_encoding, known_face_encodings, known_face_names= pickle.load(f)
# Initialisation des variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
while True:
# prendre une seule trame de video
ret, frame = video_capture.read()
# Redimensionnement de trame du video a 1/4 pour une procession plus rapide
small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
# Convertion d'image de BGR color a RGB
rgb_small_frame = small_frame[:, :, ::-1]
# processus de comparaison
if process_this_frame:
# recherchre des vecteurs de model dans video
face_locations = face_recognition.face_locations(rgb_small_frame)
face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
face_names = []
for face_encoding in face_encodings:
# si l'image ne coreespond a persoone du base
matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
name = "Unknown"
# Si l'image corespond a un model deja connus
face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
best_match_index = np.argmin(face_distances)
if matches[best_match_index]:
name = known_face_names[best_match_index]
face_names.append(name)
process_this_frame = not process_this_frame
# afficher les resultas
for (top, right, bottom, left), name in zip(face_locations, face_names):
# retablir la taille original de trame
top *= 4
right *= 4
bottom *= 4
left *= 4
if name !== "Unknown":
led.on()
sleep(2)
led.off()
# dessiner un cadre sur le visage
cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
# dessiner un cadre contenant le nom sur le visage
cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
font = cv2.FONT_HERSHEY_DUPLEX
cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
# afficher l'image
cv2.imshow('Video', frame)
# appuyer sur 'q' pour quitter!
if cv2.waitKey(1) & 0xFF == ord('q'):
break
# lachement de webcam
video_capture.release()
cv2.destroyAllWindows()
