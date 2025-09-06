# USAGE : python test.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array

# Chargement du classifieur de visages (Haarcascade)
face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

# Chargement du modèle entraîné
classifier = load_model('./Emotion_Detection.h5')

# Les labels utilisés pour les prédictions
class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Dictionnaire des couleurs par émotion
colors = {
    'Angry': (0, 0, 255),      # Rouge
    'Happy': (0, 255, 0),      # Vert
    'Neutral': (255, 0, 0),    # Bleu
    'Sad': (128, 0, 128),      # Violet
    'Surprise': (0, 255, 255)  # Jaune
}

# Démarrage de la webcam (0 = caméra par défaut)
cap = cv2.VideoCapture(0)

while True:
    # Lecture d’une frame depuis la webcam
    ret, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Prédiction sur la ROI
            preds = classifier.predict(roi, verbose=0)[0]
            label = class_labels[preds.argmax()]

            # Choisir la couleur associée à l’émotion
            color = colors[label]

            # Dessiner le rectangle avec la bonne couleur
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Ajouter le texte de l’émotion
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        else:
            cv2.putText(frame, 'No Face Found', (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Affichage du résultat
    cv2.imshow('Emotion Detector', frame)

    # Quitter avec la touche Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
