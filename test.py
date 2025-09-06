# ==============================
# EMOTION DETECTOR
# ==============================
# USAGE : python test.py
# Description :
#   - Capture en temps réel via webcam
#   - Détection de visage avec Haarcascade
#   - Prédiction de l'émotion avec un modèle Keras entraîné
#   - Affichage avec rectangle + texte de l'émotion
# ==============================

# ---- Importation des librairies ----
import cv2                         # Pour la vision par ordinateur (OpenCV)
import numpy as np                 # Pour le traitement des tableaux (images)
from tensorflow.keras.models import load_model    # Pour charger le modèle Keras
from tensorflow.keras.utils import img_to_array   # Pour convertir image -> array

# ---- Chargement du classifieur de visages ----
# Le fichier Haarcascade est un modèle pré-entraîné d'OpenCV pour détecter les visages
face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

# ---- Chargement du modèle d'émotion entraîné ----
classifier = load_model('./Emotion_Detection.h5')

# ---- Labels des émotions correspondants aux sorties du modèle ----
class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# ---- Définition des couleurs pour chaque émotion (format BGR pour OpenCV) ----
colors = {
    'Angry': (0, 0, 255),      # Rouge
    'Happy': (0, 255, 0),      # Vert
    'Neutral': (255, 0, 0),    # Bleu
    'Sad': (128, 0, 128),      # Violet
    'Surprise': (0, 255, 255)  # Jaune
}

# ---- Initialisation de la webcam ----
# 0 = caméra par défaut de l’ordinateur
cap = cv2.VideoCapture(0)

# ---- Boucle principale pour traitement en continu ----
while True:
    # Lire une image de la webcam
    ret, frame = cap.read()

    # Conversion en niveaux de gris (nécessaire pour la détection de visages)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détection des visages dans l'image
    # Paramètres : (image, facteur d’échelle, nb min de voisins)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    # Parcours de tous les visages détectés
    for (x, y, w, h) in faces:
        # Extraire la région du visage (ROI = Region Of Interest)
        roi_gray = gray[y:y + h, x:x + w]

        # Redimensionner à 48x48 (taille d'entrée attendue par le modèle)
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # Vérifier que la ROI n'est pas vide
        if np.sum([roi_gray]) != 0:
            # Normalisation entre 0 et 1
            roi = roi_gray.astype('float') / 255.0

            # Conversion en array compatible Keras
            roi = img_to_array(roi)

            # Ajout d’une dimension batch : (1, 48, 48, 1)
            roi = np.expand_dims(roi, axis=0)

            # ---- Prédiction de l'émotion ----
            preds = classifier.predict(roi, verbose=0)[0]
            label = class_labels[preds.argmax()]  # Émotion la plus probable

            # Récupérer la couleur correspondant à l'émotion
            color = colors[label]

            # Dessiner un rectangle autour du visage
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Ajouter le texte (nom de l'émotion)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        else:
            # Si aucun visage n'est trouvé
            cv2.putText(frame, 'No Face Found', (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # ---- Affichage du flux vidéo avec annotations ----
    cv2.imshow('Emotion Detector', frame)

    # ---- Sortie avec la touche "q" ----
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---- Libérer les ressources ----
cap.release()
cv2.destroyAllWindows()
