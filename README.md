# 🎭 Détection des Émotions en Temps Réel avec Python & OpenCV  

## 📌 Description
Ce projet est une application de **détection des émotions faciales en temps réel** utilisant la **webcam**, **OpenCV**, et un modèle pré-entraîné en Deep Learning (`Emotion_Detection.h5`).  
L'application identifie automatiquement plusieurs émotions parmi :  
😠 Angry | 😀 Happy | 😐 Neutral | 😢 Sad | 😲 Surprise  

Les visages sont encadrés avec des **couleurs différentes selon l’émotion détectée**.  

## ⚙️ Fonctionnalités
✔️ Détection faciale avec **Haar Cascade**  
✔️ Prédiction des émotions avec un modèle entraîné (`Emotion_Detection.h5`)  
✔️ Support **multi-personnes** dans la même image  
✔️ Couleurs différentes pour chaque émotion (ex: rouge = Angry, vert = Happy, jaune = Surprise, etc.)  
✔️ Exécution en **temps réel via webcam**  

## 📂 Structure du projet
emotion-detector/
│── haarcascade_frontalface_default.xml # Classifieur de visages
│── Emotion_Detection.h5 # Modèle entraîné
│── test.py # Script principal
│── train.py (optionnel) # Script d’entraînement (si tu veux re-trainer)
│── requirements.txt # Dépendances Python

## 🐍 Installation de Python 3.10

⚠️ Ce projet fonctionne uniquement avec **Python 3.10.x**. TensorFlow 2.10+ peut ne pas être compatible avec Python 3.11 ou supérieur.

### 1️⃣ Télécharger Python 3.10
- Rendez-vous sur la page officielle : [Python 3.10 Downloads](https://www.python.org/downloads/release/python-31011/)  
- Téléchargez la version correspondant à votre système (Windows / Linux / Mac).

### 2️⃣ Installer Python 3.10
- Pendant l’installation, cochez **"Add Python to PATH"** (très important).  
- Vérifiez l’installation avec :
python --version
Cela doit afficher quelque chose comme :
Python 3.10.11

### 3️⃣ Créer un environnement virtuel (recommandé)
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

## 🚀 Installation & Utilisation du projet

## 1️⃣ Cloner le dépôt :
git clone https://github.com/ton-nom-utilisateur/emotion-detector.git
cd emotion-detector

## 2️⃣ Installer les dépendances :
pip install -r requirements.txt

## 3️⃣ Lancer l’application :
python test.py

## 4️⃣ Quitter : Appuyez sur Q pour fermer la fenêtre.

## 📘 Explications techniques
Détection des visages : avec haarcascade_frontalface_default.xml (classifieur OpenCV).
Prétraitement : les visages sont redimensionnés en 48x48 et normalisés avant passage au modèle.
Prédiction : le modèle Emotion_Detection.h5 prédit l’émotion la plus probable.
Affichage : le visage est entouré d’un rectangle coloré selon l’émotion, avec le label affiché au-dessus.

## 🔮 Améliorations possibles
Ajouter une interface graphique (Tkinter ou PyQt).
Enregistrer les résultats (vidéo avec annotations).
Héberger le modèle sur une application web (Streamlit / Flask).
Améliorer le dataset avec un nouvel entraînement.
