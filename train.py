# =========================================================
# Projet : Détection des émotions avec MobileNet
# Auteur : (Ton Nom)
# Description : Entraînement d’un modèle basé sur MobileNet
#               pour classifier les émotions du visage
# =========================================================

# --- Importation des librairies nécessaires ---
from keras.applications import MobileNet
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ---------------------------------------------------------
# 1. Paramètres de base du modèle
# ---------------------------------------------------------

# Dimensions des images (MobileNet fonctionne avec 224x224 pixels en entrée)
img_rows, img_cols = 224, 224

# Nombre de classes (émotions à reconnaître)
num_classes = 5   # Exemple : Angry, Happy, Neutral, Sad, Surprise

# Chemins vers les datasets (à adapter selon ton PC)
train_data_dir = "fer2013/train"
validation_data_dir = "fer2013/validation"

# ---------------------------------------------------------
# 2. Chargement du modèle MobileNet pré-entraîné (ImageNet)
# ---------------------------------------------------------

# On charge MobileNet sans les couches finales (include_top=False)
base_model = MobileNet(weights="imagenet",
                       include_top=False,
                       input_shape=(img_rows, img_cols, 3))

# Option : rendre toutes les couches entraînables
for layer in base_model.layers:
    layer.trainable = True

# ---------------------------------------------------------
# 3. Ajout de nouvelles couches (classification des émotions)
# ---------------------------------------------------------

def ajouter_couches_personnalisees(base, nb_classes):
    """
    Ajoute les couches supérieures au modèle de base.
    Ici on crée un "head" pour classifier les émotions.
    """
    x = base.output
    x = GlobalAveragePooling2D()(x)       # Réduction des dimensions
    x = Dense(1024, activation="relu")(x) # Couche dense 1
    x = Dense(1024, activation="relu")(x) # Couche dense 2
    x = Dense(512, activation="relu")(x)  # Couche dense 3
    predictions = Dense(nb_classes, activation="softmax")(x) # Sortie
    return predictions

# Création du modèle final
FC_Head = ajouter_couches_personnalisees(base_model, num_classes)
model = Model(inputs=base_model.input, outputs=FC_Head)

# Affichage de l’architecture du modèle
print(model.summary())

# ---------------------------------------------------------
# 4. Préparation des données (Data Augmentation)
# ---------------------------------------------------------

# Générateur pour l’entraînement (avec augmentation des données)
train_datagen = ImageDataGenerator(
    rescale=1./255,           # Normalisation
    rotation_range=30,        # Rotation aléatoire
    width_shift_range=0.3,    # Décalage horizontal
    height_shift_range=0.3,   # Décalage vertical
    horizontal_flip=True,     # Retournement horizontal
    fill_mode="nearest"       # Remplissage des pixels manquants
)

# Générateur pour la validation (juste normalisation)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Création des flux d’images
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_rows, img_cols),
    batch_size=32,
    class_mode="categorical"
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_rows, img_cols),
    batch_size=32,
    class_mode="categorical"
)

# ---------------------------------------------------------
# 5. Callbacks (sauvegarde, arrêt anticipé, ajustement du LR)
# ---------------------------------------------------------

checkpoint = ModelCheckpoint(
    "emotion_face_mobilNet.h5",  # Sauvegarde du meilleur modèle
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    verbose=1
)

earlystop = EarlyStopping(
    monitor="val_loss",
    patience=10,                # Arrêt si pas d’amélioration
    restore_best_weights=True,
    verbose=1
)

learning_rate_reduction = ReduceLROnPlateau(
    monitor="val_accuracy",      # Réduction du learning rate si stagnation
    patience=5,
    factor=0.2,
    min_lr=0.0001,
    verbose=1
)

callbacks = [earlystop, checkpoint, learning_rate_reduction]

# ---------------------------------------------------------
# 6. Compilation du modèle
# ---------------------------------------------------------

model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(lr=0.001),
    metrics=["accuracy"]
)

# ---------------------------------------------------------
# 7. Entraînement du modèle
# ---------------------------------------------------------

nb_train_samples = 24176       # Nombre total d’images d’entraînement
nb_validation_samples = 3006   # Nombre total d’images de validation
epochs = 25                    # Nombre d’époques

history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // 32,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // 32
)
