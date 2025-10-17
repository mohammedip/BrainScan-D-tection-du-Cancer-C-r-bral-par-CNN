# 🧠 Brain Tumor MRI Classifier

Ce projet est une application basée sur **Deep Learning** permettant de **classifier les tumeurs cérébrales à partir d’images IRM**.  
L’application utilise un modèle **CNN (Convolutional Neural Network)** entraîné avec TensorFlow/Keras et est déployée via **Streamlit** pour permettre une utilisation simple et interactive.

---

## 📋 Description du projet

L’objectif est de détecter automatiquement le type de tumeur cérébrale à partir d’une image d’IRM.  
Le modèle est capable de reconnaître quatre classes :

- 🧬 **Glioma Tumor**  
- 🧠 **Meningioma Tumor**  
- ⚙️ **Pituitary Tumor**  
- ✅ **No Tumor**

Le pipeline du projet comprend :
1. **Prétraitement des données** (chargement, redimensionnement, normalisation, augmentation des données)
2. **Entraînement d’un CNN** sur les données équilibrées
3. **Évaluation du modèle** (accuracy, matrice de confusion, rapport de classification)
4. **Déploiement via Streamlit**, où l’utilisateur peut charger une image IRM et obtenir une prédiction instantanée

---

## 🧩 Structure des fichiers


📁 brain_tumor_app/
│
├── Data/ # Dossier contenant les images des tumeurs (par classe)
│
├── model/
│ └── best_model.h5 # Modèle entraîné sauvegardé
│
├── preprocess.py # Chargement, traitement et équilibrage des données
├── train_model.py # Script d’entraînement du modèle CNN
│
├── Home.py # Interface principale Streamlit pour la prédiction
│
└── pages/
└── 1_Documentation.py # Page de documentation intégrée dans Streamlit

yaml
Copier le code

---

## ⚙️ Technologies utilisées

| Catégorie | Outils |
|------------|--------|
| Langage | Python |
| Framework Deep Learning | TensorFlow / Keras |
| Interface Web | Streamlit |
| Traitement d’images | OpenCV, PIL |
| Visualisation | Matplotlib, Seaborn |
| Manipulation de données | NumPy, Pandas, Scikit-learn |

---

## 🚀 Instructions pour exécuter le projet

### 1️⃣ Prérequis

Installez Python (>=3.8) puis installez les dépendances suivantes :

```bash
pip install streamlit tensorflow numpy opencv-python pillow matplotlib seaborn scikit-learn pandas
2️⃣ Organisation des fichiers
Assurez-vous que le modèle entraîné se trouve dans le dossier model/ :

bash
Copier le code
model/best_model.h5
et que le fichier Home.py se trouve à la racine du projet.

3️⃣ Lancer l’application Streamlit
Exécutez la commande suivante dans le terminal :

bash
Copier le code
streamlit run Home.py
Une fois lancé, ouvrez le lien affiché dans le terminal (généralement http://localhost:8501) dans votre navigateur.

4️⃣ Utilisation
Chargez une image IRM au format .jpg, .jpeg ou .png

Cliquez sur Classify Image

Le modèle affichera :

Le type de tumeur prédit

Le taux de confiance

Vous pouvez également consulter la page Documentation dans la barre latérale pour plus d’informations.

📊 Résultats du modèle
Architecture : CNN à 3 blocs convolutionnels + couches denses

Nombre d’époques : 35

Taille des images : 224×224

Optimiseur : Adam (lr = 0.001)

Accuracy élevée sur les données de test

Visualisations : courbes d’entraînement, matrice de confusion, exemples de prédictions correctes/incorrectes