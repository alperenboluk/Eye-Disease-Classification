# EyeAI – Deep Learning-Based Eye Disease Classification 👁️🧠

**EyeAI** is a deep learning-based image classification system developed as a **graduation project**. It is designed to assist in detecting and classifying eye-related diseases from medical images using modern image preprocessing and ensemble learning techniques.

## 🧠 Project Overview

This project combines image enhancement, deep learning model training, and backend deployment:

- A dataset of nearly **4,000 retinal images** was used for training.
  - Nearly 1,000 images per class: **Normal**, **Glaucoma**, **Cataract**, **Diabetic Retinopathy**
- Images were preprocessed using **CLAHE** and **bilateral filtering** with **OpenCV**.
- Three CNN models based on **EfficientNetB0** architecture were trained using **TensorFlow/Keras**.
- **Focal loss** was applied to address class imbalance.
- Models were trained in **Google Colab** and saved in `.keras` format.
- A **Flask REST API** serves real-time predictions using ensemble averaging.
- The API returns the predicted class, confidence score, and probabilities for all classes.
- **Grad-CAM** is used to visualize model attention areas and improve interpretability.

## 🚀 Technologies Used

- **Python**
- **TensorFlow / Keras**
- **OpenCV**
- **NumPy & Pandas**
- **Scikit-learn**
- **Flask**
- **Joblib**
- **Google Colab**
- **Git & GitHub**

## 🔬 Features

- Real-time classification via `/predict` endpoint
- Ensemble prediction from 3 trained models
- Automatic class probability calculation
- Grad-CAM visualization for model interpretation
- Dynamic loading of models and class labels

```
## 📁 Project Structure

├── app.py # Flask API backend
├── ensemble_model_1.keras # Trained model 1
├── ensemble_model_2.keras # Trained model 2
├── ensemble_model_3.keras # Trained model 3
├── ensemble_class_labels.pkl # Class label dictionary
├── site.html (optional) # Demo or test interface
└── Additional assets (poster, reports, images, etc.)



## 📸 Example API Response

```json
{
  "class": "cataract",
  "confidence": 0.93,
  "all_probs": {
    "cataract": 0.93,
    "normal": 0.05,
    "glaucoma": 0.02
  }
}


TTA Classification Report
                      precision    recall  f1-score   support

            cataract       0.91      0.94      0.92       156
diabetic_retinopathy       0.99      1.00      1.00       165
            glaucoma       0.82      0.82      0.82       151
              normal       0.89      0.86      0.87       161

            accuracy                           0.91       633
           macro avg       0.90      0.90      0.90       633
        weighted avg       0.90      0.91      0.90       633




📍 Project Status
✅ Completed as a graduation project
✅ Fully working ensemble-based classifier
✅ Ready for real-world medical AI integration (research use)

Note: This project was developed for educational and research purposes and should not be used in clinical environments without further validation.
