# 🧠 doc.ai: MRI Diagnostic Portal

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app/) <!-- Replace with your actual Streamlit URL -->
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://tensorflow.org)

## 📌 Project Overview
**doc.ai** is an end-to-end Machine Learning web application designed to assist medical professionals in diagnosing brain tumors from Magnetic Resonance Imaging (MRI) scans. Leveraging Deep Learning and Computer Vision, the system automatically classifies MRI scans into four distinct categories with high confidence, providing a rapid secondary validation tool to reduce diagnostic bottlenecks.

**Diagnostic Categories:**
* Glioma
* Meningioma
* Pituitary Tumor
* No Tumor (Healthy)

## ⚙️ Tech Stack
* **Deep Learning Framework:** TensorFlow / Keras
* **Web Framework:** Streamlit (Python)
* **Data Processing & Vision:** NumPy, Pillow (PIL)
* **Cloud Integration:** Google Drive API (via `gdown` for handling large `.keras` model weights)

## 🏗️ Model Architecture & Pipeline
The core model is a Convolutional Neural Network (CNN) built to extract hierarchical spatial features from complex neurological scans.
1. **Preprocessing:** All uploaded MRI scans are dynamically resized to a uniform `224x224` resolution and converted to numpy arrays for tensor ingestion.
2. **Feature Extraction:** Utilizes sequential 2D Convolutional blocks paired with Max-Pooling layers.
3. **Regularization:** Dropout layers are implemented to prevent model overfitting on localized imaging artifacts.
4. **Classification:** A dense Softmax output layer generates a precise probability distribution across the four diagnostic classes.

## 🛡️ Clinical Safety Protocol (Guardrails)
To ensure ethical AI deployment in a healthcare context, this application features a deterministic **Uncertainty Threshold**. 
* If the neural network's maximum confidence score falls below **80.0%**, the system intentionally suppresses a definitive diagnostic output.
* Instead, it triggers a visual warning alert advising the user that the AI is uncertain and that the scan requires manual review by a certified radiologist.

## 🚀 How to Run Locally

If you want to run this application on your local machine rather than the cloud server, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/doc-ai-tumor-scanner.git](https://github.com/yourusername/doc-ai-tumor-scanner.git)
   cd doc-ai-tumor-scanner
