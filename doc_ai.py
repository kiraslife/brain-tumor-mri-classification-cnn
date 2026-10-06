import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os
# 1. UI/UX Clinical Design
st.markdown("""
<style>
    .stApp { background-color: #F8FAFC; color: #2C3E50; }
    h1, h2, h3 { color: #0056D2 !important; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-weight: 600; }
    .stFileUploader label { color: #2C3E50 !important; font-weight: 500; font-size: 16px; }
    .prediction-box { background-color: #EBF5FF; border-left: 5px solid #0056D2; padding: 15px; border-radius: 5px; margin-top: 20px; }
    .warning-box { background-color: #FEF3C7; border-left: 5px solid #F59E0B; padding: 15px; border-radius: 5px; margin-top: 20px; }
</style>
""", unsafe_allow_html=True)

# 2. Main UI
st.title("🩺 doc.ai: MRI Diagnostic Portal")
st.markdown("Upload a patient's brain MRI scan for a rapid, AI-assisted preliminary diagnosis.")



@st.cache_resource
def load_model():
    model_path = 'best_brain_tumor_model.keras'
    
    # Download from Drive if it doesn't exist on the Streamlit server
    if not os.path.exists(model_path):
        file_id = '1ETe6l5ugyMqFGAZwga8tUsgXvgymvzWA'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, model_path, quiet=False)
        
    return tf.keras.models.load_model(model_path)

model = load_model()
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

uploaded_file = st.file_uploader("Select Patient MRI (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Scanned MRI Image', use_container_width=True)

    st.info("🔄 Processing image through the doc.ai Neural Network...")

    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # 4. Generate the prediction
    predictions = model.predict(img_array)

    # FIX: Removed tf.nn.softmax(). The model already outputs correct percentages!
    score = predictions[0]
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    # 5. Safety Protocol (Uncertainty Threshold)
    if confidence >= 80.0:
        st.markdown(f"""
        <div class="prediction-box">
            <h3 style="margin-top: 0; color: #0056D2;">Diagnosis: {predicted_class}</h3>
            <p style="margin-bottom: 0; font-size: 18px; font-weight: bold; color: #2C3E50;">AI Confidence: {confidence:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="warning-box">
            <h3 style="margin-top: 0; color: #D97706;">Diagnosis: {predicted_class}</h3>
            <p style="margin-bottom: 0; font-size: 18px; font-weight: bold; color: #2C3E50;">AI Confidence: {confidence:.2f}% (Low)</p>
        </div>
        """, unsafe_allow_html=True)
        st.warning("⚠️ **AI Uncertainty Alert:** The neural network is not highly confident about this scan. Please do not rely on this preliminary result. Consult a doctor or professional radiologist for an accurate manual diagnosis.")
