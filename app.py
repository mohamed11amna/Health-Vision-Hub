import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Health Vision Hub",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-title {font-size: 3rem; color: #2E86C1; text-align: center;}
    .sub-title {font-size: 1.5rem; color: #555; text-align: center; margin-bottom: 2rem;}
    .report-box {border: 2px solid #ddd; padding: 20px; border-radius: 10px; background-color: #f9f9f9;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🏥 Health Vision Hub</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Powered Multi-Disease Diagnosis System</div>', unsafe_allow_html=True)

# --- 1. MODEL LINKS ---
MODEL_LINKS = {
    "eye": "https://drive.google.com/file/d/1a4xISBHgmdG7wRcd5vui9F4vjocFYh7k/view?usp=drivesdk",
    "chest": "https://drive.google.com/file/d/1M1V7nCc6-lj8cqrhy0D4fQxW8hQ0mmHy/view?usp=drivesdk",
    "skin": "https://drive.google.com/file/d/1HdqtSAFEhHP0pIITgOo0Ncpp_vltS-qc/view?usp=drivesdk"
}

# --- 2. CLASS LABELS ---
CHEST_CLASSES = ['Atelectasis', 'Effusion', 'Infiltration', 'No Finding', 'Nodule']
SKIN_CLASSES = ['1st Degree Burn', '2nd Degree Burn', '3rd Degree Burn']
# Note: EYE_CLASSES is removed because the Eye model is Binary (Healthy vs Sick)

# --- 3. HELPER FUNCTIONS ---

@st.cache_resource
def load_model_from_drive(model_type):
    """Downloads model from Drive (if not present) and loads it."""
    url = MODEL_LINKS.get(model_type)
    if not url or "PASTE" in url:
        st.error(f"⚠️ Please put the Google Drive link for the {model_type} model in the code!")
        return None

    filename = f"{model_type}_model.h5"
    
    if not os.path.exists(filename):
        with st.spinner(f"Downloading {model_type.title()} Model... (One-time setup)"):
            try:
                gdown.download(url, filename, quiet=False, fuzzy=True)
            except Exception as e:
                st.error(f"Download failed: {e}")
                return None
    
    try:
        model = tf.keras.models.load_model(filename)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# --- MODIFIED: Accepts target_size AND scaling option ---
def preprocess_image(uploaded_file, target_size=(224, 224), scale=True):
    """
    Resizes the image.
    scale=True: Divides by 255 (for Skin models).
    scale=False: Keeps raw pixels 0-255 (for Eye/Chest models).
    """
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Patient Scan", width=300)
    
    # Resize to the specific target size required by the model
    image = image.resize(target_size)
    
    img_array = np.array(image)
    
    # Only normalize if the model was trained with scaling
    if scale:
        img_array = img_array / 255.0
    
    img_array = np.expand_dims(img_array, axis=0) # Batch dimension
    return img_array

# --- 4. SIDEBAR ---
st.sidebar.header("Select Department")
app_mode = st.sidebar.radio("Go to:", 
    ["👁️ Diabetic Retinopathy (Eye)", "🩻 Chest Disease Analysis", "🔥 Skin Burn Level"])

st.sidebar.markdown("---")
st.sidebar.info("Health Vision Hub uses EfficientNetB3 models for high-accuracy medical image analysis.")

# --- 5. APP MODULES ---

# === EYE MODULE (Binary Model) ===
if "Eye" in app_mode:
    st.header("👁️ Diabetic Retinopathy (DR) Diagnosis")
    st.write("Upload a Retinal Fundus image.")
    
    model = load_model_from_drive("eye")
    
    uploaded_file = st.file_uploader("Upload Eye Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file and model:
        # FIX 1 & 2: Size 300x300 AND scale=False (Raw Pixels)
        processed_img = preprocess_image(uploaded_file, target_size=(300, 300), scale=False)
        
        if st.button("Analyze Eye"):
            with st.spinner("Analyzing Retina..."):
                # FIX 3: Binary Prediction (Single Value)
                # Returns something like [[0.02]] or [[0.98]]
                prediction = model.predict(processed_img)[0][0]
                
                # Logic: 0 = Healthy, 1 = Sick (Any Stage)
                # Training code: '0' if x == 0 else '1'
                
                confidence = prediction * 100
                
                st.markdown("### Diagnosis Report")
                
                if prediction < 0.5:
                    # Score is close to 0 -> Healthy
                    safe_confidence = (1 - prediction) * 100
                    st.success(f"✅ **Result: HEALTHY (No DR)**")
                    st.write(f"Confidence: {safe_confidence:.2f}%")
                else:
                    # Score is close to 1 -> Sick
                    st.error(f"⚠️ **Result: SICK (Diabetic Retinopathy Detected)**")
                    st.write(f"Confidence: {confidence:.2f}%")
                    st.warning("Note: This model detects the presence of DR. Please consult an ophthalmologist for staging.")

# === CHEST MODULE ===
elif "Chest" in app_mode:
    st.header("🩻 Chest X-Ray Analysis")
    st.write("Detects: Atelectasis, Effusion, Infiltration, Nodule, or No Finding.")
    
    model = load_model_from_drive("chest")
    
    uploaded_file = st.file_uploader("Upload Chest X-Ray", type=["jpg", "png", "jpeg"])
    
    if uploaded_file and model:
        # Chest: 224x224, Raw Pixels (scale=False)
        processed_img = preprocess_image(uploaded_file, target_size=(224, 224), scale=False)
        
        if st.button("Analyze X-Ray"):
            with st.spinner("Scanning Lungs..."):
                prediction = model.predict(processed_img)[0]
                
                st.markdown("### Diagnosis Report")
                found_disease = False
                
                # Check "No Finding" first (Index 3)
                no_finding_score = prediction[3] 
                
                if no_finding_score > 0.5 and max(prediction) == no_finding_score:
                     st.success(f"✅ **Result: No Findings (Healthy)**")
                     st.write(f"Confidence: {no_finding_score*100:.2f}%")
                else:
                    for i, score in enumerate(prediction):
                        label = CHEST_CLASSES[i]
                        if label != "No Finding" and score > 0.5:
                            st.error(f"⚠️ **Detected: {label}** ({score*100:.2f}%)")
                            found_disease = True
                    
                    if not found_disease:
                        st.info("No specific disease crossed the 50% threshold.")

# === SKIN BURN MODULE ===
elif "Skin" in app_mode:
    st.header("🔥 Skin Burn Severity Level")
    st.write("Classifies burns into 1st, 2nd, or 3rd degree for first aid advice.")
    
    model = load_model_from_drive("skin")
    
    uploaded_file = st.file_uploader("Upload Burn Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file and model:
        # Skin: 224x224, Normalized (scale=True)
        processed_img = preprocess_image(uploaded_file, target_size=(224, 224), scale=True)
        
        if st.button("Analyze Burn"):
            with st.spinner("Analyzing Tissue Damage..."):
                prediction = model.predict(processed_img)[0]
                
                predicted_class_index = np.argmax(prediction)
                confidence = np.max(prediction) * 100
                result_label = SKIN_CLASSES[predicted_class_index]
                
                st.markdown(f"### Diagnosis: **{result_label}**")
                st.progress(int(confidence))
                st.write(f"Confidence: {confidence:.2f}%")
                
                st.markdown("#### 🚑 First Aid Advice:")
                if predicted_class_index == 0:
                    st.info("Cool the burn with cool running water. Apply aloe vera. Do not use ice.")
                elif predicted_class_index == 1:
                    st.warning("Cool the area. Do NOT break blisters. Apply antibiotic ointment.")
                elif predicted_class_index == 2:
                    st.error("🚨 EMERGENCY: Call 911. Cover with clean cloth.")
