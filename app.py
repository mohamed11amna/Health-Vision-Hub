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

# --- 1. MODEL LINKS (PASTE YOUR LINKS HERE) ---
MODEL_LINKS = {
    # Replace these strings with your actual Google Drive Share Links
    "eye": "https://drive.google.com/file/d/1a4xISBHgmdG7wRcd5vui9F4vjocFYh7k/view?usp=drivesdk",
    "chest": "https://drive.google.com/file/d/1M1V7nCc6-lj8cqrhy0D4fQxW8hQ0mmHy/view?usp=drivesdk",
    "skin": "https://drive.google.com/file/d/1HdqtSAFEhHP0pIITgOo0Ncpp_vltS-qc/view?usp=drivesdk"
}

# --- 2. CLASS LABELS ---
# Eye: Standard 5 stages. Logic: Index 0 is Healthy, 1-4 are Sick
EYE_CLASSES = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']

# Chest: Your "Big 5"
CHEST_CLASSES = ['Atelectasis', 'Effusion', 'Infiltration', 'No Finding', 'Nodule']

# Skin Burn: 3 Classes
SKIN_CLASSES = ['1st Degree Burn', '2nd Degree Burn', '3rd Degree Burn']

IMG_SIZE = 224

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

def preprocess_image(uploaded_file):
    """Resizes and normalizes the image for B3 models."""
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Patient Scan", width=300) # Show image
    
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image)
    img_array = img_array / 255.0  # Normalize [0,1]
    img_array = np.expand_dims(img_array, axis=0) # Batch dimension
    return img_array

# --- 4. SIDEBAR ---
st.sidebar.header("Select Department")
app_mode = st.sidebar.radio("Go to:", 
    ["👁️ Diabetic Retinopathy (Eye)", "🩻 Chest Disease Analysis", "🔥 Skin Burn Level"])

st.sidebar.markdown("---")
st.sidebar.info("Health Vision Hub uses EfficientNetB3 models for high-accuracy medical image analysis.")

# --- 5. APP MODULES ---

# === EYE MODULE ===
if "Eye" in app_mode:
    st.header("👁️ Diabetic Retinopathy (DR) Diagnosis")
    st.write("Upload a Retinal Fundus image.")
    
    model = load_model_from_drive("eye")
    
    uploaded_file = st.file_uploader("Upload Eye Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file and model:
        processed_img = preprocess_image(uploaded_file)
        
        if st.button("Analyze Eye"):
            with st.spinner("Analyzing Retina..."):
                prediction = model.predict(processed_img)[0]
                
                # Logic: Softmax (Single Label)
                predicted_class_index = np.argmax(prediction)
                confidence = np.max(prediction) * 100
                result_label = EYE_CLASSES[predicted_class_index]
                
                st.markdown("### Diagnosis Report")
                
                # Logic: Index 0 is Healthy, others are Sick
                if predicted_class_index == 0:
                    st.success(f"✅ **Result: HEALTHY ({result_label})**")
                    st.write(f"Confidence: {confidence:.2f}%")
                else:
                    st.error(f"⚠️ **Result: SICK - Detected {result_label}**")
                    st.write(f"Confidence: {confidence:.2f}%")
                    st.warning("Recommendation: Please consult an ophthalmologist immediately.")

# === CHEST MODULE ===
elif "Chest" in app_mode:
    st.header("🩻 Chest X-Ray Analysis")
    st.write("Detects: Atelectasis, Effusion, Infiltration, Nodule, or No Finding.")
    
    model = load_model_from_drive("chest")
    
    uploaded_file = st.file_uploader("Upload Chest X-Ray", type=["jpg", "png", "jpeg"])
    
    if uploaded_file and model:
        processed_img = preprocess_image(uploaded_file)
        
        if st.button("Analyze X-Ray"):
            with st.spinner("Scanning Lungs..."):
                # Logic: Sigmoid (Multi-Label) -> Returns 5 scores
                prediction = model.predict(processed_img)[0]
                
                st.markdown("### Diagnosis Report")
                
                # We interpret the Big 5
                # prediction order corresponds to alphabetic order of classes usually, 
                # OR the order you defined in training. 
                # Assuming standard order: Atelectasis, Effusion, Infiltration, No Finding, Nodule
                
                found_disease = False
                
                # Check "No Finding" first (Index 3 in your list)
                no_finding_score = prediction[3] 
                
                if no_finding_score > 0.5 and max(prediction) == no_finding_score:
                     st.success(f"✅ **Result: No Findings (Healthy)**")
                     st.write(f"Confidence: {no_finding_score*100:.2f}%")
                else:
                    # Check other diseases
                    for i, score in enumerate(prediction):
                        label = CHEST_CLASSES[i]
                        if label != "No Finding" and score > 0.5: # Threshold 50%
                            st.error(f"⚠️ **Detected: {label}** ({score*100:.2f}%)")
                            found_disease = True
                    
                    if not found_disease:
                        st.info("No specific disease crossed the 50% threshold, but 'No Finding' was also low. Clinical correlation recommended.")

# === SKIN BURN MODULE ===
elif "Skin" in app_mode:
    st.header("🔥 Skin Burn Severity Level")
    st.write("Classifies burns into 1st, 2nd, or 3rd degree for first aid advice.")
    
    model = load_model_from_drive("skin")
    
    uploaded_file = st.file_uploader("Upload Burn Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file and model:
        processed_img = preprocess_image(uploaded_file)
        
        if st.button("Analyze Burn"):
            with st.spinner("Analyzing Tissue Damage..."):
                prediction = model.predict(processed_img)[0]
                
                # Logic: Softmax (3 Classes)
                predicted_class_index = np.argmax(prediction)
                confidence = np.max(prediction) * 100
                result_label = SKIN_CLASSES[predicted_class_index]
                
                st.markdown(f"### Diagnosis: **{result_label}**")
                st.progress(int(confidence))
                st.write(f"Confidence: {confidence:.2f}%")
                
                st.markdown("#### 🚑 First Aid Advice:")
                if predicted_class_index == 0: # 1st Degree
                    st.info("cool the burn with cool (not cold) running water. Apply aloe vera or lotion. Do not use ice.")
                elif predicted_class_index == 1: # 2nd Degree
                    st.warning("Cool the area. **Do not break blisters.** Apply antibiotic ointment and cover loosely with gauze.")
                elif predicted_class_index == 2: # 3rd Degree
                    st.error("🚨 **EMERGENCY:** Call emergency services. Do not apply water or removing clothing stuck to the burn. Cover with a clean, cool cloth.")