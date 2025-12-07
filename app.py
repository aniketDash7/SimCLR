import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from src.model import SimCLR

# --- Configuration ---
MODEL_PATH = "finetuned_simclr_ucmerced.pth" # Local path
MODEL_URL = "https://huggingface.co/aniketDS/SimCLR_finetuned/resolve/main/finetuned_ds_state_dict.pth" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = [
    'agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings', 
    'chaparral', 'denseresidential', 'forest', 'freeway', 'golfcourse', 
    'harbor', 'intersection', 'mediumresidential', 'mobilehomepark', 
    'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential', 
    'storagetanks', 'tenniscourt'
]

# --- Helper Functions ---
def download_model(url, dest_path):
    if not os.path.exists(dest_path):
        st.info(f"Downloading model from {url}...")
        import requests
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.success("Model downloaded!")
        else:
            st.error(f"Failed to download model. Status code: {response.status_code}")
            return False
    return True

@st.cache_resource
def load_model(model_path):
    # If model doesn't exist locally, try to download
    if not os.path.exists(model_path):
        if MODEL_URL and "your-model-url-here" not in MODEL_URL:
             if not download_model(MODEL_URL, model_path):
                 return None
        else:
             # If no URL provided and no file, return None
             return None
    
    # Reconstruct model architecture
    backbone = models.resnet101(weights=None)
    # IMPORTANT: The trained model had its fc layer replaced with Identity
    backbone.fc = nn.Identity()
    
    # We need to wrap it exactly as it was saved. 
    # If we saved the whole 'FineTuneModel', we load it directly.
    # If we saved state_dict, we init and load.
    
    # Assuming we are loading the FineTuneModel structure
    model = FineTuneModel(backbone, num_classes=len(CLASSES))
    
    try:
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
    except Exception as e:
        st.error(f"Error loading state_dict: {e}")
        return None

    model.to(DEVICE)
    model.eval()
    return model

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
    
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0) # Add batch dim

# --- UI Layout ---
st.set_page_config(page_title="Land Use Classifier", page_icon="🌍")

st.title("🌍 Land Use Classification")
st.markdown("Powered by **SimCLR (ResNet-101)** | Fine-tuned on **UC Merced**")

# Sidebar for Model
st.sidebar.header("Model Configuration")
uploaded_model = st.sidebar.file_uploader("Upload Model Weights (.pth)", type=["pth"])

model = None
if uploaded_model:
    # Save temp file
    with open("temp_model.pth", "wb") as f:
        f.write(uploaded_model.getbuffer())
    model = load_model("temp_model.pth")
    st.sidebar.success("Custom model loaded!")
else:
    # Attempt to load default model (will download if missing)
    model = load_model(MODEL_PATH)
    
    if model:
        st.sidebar.success(f"Loaded default model from URL")
    else:
        st.sidebar.warning("No model found. Please upload .pth file.")

# Main Interface
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Image")
    uploaded_file = st.file_uploader("Choose an aerial image...", type=["jpg", "png", "jpeg", "tif"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if model:
            # Inference
            input_tensor = preprocess_image(image).to(DEVICE)
            
            # Debug: Show what the model sees
            with st.expander("Debug: See Preprocessed Input"):
                # Denormalize for visualization
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_vis = input_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
                img_vis = std * img_vis + mean
                img_vis = np.clip(img_vis, 0, 1)
                st.image(img_vis, caption="Model Input (224x224 Center Crop)", width=224)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)
                
            prediction = CLASSES[pred.item()]
            confidence = conf.item()

with col2:
    st.subheader("Prediction")
    if uploaded_file is not None and model:
        st.metric(label="Class", value=prediction.title())
        st.metric(label="Confidence", value=f"{confidence:.2%}")
        
        # Bar Chart of Top 5
        st.markdown("### Top 5 Probabilities")
        top5_prob, top5_catid = torch.topk(probs, 5)
        
        top5_probs = top5_prob.cpu().numpy()[0]
        top5_classes = [CLASSES[id] for id in top5_catid.cpu().numpy()[0]]
        
        fig, ax = plt.subplots()
        y_pos = np.arange(len(top5_classes))
        ax.barh(y_pos, top5_probs, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top5_classes)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Probability')
        st.pyplot(fig)

    elif uploaded_file is None:
        st.info("Upload an image to see predictions.")
    elif not model:
        st.error("Model not loaded.")

st.markdown("---")
st.markdown("Built with Streamlit & PyTorch")
