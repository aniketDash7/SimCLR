# SimCLR Remote Sensing Adaptation
This repository contains a PyTorch implementation of SimCLR (Simple Framework for Contrastive Learning of Visual Representations) adapted for Remote Sensing tasks, specifically Land Use Classification on the UC Merced dataset.

## Project Structure
```
SimCLR-RemoteSensing/
├── app.py                 # Streamlit Production Demo
├── data/                  # Dataset storage
├── notebooks/             # Jupyter Notebooks (Colab)
├── scripts/
│   ├── train.py           # Training/Fine-tuning script
│   └── visualize.py       # t-SNE Visualization script
├── src/
│   ├── model.py           # SimCLR Model Definition
│   └── __init__.py
└── requirements.txt       # Dependencies
```

## Setup
1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Download Pretrained Weights**:
    Ensure `simclr_model_RN101.pth` (pretrained on EuroSAT) is in the root directory or `scripts/` folder as needed.

## Usage

### 1. Training / Fine-Tuning
To fine-tune the ResNet-101 backbone on the UC Merced dataset:
```bash
python scripts/train.py
```
This will:
- Download the UC Merced dataset automatically.
- Load the pretrained SimCLR weights.
- Fine-tune the model.
- Save the best model as `finetuned_simclr_ucmerced.pth` (or similar).

### 2. Visualization
To generate a 3D t-SNE animation of the embeddings:
```bash
python scripts/visualize.py
```

### 3. Production Demo
To run the interactive web interface:
```bash
streamlit run app.py
```
Upload an aerial image to get a classification prediction.

## Deployment

### 1. GitHub Setup
This repository includes a `.gitignore` to exclude large model files and data.
1.  Initialize a git repo: `git init`
2.  Commit your code: `git add . && git commit -m "Initial commit"`
3.  Push to GitHub.

### 2. Hosting the Model
Since the model (`.pth`) is too large for GitHub (>100MB):
1.  Upload your `finetuned_simclr_ucmerced.pth` to a cloud provider (Google Drive, Dropbox, Hugging Face).
2.  Get a **direct download link**.
3.  Update `MODEL_URL` in `app.py` with this link.

### 3. Streamlit Cloud
1.  Go to [share.streamlit.io](https://share.streamlit.io/).
2.  Connect your GitHub repository.
3.  Deploy! The app will automatically download the model from your link on startup.

## Model Details
- **Backbone**: ResNet-101
- **Pretraining**: SimCLR on EuroSAT (Sentinel-2)
- **Downstream**: Fine-tuned on UC Merced (Aerial Imagery)
