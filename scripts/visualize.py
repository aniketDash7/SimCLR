import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.animation as animation
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.train import get_uc_merced_loader, load_simclr_model, extract_features, DEVICE, MODEL_PATH, DATA_DIR
import os

def visualize_tsne(features, labels, save_path="tsne_animation.gif"):
    print("Computing t-SNE (this may take a while)...")
    # We use 3 components for a 3D animation
    tsne = TSNE(n_components=3, perplexity=30, n_iter=1000, random_state=42)
    projections = tsne.fit_transform(features)
    
    print("Generating animation...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a colormap
    classes = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(classes)))
    
    scatters = []
    for i, c in enumerate(classes):
        mask = labels == c
        # We can use the class index directly if labels are numeric, 
        # but let's be safe if they are not (though UCMerced targets are ints)
        scatters.append(ax.scatter(projections[mask, 0], projections[mask, 1], projections[mask, 2], 
                                   label=f"Class {c}", s=20, alpha=0.6, color=colors[i]))
        
    ax.set_title("3D t-SNE of SimCLR Embeddings (UC Merced)")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_zlabel("Dim 3")
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small') # Legend might be too crowded
    
    def update(angle):
        ax.view_init(elev=30, azim=angle)
        return scatters

    # Create animation: 360 degrees rotation
    ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50, blit=False)
    
    print(f"Saving animation to {save_path}...")
    ani.save(save_path, writer='pillow', fps=20)
    print("Done!")

def main():
    # 1. Load Data (Use test set for visualization to be cleaner, or a subset)
    print("Preparing Data...")
    _, test_loader = get_uc_merced_loader(DATA_DIR, batch_size=64)
    if test_loader is None:
        return

    # 2. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at {MODEL_PATH}")
        return

    model = load_simclr_model(MODEL_PATH, DEVICE)

    # 3. Extract Features
    print("Extracting features...")
    features, labels = extract_features(model, test_loader, DEVICE)
    print(f"Features shape: {features.shape}")

    # 4. Visualize
    visualize_tsne(features, labels, save_path="uc_merced_tsne_3d.gif")

if __name__ == "__main__":
    main()
