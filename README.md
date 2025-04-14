

## SimCLR + ResNet18: Self-Supervised Learning and Downstream Classification

---

### **Overview**

This notebook demonstrates a self-supervised learning pipeline using the **SimCLR** framework and a **ResNet-18** backbone. The aim is to pretrain a model on unlabeled data using contrastive learning and then transfer the learned representation to a supervised downstream classification task.

---

### **Steps to Run the Notebook**

1. **Import necessary libraries**:
   - PyTorch and torchvision for model and data utilities
   - Hugging Face `datasets` to load and process image datasets
   - SimCLR implementation (defined in a class within the notebook)

2. **Set up GPU and load dataset**:
   - Automatically detects GPU/CPU (`device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`)
   - Loads image dataset using Hugging Face's `load_dataset()` and applies SimCLR-style augmentations

3. **Define SimCLR model**:
   - Constructs a custom `SimCLR` model with a ResNet-18 backbone
   - Projection head includes a two-layer MLP for contrastive representation

4. **Train SimCLR (optional)**:
   - Includes training loop for pretraining using contrastive loss (NT-Xent)
   - In this version, training is skipped and a pretrained model is loaded instead

5. **Load Pretrained SimCLR Model**:
   ```python
   backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
   model = SimCLR(backbone=backbone, tau=0.1).to(device)
   model.load_state_dict(torch.load(PATH, weights_only=True))
   ```

6. **Extract Backbone and Fine-tune on Downstream Task**:
   - Backbone’s final `fc` layer is replaced with `nn.Identity()`
   - A custom `ClassificationModel` is defined to use the frozen backbone and a new classification head

7. **Train and Evaluate Classification Model**:
   - Uses cross-entropy loss and Adam optimizer
   - Evaluates using accuracy on a validation set

---

### **Key Code Blocks & Explanations**

#### SimCLR Model
```python
class SimCLR(nn.Module):
    def __init__(self, backbone, projection_dim=256, tau=0.1):
        ...
```
- Combines ResNet-18 feature extractor with an MLP projection head
- Uses cosine similarity and temperature-scaled NT-Xent loss

#### Contrastive Loss
```python
def contrastive_loss(z_i, z_j, temperature):
    ...
```
- Pairs of augmented views of the same image are encouraged to be close
- Other pairs are pushed apart in representation space

#### Downstream Classification
```python
class ClassificationModel(nn.Module):
    ...
```
- Wraps the ResNet backbone with a classification head
- Handles the case where `fc` is replaced by `nn.Identity()`

---

### **Challenges and Solutions**

#### **1. Model Loading: Accessing `fc.in_features` after SimCLR training**
- **Problem**: The final `fc` layer in ResNet was replaced by `nn.Identity()` during SimCLR, causing an error when trying to access `fc.in_features`.
- **Solution**: Used a dummy input to the backbone to infer the output feature size dynamically for the classifier layer.

#### **2. Incorrect loading of SimCLR model**
- **Problem**: Instead of using the recommended method to load the model (`simclr_model = torch.load(...)`), a custom `SimCLR` class was re-instantiated and state_dict loaded.
- **Solution**: This was actually fine as long as the structure of the model matched exactly. Just ensured weights were loaded using `weights_only=True`.

#### **3. Projection head interference in downstream**
- **Problem**: Needed to ensure the classifier didn’t accidentally include the projection head from SimCLR.
- **Solution**: Separated and extracted the encoder-only part (`backbone`) from the SimCLR model.

---

### 📈 **Results**

- The fine-tuned classification model successfully used the pretrained encoder to achieve reasonable accuracy on the downstream task.
- The modular design allows for experimenting with different datasets, backbones, and projection heads.

---

### **Conclusion & Takeaways**

- Self-supervised contrastive learning like SimCLR enables strong feature extractors without requiring labels.
- Downstream classification is efficient and effective when pretrained representations are properly leveraged.
- It's crucial to manage model internals like `fc` layers and projection heads to avoid errors during fine-tuning.
- SimCLR works best with strong augmentations and a well-structured projection head.

---

