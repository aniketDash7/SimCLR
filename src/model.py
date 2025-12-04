import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

LARGE_NUM = 1e9

def nt_xent(z: torch.Tensor, perm: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Pairwise normalized temperature-scaled cross-entropy loss (NT-Xent)
    """
    # Normalize features
    features = F.normalize(z, dim=1)
    sim = features @ features.T  # Cosine similarity

    # Mask self-similarities by subtracting a large number
    sim.fill_diagonal_(-LARGE_NUM)

    # Scale by temperature
    sim /= tau

    # Cross-entropy loss
    return F.cross_entropy(sim, perm)

class SimCLR(nn.Module):
    """
    SimCLR Implementation with NT-Xent loss
    """
    def __init__(self, backbone: nn.Module, tau: float, feat_dim: int = 256):
        super(SimCLR, self).__init__()
        self.backbone = backbone
        self.tau = tau

        # Define the projection head
        # ResNet backbones usually end with a fc layer. We use its input features.
        z_dim = self.backbone.fc.in_features
        
        # Remove the classification head
        self.backbone.fc = nn.Identity()  
        
        self.projection_head = nn.Sequential(
            nn.Linear(z_dim, z_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(z_dim, feat_dim, bias=False)
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for two augmented views of the same image
        """
        b = x1.size(0)

        # (2b, c, h, w)
        xp = torch.cat((x1, x2))  

        # Permutation for positive pairs
        perm = torch.cat((torch.arange(b) + b, torch.arange(b)), dim=0).to(x1.device)
        
        # (2b, z_dim)
        h = self.backbone(xp) 
        # (2b, feat_dim)
        z = self.projection_head(h)  

        return nt_xent(z, perm, tau=self.tau)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the backbone (before projection head)
        """
        self.eval()
        with torch.no_grad():
            # (B, z_dim)
            h = self.backbone(x)
        return h
