import torch
import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import SimCLR
from torchvision import models

class TestSimCLR(unittest.TestCase):
    def setUp(self):
        # Use a smaller backbone for speed in tests, or standard ResNet18
        self.backbone = models.resnet18(weights=None)
        self.feat_dim = 128
        self.model = SimCLR(backbone=self.backbone, tau=0.1, feat_dim=self.feat_dim)

    def test_simclr_initialization(self):
        """Test if SimCLR initializes correctly and replaces the fc layer."""
        self.assertIsInstance(self.model.backbone.fc, torch.nn.Identity)
        self.assertTrue(hasattr(self.model, 'projection_head'))

    def test_forward_shape(self):
        """Test the forward pass output shape (loss)."""
        batch_size = 4
        # Create dummy inputs (B, C, H, W)
        x1 = torch.randn(batch_size, 3, 224, 224)
        x2 = torch.randn(batch_size, 3, 224, 224)
        
        # Forward pass returns a scalar loss
        loss = self.model(x1, x2)
        self.assertEqual(loss.dim(), 0)
        self.assertIsInstance(loss.item(), float)

    def test_encode_shape(self):
        """Test the encoder output shape."""
        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224)
        
        # ResNet18 backbone output is 512
        features = self.model.encode(x)
        self.assertEqual(features.shape, (batch_size, 512))

if __name__ == '__main__':
    unittest.main()
