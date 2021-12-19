import numpy as np
from scipy import ndimage
import cv2
import torch
import torch.nn as nn
import pickle



class custom_loss(nn.Module):
    """Custom loss"""
    
    def __init__(self):
        super(L1_Charbonnier, self).__init__()
        self.eps = 1e-3
    
    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt( diff * diff + self.eps )
        loss = torch.sum(error)
        return loss
