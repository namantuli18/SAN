import numpy as np
from scipy import ndimage
import cv2
import torch
import torch.nn as nn

class custom_loss(nn.Module):
    """Custom loss"""
    def __init__(self):
        super(custom_loss, self).__init__()
        self.C1 = (0.01 * 255)**2
        self.C2 = (0.03 * 255)**2

    def forward(self, X, Y):
        X=X.numpy()
        Y=Y.numpy()
        kernel = cv2.getGaussianKernel(3, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(X, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(Y, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(X**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(Y**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(X * Y, -1, window)[5:-5, 5:-5] - mu1_mu2



        #ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *(sigma1_sq + sigma2_sq + C2))

        loss = (sigma1_sq+sigma2_sq-2*sigma12)
        #return ssim_map.mean()
        return loss.mean()
