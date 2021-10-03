import numpy as np
from scipy import ndimage
import cv2
import torch
import torch.nn as nn
import pickle

class custom_loss(nn.Module):
    """Custom loss"""
    def __init__(self):
        super(custom_loss, self).__init__()
        self.C1 = (0.01 * 255)**2
        self.C2 = (0.03 * 255)**2

    def forward(self, X, Y):
        with open('X.pkl','wb') as f:
            pickle.dump(X,f)
        with open('Y.pkl','wb') as f:
            pickle.dump(Y,f)
        X=X.cpu()
        Y=Y.cpu()
        X=X.detach().numpy()
        Y=Y.detach().numpy()
        X=np.float64(X)
        Y=np.float64(Y)
        print(X)
        print('\n')
        print(Y)
        kernel = cv2.getGaussianKernel(3, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(X, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(Y, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2



        #ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *(sigma1_sq + sigma2_sq + C2))

        loss = (sigma1_sq+sigma2_sq-2*sigma12)
        #return ssim_map.mean()
        return loss.mean()
