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
        X=X.cpu()
        Y=Y.cpu()
        X=X.detach().numpy()
        Y=Y.detach().numpy()
        X=np.float64(X)
        Y=np.float64(Y)
        img1=img1.cpu()
        img2=img2.cpu()
        img1=img1.detach().numpy()
        img2=img2.detach().numpy()
        img1=np.float64(img1)
        img2=np.float64(img2)

        batch_loss=[]
        for cnt,i in enumerate(img1):
            # print(img1[cnt].shape,img2[cnt].shape)
            h=img1[cnt].shape[2]
            w=img1[cnt].shape[1]
            c=img1[cnt].shape[0]


            X=img1[cnt].reshape((w,h,c))
            Y=img2[cnt].reshape((w,h,c))


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




            loss = (sigma1_sq+sigma2_sq-2*sigma12)
            batch_loss.append(np.mean(loss))
        #return ssim_map.mean()
        return np.mean(batch_loss)
