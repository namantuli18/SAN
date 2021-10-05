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

        batch_loss=[]
        for cnt,i in enumerate(X):
            # print(img1[cnt].shape,img2[cnt].shape)
            h=X[cnt].shape[2]
            w=X[cnt].shape[1]
            c=X[cnt].shape[0]


            x=X[cnt].reshape((w,h,c))
            y=Y[cnt].reshape((w,h,c))


            kernel = cv2.getGaussianKernel(3, 1.5)
            window = np.outer(kernel, kernel.transpose())

            mu1 = cv2.filter2D(x, -1, window)[5:-5, 5:-5]  # valid
            mu2 = cv2.filter2D(y, -1, window)[5:-5, 5:-5]
            mu1_sq = mu1**2
            mu2_sq = mu2**2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = cv2.filter2D(x**2, -1, window)[5:-5, 5:-5] - mu1_sq
            sigma2_sq = cv2.filter2D(y**2, -1, window)[5:-5, 5:-5] - mu2_sq
            sigma12 = cv2.filter2D(x * y, -1, window)[5:-5, 5:-5] - mu1_mu2




            loss = (sigma1_sq+sigma2_sq-2*sigma12)
            batch_loss.append(np.mean(loss))
        #return ssim_map.mean() 
        X=torch.from_numpy(X)
        Y=torch.from_numpy(Y)
        returner = torch.from_numpy(np.array(batch_loss)).sum()
        returner.requires_grad=True
        return returner
