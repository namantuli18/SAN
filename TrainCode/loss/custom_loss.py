import numpy as np
from scipy import ndimage
import cv2
import torch
import torch.nn as nn
import pickle

def my_filter2D(image, kernel):
    kx, ky = kernel.shape
    x_n = kx//2
    y_n = kx//2
    padded_image = torch.nn.functional.pad(image, (0, 0, ky//2, ky//2, kx//2, kx//2))
    filtimg = torch.Tensor(image)
    px, py, nc = padded_image.shape
    for x in range(px):
        for y in range(py):
            for c in range(nc):
                if x - x_n < 0:
                    continue
                if x + x_n >= px:
                    continue
                if y - y_n < 0:
                    continue
                if y + y_n >= py:
                    continue
                total = 0.0
                image_n = padded_image[x-x_n:x+x_n+1, y-y_n:y+y_n+1,c]
                for x_i in range(kx):
                    for y_i in range(ky):
                        total += image_n[x_i,y_i]*kernel[x_i,y_i]
                filtx = x-x_n
                filty = y-y_n
                filtimg[filtx,filty,c] = total
    return filtimg

class custom_loss(nn.Module):
    """Custom loss"""
    def __init__(self):
        super(custom_loss, self).__init__()
        self.C1 = (0.01 * 255)**2
        self.C2 = (0.03 * 255)**2
        self.eps = 1e-3
    
        
    def forward(self,X, Y):
        i=0
        batch_loss=torch.tensor([],device='cuda:0')
        for _ in X:

            kernel=torch.tensor([[0.09474166, 0.11831801, 0.09474166],
           [0.11831801, 0.14776132, 0.11831801],
           [0.09474166, 0.11831801, 0.09474166]],device='cuda:0')
            window=torch.tensor([[0.09474166, 0.11831801, 0.09474166],
           [0.11831801, 0.14776132, 0.11831801],
           [0.09474166, 0.11831801, 0.09474166]],device='cuda:0')
            mu1=my_filter2D(X[i],window)[0][5:-5, 5:-5]
            mu2=my_filter2D(Y[i],window)[0][5:-5, 5:-5]
            mu1_sq=torch.multiply(mu1,mu1)
            mu2_sq=torch.multiply(mu2,mu2)
            mu1_mu2=torch.multiply(mu1,mu2)
            sigma1_sq=my_filter2D(torch.multiply(X[i],X[i]),window)[0][5:-5,5:-5]-mu1_sq
            sigma2_sq=my_filter2D(torch.multiply(Y[i],Y[i]),window)[0][5:-5,5:-5]-mu2_sq
            sigma12=my_filter2D(torch.multiply(X[i],Y[i]),window)[0][5:-5,5:-5]-mu1_mu2
            # sigma1_sq=my_filter2D(torch.multiply(X[i],X[i]),window)[5:-5, 5:-5] - mu1_sq
            # sigma2_sq=my_filter2D(torch.multiply(Y[i],Y[i]),window)[5:-5, 5:-5] - mu2_sq
            # sigma12=my_filter2D(torch.multiply(X[i],Y[i]),window)[5:-5, 5:-5] - mu1_mu2
            loss=torch.subtract(torch.multiply(sigma1_sq,sigma2_sq),2*sigma12)
            batch_loss=torch.add(loss.mean(),batch_loss)
            i+=1
        return batch_loss.sum()
#     def forward(self, X, Y):
#         diff = torch.add(X, -Y)
#         error = torch.sqrt( diff * diff + self.eps )
#         loss = torch.sum(error)
#         return loss
#     def forward(self, X, Y):
#         X=X.cpu()
#         Y=Y.cpu()
#         X=X.detach().numpy()
#         Y=Y.detach().numpy()
#         X=np.float64(X)
#         Y=np.float64(Y)

#         batch_loss=[]
#         for cnt,i in enumerate(X):
            
#             # print(img1[cnt].shape,img2[cnt].shape)
#             h=X[cnt].shape[2]
#             w=X[cnt].shape[1]
#             c=X[cnt].shape[0]

            
#             x=X[cnt].reshape((w,h,c))
#             y=Y[cnt].reshape((w,h,c))
#             x=x/255
#             y=y/255
#             kernel = cv2.getGaussianKernel(3, 1.5)
#             window = np.outer(kernel, kernel.transpose())

#             mu1 = cv2.filter2D(x, -1, window)[5:-5, 5:-5]  # valid
#             mu2 = cv2.filter2D(y, -1, window)[5:-5, 5:-5]
#             mu1_sq = mu1**2
#             mu2_sq = mu2**2
#             mu1_mu2 = mu1 * mu2
#             sigma1_sq = cv2.filter2D(x**2, -1, window)[5:-5, 5:-5] - mu1_sq
#             sigma2_sq = cv2.filter2D(y**2, -1, window)[5:-5, 5:-5] - mu2_sq
#             sigma12 = cv2.filter2D(x * y, -1, window)[5:-5, 5:-5] - mu1_mu2




#             loss = (sigma1_sq*sigma2_sq-2*sigma12)
#             batch_loss.append(np.mean(loss))
#         #return ssim_map.mean() 
#         X=torch.from_numpy(X)
#         Y=torch.from_numpy(Y)
#         returner = torch.from_numpy(np.array(batch_loss)).sum()
#         returner.requires_grad=True
#         return returner
