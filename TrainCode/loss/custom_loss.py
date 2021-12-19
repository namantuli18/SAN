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
        self.eps = 1e-3
    def my_filter2D(self,image, kernel):
        image=image.cpu().detach().numpy()
        kernel=kernel.cpu().detach().numpy()
        return torch.tensor(cv2.filter2D(image, -1, kernel),device='cuda:0', requires_grad=True)
    
    def utils(self,X,window):
        dest=torch.empty(X.shape,device='cuda:0')
        anchor=torch.tensor([-1,-1],device='cuda:0')
        w,h,c=X.shape
        for i in range(w):
            for j in range(h):
                s=0
                for ki in range(window.shape[0]):
                    for kj in range(window.shape[1]):
                        try:
                            s+=window[ki][kj]*X[i+ki+1][j+kj+1]
                        except:
                            pass

                dest[i][j]=s
        return dest
    
    
    def forward(self,X, Y):
        X=X/255
        Y=Y/255
        batch_loss=torch.tensor(0.0,device='cuda:0', requires_grad=True)
        for cnt,i in enumerate(X):
            c,h,w=X[cnt].shape
            x=X[cnt].reshape((w,h,c))
            y=Y[cnt].reshape((w,h,c))
            x=x.cpu().detach().numpy()
            y=y.cpu().detach().numpy()
            kernel=torch.tensor(([[0.30780133],
                [0.38439734],
                [0.30780133]]),device='cuda:0')
            window=torch.tensor(([[0.30780133],
                [0.38439734],
                [0.30780133]]),device='cuda:0')
            mu1=cv2.filter2D(x,window)[5:-5, 5:-5]
            mu2=cv2.filter2D(y,window)[5:-5, 5:-5]
            mu1_sq=torch.mul(mu1,mu1)
            mu2_sq=torch.mul(mu2,mu2)
            mu1_mu2=torch.mul(mu1,mu2)
            sigma1_sq=cv2.filter2D(torch.mul(x,x),window)[5:-5,5:-5]-mu1_sq
            sigma2_sq=cv2.filter2D(torch.mul(y,y),window)[5:-5,5:-5]-mu2_sq
            sigma12=cv2.filter2D(torch.mul(x,y),window)[5:-5,5:-5]-mu1_mu2
            loss=torch.sub(torch.mul(sigma1_sq,sigma2_sq),2*sigma12)
            batch_loss=torch.add(loss.mean(),batch_loss)
            print("Batch : {} :: Loss : {}".format(cnt,batch_loss))
        return batch_loss
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
