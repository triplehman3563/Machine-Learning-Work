import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread

### Scartch ##############################################################
def conv2d(X, kernel, pad, stride):
    
    X_pad = np.pad(X, ((pad, pad), (pad, pad)), 'constant', constant_values=0)
    #print(X_pad)
    
    kernel_cov = kernel[::-1, ::-1]
    #print(kernel_cov)
    
    k_x = kernel.shape[0]
    k_y = kernel.shape[1]
    s_x = stride
    s_y = stride
    
    opt_x = int(((X_pad.shape[0]-k_x)/s_x)+1)
    opt_y = int(((X_pad.shape[1]-k_y)/s_y)+1)

    cov = np.zeros((opt_x, opt_y))
    
    for i in range(opt_x):
        for j in range(opt_y):
            cov[i][j] = np.sum(np.multiply(X_pad[i*s_x:i*s_x+k_x, j*s_y:j*s_y+k_y], kernel_cov))
            #print(cov)
    return cov


def max_pool2d(X, pool_size, stride):
    p_x = pool_size
    p_y = pool_size
    s_x = stride
    s_y = stride
    
    opt_x = int(((X.shape[0]-p_x)/s_x)+1)
    opt_y = int(((X.shape[1]-p_y)/s_y)+1)

    mp = np.zeros((opt_x, opt_y))
    
    for i in range(opt_x):
        for j in range(opt_y):
            mp[i][j] = np.max(X[i*s_x:i*s_x+p_x, j*s_y:j*s_y+p_y])
            #print(mp)
    return mp

## Load dataset
img = imread('lena_gray.png')
plt.subplot(221)
plt.imshow(img, cmap='gray', interpolation='nearest')
plt.title('Raw PIC')
print(img.shape)

## define kernel
# edge detection via Sobel operator (vertical gradient)
k=np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
print ('kernel', k)

## convolution
x_0_conv1 = conv2d(img, k, pad=0, stride=1)
print('cov shape ',x_0_conv1.shape)

## plot result
plt.subplot(222)
plt.imshow(x_0_conv1, cmap='gray', interpolation='nearest')
plt.title('After Conv')
plt.tight_layout()

## max pooling
mp1 = max_pool2d(x_0_conv1, pool_size = 2, stride = 2)
print('pool shape',mp1.shape)

## plot result
plt.subplot(223)
plt.imshow(mp1, cmap='gray', interpolation='nearest')
plt.title('After Pooling')
plt.tight_layout()
plt.show()