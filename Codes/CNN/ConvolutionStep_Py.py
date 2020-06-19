import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
df = pd.read_csv('mnist_784.csv', header=0)
x_0 = df.iloc[0, 0:-1].values
print(x_0.shape)

# change to 2D
x_0 = x_0.reshape(28, 28)
print(x_0.shape)
plt.imshow(x_0, cmap='gray', interpolation='nearest')
plt.title('Raw PIC')
plt.tight_layout()
plt.show()

## define kernel
np.random.seed(1)
k = np.random.rand(5, 5)
print('kernel', k)

## convolution
x_0_conv1 = conv2d(x_0, k, pad=0, stride=1)
print(x_0_conv1.shape)

## plot result
plt.imshow(x_0_conv1, cmap='gray', interpolation='nearest')
plt.title('After Conv')
plt.tight_layout()
plt.show()

## max pooling
mp1 = max_pool2d(x_0_conv1, pool_size = 2, stride = 2)

## plot result
plt.imshow(mp1, cmap='gray', interpolation='nearest')
plt.title('After Pooling')
plt.tight_layout()
plt.show()