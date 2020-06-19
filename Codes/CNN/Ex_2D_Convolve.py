import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from scipy import signal

img = imread('lena_gray.png')
plt.subplot(221)
plt.imshow(img, cmap='gray', interpolation='nearest')
print(img.shape)
# smoothing operation
b1=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])/9
convImg1 = signal.convolve2d(img, b1) 
plt.subplot(222)
plt.imshow(convImg1, cmap='gray', interpolation='nearest')

# edge detection via Sobel operator (horizontal gradient)
b2=np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
convImg2 = np.abs(signal.convolve2d(img, b2)) 
plt.subplot(223)
plt.imshow(convImg2, cmap='gray', interpolation='nearest')

# edge detection via Sobel operator (vertical gradient)
b3=np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
convImg3 = np.abs(signal.convolve2d(img, b3)) 
plt.subplot(224)
plt.imshow(convImg3, cmap='gray', interpolation='nearest')
plt.show()
