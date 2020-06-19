import matplotlib.pyplot as plt
import pandas as pd
import torch as t
import torch.nn as nn
import torch.nn.functional as F

### pytorch #########################################################
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 1 output channels, 5x5 square convolution
        self.conv1 = nn.Conv2d(1, 1, 5)
        
    def forward(self, x):
        x = self.conv1(x)  # 2D covolution
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(x, (2, 2))
    
        return x


# Load dataset
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

# change to tensor: 1  sample number, 1 input image channel, 5x5 square convolution
x_0 = x_0.reshape(1, 1, 28, 28)
x_0_tensor = t.tensor(x_0, dtype=t.float)

# Build a model
model = Net()
print('kernel tensor', model.conv1.weight.data)

# forward propagation 
net_out = model(x_0_tensor)

## change back to numpy
x_0_out = net_out.data.numpy()
x_0_out = x_0_out.reshape(12, 12)




## plot result
plt.imshow(x_0_out, cmap='gray', interpolation='nearest')
plt.title('After Conv & Pooling (Pytorch)')
plt.tight_layout()
plt.show()