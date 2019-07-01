import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

root="./"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def standardize(x, mean, std):
	return (x - mean)/(std)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(       # input_size=(3*36*36) 
             torch.nn.Conv2d(3, 6, 5, 1, 2), 
             torch.nn.ReLU(), 
             torch.nn.MaxPool2d(kernel_size=2, stride=2), # output_size=(6*18*18) 
        )
        self.conv2 = torch.nn.Sequential( 
             torch.nn.Conv2d(6, 16, 5, 1, 2), 
             torch.nn.ReLU(), 
             torch.nn.MaxPool2d(2, 2) # output_size=(16*9*9) 
        ) 
        self.fc1 = torch.nn.Sequential( 
             torch.nn.Linear(16*9*9, 200), 	
             torch.nn.ReLU() 
        ) 
        self.fc2 = torch.nn.Sequential( 
             torch.nn.Linear(200, 80), 
             torch.nn.ReLU() 
        ) 

        self.fc3 = torch.nn.Linear(80, 36)   # 36 classes,  0~9, a~z => 36

    def forward(self, x):
        conv1_out = self.conv1(x)
        #print('conv1_out = ', conv1_out.shape)
        conv2_out = self.conv2(conv1_out)
        #print('conv2_out = ', conv2_out.shape)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        res = conv2_out.view(conv2_out.size(0), -1)
        #print('res = ', res.shape)
        o1  = self.fc1(res)
        #print('o1 = ', o1.shape)
        o2  = self.fc2(o1)
        out = self.fc3(o2)
        return out


load_model = Net()

load_model.load_state_dict(torch.load('./model/cnn36x36_dict.pt'))

loss_func2 = torch.nn.CrossEntropyLoss()
print (load_model)

model = load_model.to(device)
model.eval() 

img = cv2.imread('./evl1.jpg', 0) 

print ('img =', img.shape) 
img = cv2.resize(img, (36,36)) 

arr = np.asarray(img, dtype="float32") 
print ('arr type= ', arr.shape) 

plt.figure()
plt.imshow(arr) 
plt.show()  # display i

xxx = [arr, arr, arr];
arr_xxx = np.array(xxx)
data_x = np.array([arr_xxx])

data_x = data_x / 255 

data_x = torch.from_numpy(data_x)
data_x = data_x.cuda()

#print ('data_x = ', data_x)

model_out = model(data_x)

#print ('model_out = ', model_out)

pred_y = torch.max(model_out, 1)[1]

print('pred result =', pred_y)

