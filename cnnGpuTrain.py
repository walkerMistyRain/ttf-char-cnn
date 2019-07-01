import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
root="./"


use_gpu = torch.cuda.is_available()
print("GPU status = ", use_gpu)

# -----------------ready the dataset--------------------------
def default_loader(path):
    return Image.open(path).convert('RGB')
class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],int(words[1])))
        self.imgs = imgs
        #self.transform = transform
        self.transform = transforms.Compose([ 
                   transforms.ToTensor()
        ])
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        #print ('before transform img =', img)
        if self.transform is not None:
            img = self.transform(img)
        #print ('after transform img =', img.shape) 
        return img,label

    def __len__(self):
        return len(self.imgs)

#------------------------===========================================
class MyDataset2(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],int(words[1])))
        self.imgs = imgs
        self.transform = transform        
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        return img,label

    def __len__(self):
        return len(self.imgs)


#-----------------create the Net and training------------------------
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Net()
model = model.to(device)

print(model)

train_data=MyDataset(txt=root+'train.txt', transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_data, batch_size=128, shuffle=True)

optimizer = torch.optim.Adam(model.parameters())
loss_func = torch.nn.CrossEntropyLoss()

model.train()
#train
for epoch in range(3):
    print('epoch {}'.format(epoch + 1))
    # training-----------------------------
    train_loss = 0.
    train_acc = 0.
    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        batch_x, batch_y = Variable(batch_x), Variable(batch_y)
        
        out = model(batch_x)
        loss = loss_func(out, batch_y)

        train_loss += loss.item()
        pred = torch.max(out, 1)[1]
        train_correct = (pred == batch_y).sum()

        train_acc += train_correct.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
        train_data)), train_acc / (len(train_data))))


torch.save(model,"model/cnn36x36.pkl")
torch.save(model.state_dict(),'./model/cnn36x36_dict.pt')

load_model = Net()
load_model.load_state_dict(torch.load('./model/cnn36x36_dict.pt'))

test_data=MyDataset(txt=root+'test.txt', transform=transforms.ToTensor())
test_loader = DataLoader(dataset=test_data, batch_size=64)
loss_func2 = torch.nn.CrossEntropyLoss()
print (load_model)

load_model.eval()
for epoch in range(1):   
    # evaluation--------------------------------
    eval_loss = 0.
    eval_acc = 0.
    for batch_x, batch_y in test_loader:

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        batch_x, batch_y = Variable(batch_x), Variable(batch_y)
        out = load_model(batch_x)
        loss = loss_func2(out, batch_y)
        eval_loss += loss.item()
        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch_y).sum()
        eval_acc += num_correct.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_data)), eval_acc / (len(test_data))))



