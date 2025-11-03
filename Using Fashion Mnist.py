import torch 
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets , transforms
from torch.utils.data.sampler import SubsetRandomSampler 
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,),)])

trainset = datasets.FashionMNIST('-/.pytorch/F_MNIST_data',download= True,train = True, transform = transform )
testset = datasets.FashionMNIST('-/pytorch/F_MNIST_data',download=True,train = False, transform = transform)

indices = list(range(len(trainset)))
np.random.shuffle(indices)

split = int(np.floor(0.2 * len(trainset)))
train_sample = SubsetRandomSampler(indices[:split])
valid_sample = SubsetRandomSampler(indices[split:])

trainloader = torch.utils.data.DataLoader(trainset ,sampler = train_sample ,batch_size=64)
validloader = torch.utils.data.DataLoader(trainset ,sampler=valid_sample ,batch_size=64)
testloader =  torch.utils.data.DataLoader(testset , batch_size=64, shuffle=True)

trainset
trainloader

dataiter = iter(trainloader)
# print("dataIter :",dataiter)
try:
    images ,labels = next(dataiter)
except Exception as e:
    print("Error while loading data:",e)

fig = plt.figure(figsize=(15,5))

for idx in np.arange(min(20,images.size(0))):
    ax = fig.add_subplot(4,int(20/4),idx+1 ,xticks=[],yticks=[])
    ax.imshow(np.squeeze(images[idx]),cmap='gray')
    ax.set_title(labels[idx].item())

fig.tight_layout()

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.fcl = nn.Linear(784, 120)
        self.fc2 = nn.Linear(120, 120)
        self.fc3 = nn.Linear(120, 10)
        self.dropout = nn.Dropout(0.2)


    def forward(self ,x):
        x = x.view(x.shape[0], -1)
        x = self.dropout(F.relu(self.fcl(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.log_softmax(self.fc3(x),dim=1)
        return x
    
model = Classifier()

criterion = nn.NLLLoss()

optimizer = optim.SGD(model.parameters(),lr=0.01)

valid_loss_min =np.inf

epochs = 2

steps = 0

model.train()

train_losses , valid_losses =[] ,[]

for e in range(epochs):
    running_loss = 0
    valid_loss = 0

    for images ,labels in trainloader:
        optimizer.zero_grad()
        log_ps = model(images)
        loss = criterion(log_ps ,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    for images , labels in validloader:
        log_ps =model(images)
        loss = criterion(log_ps ,labels)
        valid_loss +=loss.item() * images.size(0) 

    running_loss = running_loss /len(trainloader.sampler)
    valid_loss = valid_loss / len(validloader.sampler)

    train_losses.append(running_loss)
    valid_losses.append(valid_loss)

    print('Epoch: {}  Traning Loss :{:.6f} \t Validation Loss: {:.6f}'.format(e+1 ,running_loss ,valid_loss))
    if valid_loss <=valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}) .Saving Model ...'.format(e+1 ,running_loss ,valid_loss))
        torch.save(model.state_dict(),'model.pt')
        valid_loss_min  = valid_loss

plt.plot(train_losses , label='Train Loss')
plt.plot(valid_losses , label='Valid Loss')
plt.legend()

model.load_state_dict(torch.load('model.pt'))

test_loss = 0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval()
for images,labels in testloader:
    output = model(images)
    loss = criterion(output , labels)
    test_loss +=loss.item()*images.size(0)
    _, pred =torch.max(output ,1)
    correct = np.squeeze(pred.eq(labels.data.view_as(pred)))
    for i in range(len(labels)):
        label = labels.data[i]
        class_correct[label] +=correct[i].item()
        class_total[label] +=1

test_loss = test_loss/len(testloader.sampler)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)'%
              (str(i),100*class_correct[i]/class_total[i],np.sum(class_correct[i]),np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s:  N/A(no traning examples)' % classes[i])

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)'%(
    100.*np.sum(class_correct) /np.sum(class_total),
    np.sum(class_correct),np.sum(class_total)
))


dataiter =iter(testloader)
images , labels = next(dataiter)
putput = model(images)
_,preds = torch.max(output,1)
images =images.numpy()

fig = plt.figure(figsize=(25,4))
for idx in np.arange(20):
    ax = fig.add_subplot(2,int(20/2),idx+1, xticks=[],yticks=[])
    ax.imshow(np.squeeze(images[idx]),cmap='gray')
    ax.set_title("{} ({})".format(str(preds[idx].item()),str(labels[idx].item())),color =("green" if preds[idx].item()==labels[idx].item() else "red"))



