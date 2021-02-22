import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.dataset import Dataset
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt

BATCH_SIZE = 10
ITEM_NUM = 10
TRAIN_NUM = 10000


class CtmDataset(Dataset):
    '''初始化数据集'''

    def __init__(self, img_path, img_label=None, transform=None):
        '''其中img_path和img_label均为list'''
        self.img_path = img_path
        self.img_label = img_label
        self.transform = transform

    '''根据下标返回数据(img和label)'''

    def __getitem__(self, index):
        if self.img_label != None:
            img = Image.open(self.img_path[index]).convert('RGB')
            # 转换成整形
            label = np.array(self.img_label[index], dtype=np.int)

            if (self.transform != None):
                img = self.transform(img)
            return img, torch.from_numpy(label)
        else:
            img = Image.open(self.img_path[index]).convert('RGB')
            if (self.transform != None):
                img = self.transform(img)
            return img, torch.from_numpy(np.array([]))

    '''返回数据集长度'''

    def __len__(self):
        return len(self.img_path)


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomAffine(8),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

path, tag = [], []
for root, dirs, files in os.walk(r"./data/train"):
    for file in files:
        path.append(os.path.join(root, file))
        tag.append(os.path.split(root)[-1])

trainset = CtmDataset(path, tag, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

path, tag = [], []
for root, dirs, files in os.walk(r"./data/test"):
    for file in files:
        path.append(os.path.join(root, file))
        tag.append(os.path.split(root)[-1])

testset = CtmDataset(path, tag, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')


# functions to show an image


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(BATCH_SIZE)))

net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(3072, 2048),
    nn.ReLU(),
    nn.Linear(2048, 10)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(100):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

PATH = './HandWriteNumberRecognition.pth'
torch.save(net.state_dict(), PATH)

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(BATCH_SIZE)))

net.load_state_dict(torch.load(PATH))

outputs = net(images)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(BATCH_SIZE)))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(BATCH_SIZE):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(ITEM_NUM):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
