import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

class NN(nn.Module):  #定义神经网络
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(NN, self).__init__()

        self.fc1 = nn.Sequential(  #使定义的池化层形状和全连接层输入的形状一样
            nn.Flatten(),#压平
            nn.Linear(2352,input_size),
            nn.Linear(input_size, hidden_size1),               
        ) 
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.out = nn.Linear(hidden_size2, num_classes)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.out(x)
        return x
    
def evaluate_model_metrics(model, test_loader):#指标计算函数
    TP=[0 for i in range (10)]
    TN=[0 for i in range (10)]
    FP=[0 for i in range (10)]
    FN=[0 for i in range (10)]
    model.eval()
    with torch.no_grad():
        for i,(images, labels) in enumerate(test_loader):
            images = images
            labels = labels
            outputs = model(images)
            predicted = outputs.argmax(1)
            for i in range (len(predicted)):
                label=labels[i]
                predict=predicted[i]
                if label==predict:
                    TP[label]+=1
                    for j in range (10):
                        if j==label:
                            continue
                        TN[j]+=1
                else :
                    FN[label]+=1
                    FP[predict]+=1
                    for j in range(10):
                        if j==label or j==predict:
                            continue
                        TN[j]+=1
            if (i+1) % 100 == 0:
                print(f'Step [{i+1}/{len(test_loader)}]')
    accuracy=[0 for i in range (10)]
    precision=[0 for i in range (10)]
    recall=[0 for i in range (10)]
    f1_score=[0 for i in range (10)]
    for i in range (10):
        accuracy[i]=(TP[i] + TN[i]) / (TP[i] + TN[i] + FP[i] + FN[i])
        precision[i] = TP[i] / (TP[i] + FP[i]) if (TP[i] + FP[i]) != 0 else 0
        recall[i] = TP[i] / (TP[i]+ FN[i]) if (TP[i]+ FN[i]) != 0 else 0
        f1_score[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) != 0 else 0
    
    return accuracy, precision, recall, f1_score

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(), 
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.ImageFolder(root='.\\mnist\\mnist_train', transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
test_dataset = datasets.ImageFolder(root='.\\mnist\\mnist_test', transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=True)
model = NN(input_size=28*28, hidden_size1=128, hidden_size2=256, num_classes=10)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs=3
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images
        labels = labels
        # print(images.shape)
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
print('Training Finished')

accuracy, precision, recall, f1_score = evaluate_model_metrics(model, test_loader)
for i in range (10):
    print(i,':')
    print(f'Accuracy: {accuracy[i]:.4f}')
    print(f'Precision: {precision[i]:.4f}')
    print(f'Recall: {recall[i]:.4f}')
    print(f'F1 Score: {f1_score[i]:.4f}')