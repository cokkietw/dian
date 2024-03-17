import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# 定义RNN神经网络
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.flatten = nn.Sequential(
            nn.Flatten(2,-1),
        )
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        input = self.flatten(input)
        # 遍历输入序列的每个时间步
        for i in range(input.shape[1]):
            # 获取当前时间步的输入
            input_step = input[:,i,:]
            # 将输入与隐藏状态连接
            combined = torch.cat((input_step, hidden), 1)
            # 更新隐藏状态
            hidden = torch.relu(self.i2h(combined))
            # 计算输出
            output = self.i2o(combined)
            output = self.softmax(output)
        return output, hidden
    
    def initHidden(self,batch_size):
        return torch.zeros(batch_size, self.hidden_size)
    
# 定义训练函数
def train(model, criterion, optimizer, data):
    model.train()
    for i,(images, labels) in enumerate(data):
        hidden = model.initHidden(images.shape[0])
        
        loss = 0
        # print(images.shape,images[0].shape,images[0].view(1, -1),images.size(0))
        output, hidden = model(images, hidden)
        loss = criterion(output, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
def evaluate_model_metrics(model, test_loader):#指标计算函数&测试函数
    TP=[0 for i in range (10)]
    TN=[0 for i in range (10)]
    FP=[0 for i in range (10)]
    FN=[0 for i in range (10)]
    model.eval()
    with torch.no_grad():
        for i,(images, labels) in enumerate(test_loader):
            images = images
            labels = labels
            hidden = model.initHidden(images.shape[0])
            outputs , _ = model(images,hidden)
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

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

# 加载数据集
batch_size=128
train_dataset = datasets.ImageFolder(root='.\\fashion-mnist\\mnist_train', transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.ImageFolder(root='.\\fashion-mnist\\mnist_test', transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# 训练模型
model = RNN(input_size=28*28, hidden_size=128, output_size=10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 3
for epoch in range(num_epochs):
    train(model, criterion, optimizer, train_loader)

# 测试模型
accuracy, precision, recall, f1_score = evaluate_model_metrics(model, test_loader)
for i in range (10):
    print(i,':')
    print(f'Accuracy: {accuracy[i]:.4f}')
    print(f'Precision: {precision[i]:.4f}')
    print(f'Recall: {recall[i]:.4f}')
    print(f'F1 Score: {f1_score[i]:.4f}')
