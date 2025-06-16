# 该文件用于是否戴眼镜和口罩训练

import csv
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.models import resnet50
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import cv2
# PyTorch的torchvision库提供了大量的预训练模型，如ResNet、VGG、AlexNet等，这些模型通常用于图像识别任务。
# transforms进行图像裁剪的操作


def load_trainLoader(root):
    # 定义图像预处理步骤
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(root=root,transform=transform_train)
    # --------------------------
    print(dataset.classes)
    # batch_size = 32 每次从数据集中读取的数据量为 32 个样本
    # shuffle = True 则在每个epoch开始时，DataLoader会随机打乱数据集中的样本顺序
    data_loader = DataLoader(dataset,batch_size = 2,shuffle=True)
    return data_loader


def load_testLoader(root):
    # 定义图像预处理步骤
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(root=root,transform=transform_test)
    # dataset = CustomDataset(root,transform_test )
    data_loader = DataLoader(dataset,batch_size = 1,shuffle=True)
    return data_loader

def load_model():
    # 加载预训练模型.pretrained=True会从互联网下载ResNet的预训练权重（如果本地没有缓存的话），并将其加载到模型中。pretrained=True参数确保我们加载的是带有预训练权重的模型。
    # 假设你有一个名为'model_weights.pth'的预训练权重文件
    # 加载权重到模型中，这里假设权重文件的字典键与模型参数名称相匹配
    model = resnet50(pretrained=False)
    checkpoint = torch.load('resnet50-19c8e357.pth')
    model.load_state_dict(checkpoint)

    # 修改最后的全连接层
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs,4)
    return model

# 训练模型 train_model()->load_model() 加载resnet50模型及预训练权重
#                     ->load_trainLoader('data-classify-v1/train')加载训练图像并做预处理
#                     ->load_testLoader('data-classify-v1/test')加载测试图像并做预处理
def train_model():
    # 1、调用load_model()加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(device)=="cuda":
        print("模型在CUDA上运行")
    else:
        print("模型在cpu上运行")
    model = load_model()
    model.to(device)

    # 2、定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 3、加载训练数据集、测试数据集
    train_loader = load_trainLoader('data-classify-v1/train')
    test_loader = load_testLoader('data-classify-v1/test')

    best_acc = 0.
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            # 释放多余内存，防止溢出
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        torch.save(model.state_dict, f'data/runs/resnet50/Epoch{epoch+1}.pth')
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader.dataset)}')

        # 验证模型
        # 用于设置模型为评估模式
        model.eval()
        total = 0
        correct = 0
        # 在代码块内关闭梯度计算，即在代码块内执行的操作不会跟踪梯度，也不会计算梯度
        with torch.no_grad():
            for images,labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        print(f'Epoch {epoch + 1},acc: {acc:.2f}%')
        # 获取最佳准确率
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f'data/runs/resnet50/best_model.pth')
            print(f'Current best validation accuracy: {best_acc:.2f}%')



def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model()
    model.load_state_dict(torch.load('./data/runs/resnet50/best_model.pth'))
    model.to(device)

    dataloader = load_testLoader('./data-classify-v1/valid')
    # 验证模型
    model.eval()
    total = 0
    correct = 0
    all_label = []
    all_pred = []
    all_name = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_label.append(labels.cpu().numpy().item())
            all_pred.append(predicted.cpu().numpy().item())


    # all_label = torch.cat(all_label, dim=0)
    # all_pred = torch.cat(all_pred, dim=0)
    print(f'Accuracy of the network on the {len(dataloader.dataset)} test images:{100*correct/total}%')
    with open('./data/runs/result.csv','w',newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['pred','label'])
        writer.writerows([all_pred ,all_label])
        print('write in csv')


if __name__ == "__main__":
    train_model()

