# 戴口罩和眼镜推理文件-fa
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
from PIL import Image
import base64
from io import BytesIO
import os
import torchvision.transforms as transforms
from torchvision.models import resnet50
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import csv
import time

# 数据集类
class CustomImageFolder(Dataset):
    # 加载数据集，并返回图片路径列表：images
    def __init__(self, root, transform=None, loader=default_loader):
        self.root = root
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.loader = loader
        self.samples = self._make_dataset()

    # 被调用
    def _make_dataset(self):
        images = []
        for filename in os.listdir(self.root):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                path = os.path.join(self.root, filename)
                images.append(path)
        return images

    def __getitem__(self, index):
        path = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, path

    def __len__(self):
        return len(self.samples)

# 数据集类
class OneImageDataset(Dataset):
    def __init__(self, images, transform=None, loader=Image.open):
        """
        初始化数据集。

        :param images: 单张图片路径（str）或图片路径列表（list of str）
        :param transform: 可选的图片变换操作
        :param loader: 图片加载函数，默认为 PIL.Image.open
        """
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.loader = loader
        self.samples = self._make_dataset(images)

    def _make_dataset(self, images):
        """
        处理输入的图片路径，确保返回一个列表。

        :param images: 单张图片路径（str）或图片路径列表（list of str）
        :return: 图片路径列表
        """
        if isinstance(images, str):  # 如果是单张图片路径
            return [images]
        elif isinstance(images, list):  # 如果是图片路径列表
            return images
        else:
            raise ValueError("images 参数必须是字符串或字符串列表")

    def __getitem__(self, index):
        """
        根据索引获取图片和路径。

        :param index: 索引
        :return: (图片张量, 图片路径)
        """
        path = self.samples[index]
        sample = self.loader(path).convert('RGB')  # 确保图片是 RGB 格式
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, path

    def __len__(self):
        """
        返回数据集的大小。

        :return: 数据集中图片的数量
        """
        return len(self.samples)


class OneImageBase64Dataset(Dataset):
    def __init__(self, images, transform=None):
        """
        初始化数据集。

        :param images: 单张 Base64 编码的图片（str）或 Base64 编码的图片列表（list of str）
        :param transform: 可选的图片变换操作
        """
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.samples = self._make_dataset(images)

    def _make_dataset(self, images):
        """
        处理输入的 Base64 编码图片，确保返回一个列表。

        :param images: 单张 Base64 编码图片（str）或 Base64 编码图片列表（list of str）
        :return: Base64 编码图片列表
        """
        if isinstance(images, str):  # 如果是单张 Base64 编码图片
            return [images]
        elif isinstance(images, list):  # 如果是 Base64 编码图片列表
            return images
        else:
            raise ValueError("images 参数必须是字符串或字符串列表")

    def _load_image_from_base64(self, base64_str):
        """
        从 Base64 编码字符串加载图片。

        :param base64_str: Base64 编码的图片字符串
        :return: PIL.Image 对象
        """
        try:
            # 解码 Base64 数据
            image_data = base64.b64decode(base64_str)
            # 使用 BytesIO 将二进制数据转换为文件对象
            image_file = BytesIO(image_data)
            # 使用 PIL.Image 打开图片
            image = Image.open(image_file).convert('RGB')  # 确保图片是 RGB 格式
            return image
        except Exception as e:
            raise ValueError(f"无法从 Base64 数据加载图片: {e}")

    def __getitem__(self, index):
        """
        根据索引获取图片和 Base64 数据。

        :param index: 索引
        :return: (图片张量, Base64 编码字符串)
        """
        base64_str = self.samples[index]
        sample = self._load_image_from_base64(base64_str)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, base64_str

    def __len__(self):
        """
        返回数据集的大小。

        :return: 数据集中图片的数量
        """
        return len(self.samples)



def load_model(model_path, num_classes):
    """加载预训练模型"""
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model = resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.to(device)
    model.eval()
    return model, device


# 推理函数
def inference_batch(model, dataloader, device, idx_to_class):
    predictions = []
    with torch.no_grad():
        for inputs, paths in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            # ？？？
            preds = preds.cpu().numpy()

            for pred, path in zip(preds, paths):
                # predictions.append((os.path.basename(path), pred))
                predicted_class = idx_to_class[pred]
                predictions.append((os.path.basename(path), predicted_class))
    return predictions

def inference(model, dataset, device, idx_to_class):
    """
    对 OneImageBase64Dataset 数据集进行推理。

    :param model: 已加载的 PyTorch 模型
    :param dataset: OneImageBase64Dataset 实例
    :param device: 模型运行设备（如 'cuda' 或 'cpu'）
    :param idx_to_class: 索引到类别名称的映射字典
    :return: 预测结果列表 [(base64_str, predicted_class), ...]
    """
    predictions = []
    model.eval()  # 设置模型为评估模式

    with torch.no_grad():  # 关闭梯度计算
        for i in range(len(dataset)):  # 遍历数据集中的每张图片
            # 获取图片和 Base64 编码字符串
            image_tensor, base64_str = dataset[i]

            # 将图片移动到指定设备
            image_tensor = image_tensor.unsqueeze(0).to(device)  # 增加 batch 维度

            # 模型推理
            outputs = model(image_tensor)
            _, pred = torch.max(outputs, 1)  # 获取预测索引
            pred = pred.cpu().numpy()[0]  # 转换为 numpy 并提取标量值

            # 映射预测索引到类别名称
            predicted_class = idx_to_class[pred]

            # 将预测结果添加到列表中
            predictions.append((base64_str, predicted_class))

    return predictions

def write_to_csv(predictions, output_csv='predictions.csv'):
    """将预测结果写入CSV文件"""
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['ImagePath', 'Predicted Class'])  # 写入表头
        writer.writerows(predictions)  # 写入预测结果


# 执行推理
if __name__ == "__main__":
    print('开始推理')
    start_time = time.time()
    print(f'start_time:{start_time}')
    # 训练使用的文件类根路径
    # DATAROOT = './data-classify-v1/train'
    # 推理图片文件夹路径
    # infer_dir = './data-classify-v1/valid/test/jietu'
    infer_dir = './_crop_img_0.2'
    # 模型权重路径
    model_path = './best_model.pth'
   # 单张图片，处理成base64编码传入
    with open("./_crop_img_0.2/crop_with_glasses_fa.jpg", "rb") as f:
        base64_str = base64.b64encode(f.read()).decode("utf-8")

    #类别名称-类别索引 映射
    # data_class = sorted(os.listdir(DATAROOT))
    data_class = ['crop_with_glasses','crop_with_mask','no_crop_with_mask_and_glasses','crop_without_mask_and_glasses']
    custom_class_to_idx = {}
    for index, file_name in enumerate(data_class):
        custom_class_to_idx[file_name] = index

    # 类别索引-类别名称 映射
    idx_to_class = {v: k for k, v in custom_class_to_idx.items()}
    print(idx_to_class)

    # 加载推理数据集 获取数据集文件路径
    # infer_dataset = CustomImageFolder(root=infer_dir)
    # infer_loader = DataLoader(infer_dataset, batch_size=4, shuffle=False, num_workers=4)

   # 加载推理的base64编码格式数据集
    dataset = OneImageBase64Dataset(images=base64_str)
    # sample, base64_data = dataset[0]
    # print(base64_data)  # 输出 Base64 编码字符串
    # 加载模型
    model, device = load_model(model_path, num_classes=len(custom_class_to_idx))
    # model, device = load_model(model_path, num_classes=4)

    # ----进行推理-------
    # predictions = inference_batch(model, infer_loader, device, idx_to_class)
    predictions = inference(model, dataset, device, idx_to_class)

    print(predictions)
    end_time = time.time()
    print(f'end_time:{end_time}')
    print(end_time-start_time)

    # 输出结果
    for image_name, predicted_class in predictions:
        print(f'Image: {image_name[:10]}, Predicted Class: {predicted_class}')

    write_to_csv(predictions,'./output.csv')
