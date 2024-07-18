import torch
import torch.nn as nn
from yolov5.models.yolo import Model
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.nn import AdaptiveAvgPool2d
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import torch.nn.functional as F

save_dir = r"C:\Users\14869\Desktop\yolo_sim\weight"
# 加载最佳模型权重
model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth')))
model.to(device)

# 测试集数据加载器
test_dataset = YOLOv5Dataset(image_dir='path_to_test_images',
                             label_dir='path_to_test_labels',
                             transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)


def test(model, test_loader):
    model.eval()
    test_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            yolo_outputs, projections = model(images)

            # 计算损失
            loss = yolo_loss(yolo_outputs, labels) + contrastive_loss(projections, projections)
            test_loss += loss.item()

            # 保存预测结果和真实标签，用于后续计算指标
            all_predictions.append(yolo_outputs.cpu())
            all_labels.append(labels.cpu())

    test_loss /= len(test_loader)

    # 合并所有批次的预测和标签
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # 计算评估指标（准确率、召回率、F1分数等）
    # 这里假设你有一个自定义函数 calculate_metrics 计算这些指标
    metrics = calculate_metrics(all_predictions, all_labels)

    return test_loss, metrics


def calculate_metrics(predictions, labels):
    # 实现用于计算评估指标的函数
    # 返回一个包含所有指标的字典
    metrics = {}
    # 示例：计算准确率
    # metrics['accuracy'] = ...
    return metrics


# 进行测试
test_loss, test_metrics = test(model, test_loader)
print(f'Test Loss: {test_loss}')
print(f'Test Metrics: {test_metrics}')

