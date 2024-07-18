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

model = Model(cfg='yolov5s.yaml', ch=3, nc=1)  # 修改为适合你的配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# 创建用于保存模型权重的目录
save_dir = r"C:\Users\14869\Desktop\yolo_sim\weight"
os.makedirs(save_dir, exist_ok=True)

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 选择C3层的输出作为特征
feature_dim = 512  # C3层的输出特征尺寸
proj_head = ProjectionHead(input_dim=feature_dim, output_dim=128)
avg_pool = AdaptiveAvgPool2d((1, 1))

# 修改模型的前向传播函数以包含投影头
def forward_with_projection(x):
    # 获取到C3层的输出
    x = [model(x) for _ in model.model[:9](x)]


    # 应用平均池化，将特征尺寸调整为 [batch_size, 512, 1, 1]
    x = avg_pool(x)

    # 保存原始的卷积输出形状，用于在全连接层后恢复形状
    original_shape = x.shape  # 这里应该是 [batch_size, 512, 1, 1]

    # 将卷积层输出的特征展平以适配投影头的输入尺寸 [batch_size, 512]
    x_flatten = x.view(x.size(0), -1)
    projection = proj_head(x_flatten)

    # 在传递回卷积层之前，将数据从展平状态恢复到卷积层适用的形状
    x = x.view(original_shape)  # 确保这一步恢复到 [batch_size, 512, 1, 1]
    print(x)

    # 继续处理剩余的YOLOv5模型部分
    yolo_output = model.model[9:](x)
    return yolo_output, projection

model.forward = forward_with_projection



class YOLOv5Dataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = [os.path.join(image_dir, x) for x in sorted(os.listdir(image_dir))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert('RGB')

        label_path = os.path.join(self.label_dir, os.path.splitext(os.path.basename(img_path))[0] + '.txt')
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as file:
                for line in file:
                    labels.append([float(x) for x in line.strip().split()])

        if self.transform:
            img = self.transform(img)

        # 格式化为tensor
        labels = torch.tensor(labels)

        return img, labels

def collate_fn(batch):
    images, labels = zip(*batch)  # 解包图像和标签
    images = torch.stack([img for img in images])  # 堆叠图像

    # 处理标签，填充0使得所有标签的尺寸一致
    max_len = max(len(label) for label in labels)
    padded_labels = [torch.cat([label, torch.zeros(max_len - len(label), 5)]) if len(label) < max_len else label for label in labels]
    labels = torch.stack(padded_labels)

    return images, labels

# 数据转换
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])
train_dataset = YOLOv5Dataset(image_dir=r'C:\Users\14869\PycharmProjects\StudyPython\venv\Scripts\forestproject\datasets\coco128\images\train2017',
                              label_dir=r'C:\Users\14869\PycharmProjects\StudyPython\venv\Scripts\forestproject\datasets\coco128\labels\train2017',
                              transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

val_dataset = YOLOv5Dataset(image_dir=r'C:\Users\14869\PycharmProjects\StudyPython\venv\Scripts\forestproject\datasets\images\train2017',
                            label_dir=r'C:\Users\14869\PycharmProjects\StudyPython\venv\Scripts\forestproject\datasets\labels\train2017',
                            transform=transform)

val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# 对YOLO部分的损失函数
def yolo_loss(pred, target):
    # 假设pred和target的形状为 [batch_size, grid_size, grid_size, num_anchors, (5 + num_classes)]
    # 其中5表示 [x, y, w, h, conf]

    pred_boxes = pred[..., :4]
    pred_conf = pred[..., 4]
    pred_cls = pred[..., 5:]

    target_boxes = target[..., :4]
    target_conf = target[..., 4]
    target_cls = target[..., 5:]

    # 边界框损失
    box_loss = F.mse_loss(pred_boxes, target_boxes, reduction='sum')

    # 置信度损失
    conf_loss = F.binary_cross_entropy_with_logits(pred_conf, target_conf, reduction='sum')

    # 分类损失
    cls_loss = F.cross_entropy(pred_cls, target_cls, reduction='sum')

    # 总损失
    loss = box_loss + conf_loss + cls_loss

    return loss


def contrastive_loss(proj1, proj2, temperature=0.5):
    # 假设proj1和proj2的形状为 [batch_size, feature_dim]

    # 单位化特征向量
    proj1 = F.normalize(proj1, dim=1)
    proj2 = F.normalize(proj2, dim=1)

    # 计算相似度矩阵
    similarities = torch.mm(proj1, proj2.t()) / temperature

    # 计算对比损失
    batch_size = proj1.size(0)
    labels = torch.arange(batch_size).to(proj1.device)

    loss_i = F.cross_entropy(similarities, labels)
    loss_j = F.cross_entropy(similarities.t(), labels)

    loss = (loss_i + loss_j) / 2.0

    return loss

# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train_one_epoch(train_loader):
    model.train()
    total_loss = 0
    for images, _ in train_loader:
        images = images.to(device)
        optimizer.zero_grad()
        yolo_outputs, projections = model(images)

        # 计算YOLO损失和对比损失
        loss = yolo_loss(yolo_outputs, labels) + contrastive_loss(projections, projections)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def validate(val_loader):
    model.eval()
    with torch.no_grad():
        validation_loss = 0
        for images, _ in val_loader:
            images = images.to(device)
            yolo_outputs, projections = model(images)
            loss = yolo_loss(yolo_outputs, labels) + contrastive_loss(projections, projections)
            validation_loss += loss.item()
    return validation_loss / len(val_loader)


epochs = 10
best_val_loss = float('inf')  # 用于保存最佳模型

for epoch in range(epochs):
    train_loss = train_one_epoch(train_loader)
    val_loss = validate(val_loader)
    print(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Validation Loss: {val_loss}')

    # 保存每一轮的模型权重
    torch.save(model.state_dict(), os.path.join(save_dir, f'model_epoch_{epoch + 1}.pth'))

    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
