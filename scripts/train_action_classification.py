import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from model.network import Simple3DCNN



class KineticsDataset(Dataset):
    def __init__(self, csv_file, root_dir, clip_length=16, frame_interval=2, transform=None, is_train=True):
        """
        参数:
            csv_file: 包含视频标签的CSV文件路径
            root_dir: 帧数据根目录
            clip_length: 每个clip的帧数
            frame_interval: 帧采样间隔
            transform: 图像变换
            is_train: 是否为训练模式
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.clip_length = clip_length
        self.frame_interval = frame_interval
        self.transform = transform
        self.is_train = is_train
        
        # 获取所有类别（动作名称）
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        print(f"Found {len(self.classes)} classes: {self.classes}")
        print(f"Dataset size: {len(self.annotations)}")
        
        # 打印前几个样本的信息用于调试
        for i in range(min(3, len(self.annotations))):
            print(f"Sample {i}: {self.annotations.iloc[i]}")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        try:
            # 获取视频信息 - 根据CSV文件的实际格式调整
            video_info = self.annotations.iloc[idx]
            
            # 调试信息
            # print(f"Processing index {idx}: {video_info}")
            
            if len(video_info) >= 2:
                # 如果是两列，第一列是视频路径，第二列是标签
                label_name = str(video_info[0].replace(" ", "_")).strip()
                youtube_id = str(video_info[1]).strip()
                time_start = int(video_info['time_start'])
                time_end = int(video_info['time_end'])
            else:  
                print("Wrong video path!!!!!!!!!!!!")
      
            video_folder = f"{youtube_id}_{time_start:06d}_{time_end:06d}"         
            # 构建完整视频路径
            video_dir = os.path.join(self.root_dir, label_name, video_folder)
            
            
            # 获取所有帧文件
            if os.path.exists(video_dir):
                frame_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.jpg')])
            else:
                frame_files = []
            
            if len(frame_files) == 0:
                # 如果视频目录不存在或没有帧，使用随机数据并打印警告
                print(f"Warning: No frames found in {video_dir}")
                frames = torch.randn(3, self.clip_length, 112, 112)
                label = self.class_to_idx.get(label_name, 0)
                return frames, label
            
            # 计算实际可用的帧数
            total_frames = len(frame_files)
            available_length = total_frames // self.frame_interval
            
            if available_length < self.clip_length:
                # 如果帧数不足，重复采样
                frame_indices = list(range(0, total_frames, self.frame_interval))
                if len(frame_indices) < self.clip_length:
                    # 重复序列直到达到所需长度
                    repeated_indices = []
                    while len(repeated_indices) < self.clip_length:
                        repeated_indices.extend(frame_indices)
                    frame_indices = repeated_indices[:self.clip_length]
                else:
                    frame_indices = frame_indices[:self.clip_length]
            else:
                # 随机选择起始点（训练时）或从中心开始（测试时）
                if self.is_train:
                    start_idx = np.random.randint(0, available_length - self.clip_length + 1)
                else:
                    start_idx = (available_length - self.clip_length) // 2
                
                frame_indices = [start_idx * self.frame_interval + i * self.frame_interval 
                               for i in range(self.clip_length)]
            
            # 读取帧并应用变换
            frames = []
            for frame_idx in frame_indices:
                if frame_idx >= total_frames:
                    frame_idx = total_frames - 1
                    
                frame_path = os.path.join(video_dir, frame_files[frame_idx])
                try:
                    image = Image.open(frame_path).convert('RGB')
                    if self.transform:
                        image = self.transform(image)
                    frames.append(image)
                except Exception as e:
                    # 如果读取失败，使用黑色图像
                    print(f"Error loading frame {frame_path}: {e}")
                    frames.append(torch.zeros(3, 112, 112))
            
            # 将帧堆叠成 [C, T, H, W] 格式
            if len(frames) > 0:
                frames = torch.stack(frames, dim=1)  # 变成 [3, 16, 112, 112]
            else:
                frames = torch.randn(3, self.clip_length, 112, 112)
            
            # 获取标签
            label = self.class_to_idx.get(label_name, 0)
            
            return frames, label
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            # 返回随机数据作为fallback
            frames = torch.randn(3, self.clip_length, 112, 112)
            label = 0
            return frames, label


def debug_dataset(csv_file, root_dir):
    """调试函数，用于检查数据读取是否正确"""
    print("=== Debug Dataset ===")
    annotations = pd.read_csv(csv_file)
    print(f"CSV file: {csv_file}")
    print(f"Number of samples: {len(annotations)}")
    print(f"Columns: {annotations.columns.tolist()}")
    
    # 打印前5个样本
    for i in range(min(5, len(annotations))):
        row = annotations.iloc[i]
        print(f"Sample {i}:")
        print(f"  Label: {row['label']}")
        print(f"  YouTube ID: {row['youtube_id']}")
        print(f"  Time start: {row['time_start']}")
        print(f"  Time end: {row['time_end']}")
        
        # 构建视频文件夹名称
        label_name = str(row['label'].replace(" ", "_"))
        youtube_id = str(row['youtube_id']).strip()
        time_start = int(row['time_start'])
        time_end = int(row['time_end'])
        
        video_folder = f"{youtube_id}_{time_start:06d}_{time_end:06d}"
        video_dir = os.path.join(root_dir, label_name, video_folder)
        
        print(f"  Constructed path: {video_dir}")
        print(f"  Path exists: {os.path.exists(video_dir)}")
        
        if os.path.exists(video_dir):
            frame_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.jpg')])
            print(f"  Number of frames: {len(frame_files)}")
        print()

def train_model(model, train_loader, val_loader, num_epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        # 计算准确率
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_3dcnn_model.pth')
            print(f'New best model saved with val_acc: {val_acc:.2f}%')
        
        scheduler.step()
        print('-' * 60)
    
    print(f'Training completed. Best val accuracy: {best_acc:.2f}%')


def main():
    parser = argparse.ArgumentParser(description='3D CNN for Video Action Recognition')
    parser.add_argument('--data_root', type=str, default='data/kinetics400_30fps_frames', 
                       help='Root directory of dataset')
    parser.add_argument('--train_csv', type=str, default='tiny_train.csv', 
                       help='Training CSV file')
    parser.add_argument('--val_csv', type=str, default='tiny_val.csv', 
                       help='Validation CSV file')
    parser.add_argument('--clip_length', type=int, default=16, 
                       help='Number of frames per clip')
    parser.add_argument('--frame_interval', type=int, default=2, 
                       help='Interval between sampled frames')
    parser.add_argument('--batch_size', type=int, default=8, 
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=200, 
                       help='Number of training epochs')
    parser.add_argument('--num_workers', type=int, default=4, 
                       help='Number of workers for data loading')
    parser.add_argument('--debug', action='store_true', 
                       help='Debug mode to check dataset')
    
    args = parser.parse_args()
    
    # 调试模式
    if args.debug:
        debug_dataset(os.path.join(args.data_root, args.train_csv), args.data_root)
        debug_dataset(os.path.join(args.data_root, args.val_csv), args.data_root)
        return
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 数据变换
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomCrop((112, 112)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.CenterCrop((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = KineticsDataset(
        csv_file=os.path.join(args.data_root, args.train_csv),
        root_dir=args.data_root,
        clip_length=args.clip_length,
        frame_interval=args.frame_interval,
        transform=train_transform,
        is_train=True
    )
    
    val_dataset = KineticsDataset(
        csv_file=os.path.join(args.data_root, args.val_csv),
        root_dir=args.data_root,
        clip_length=args.clip_length,
        frame_interval=args.frame_interval,
        transform=val_transform,
        is_train=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # 创建模型
    model = Simple3DCNN(num_classes=len(train_dataset.classes), 
                        clip_length=args.clip_length)
    model = model.to(device)
    
    print(f'Model created with {sum(p.numel() for p in model.parameters()):,} parameters')
    print(f'Number of classes: {len(train_dataset.classes)}')
    
    # 训练模型
    train_model(model, train_loader, val_loader, args.num_epochs, device)

if __name__ == '__main__':
    main()