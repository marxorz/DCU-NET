# -*- coding: utf-8 -*-
"""
DCU-Net 迁移学习分类脚本
功能：
1. 加载预训练的 DCU-Net 去噪模型作为特征提取器 (BackBone)。
2. 冻结 BackBone 参数，只训练新加的 Classification Head。
3. 任务：区分 [纯噪声 (Class 0)] vs [流星信号 (Class 1)]。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import os
import copy
from sklearn.metrics import accuracy_score, roc_auc_score

# 配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3

# ==========================================
# 1. 基础组件 (必须与训练时一致)
# ==========================================
class ComplexConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConv1d, self).__init__()
        self.conv_real = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_imag = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
    def forward(self, x):
        channels = x.shape[1] // 2
        real, imag = x[:, :channels, :], x[:, channels:, :]
        return torch.cat([self.conv_real(real)-self.conv_imag(imag), self.conv_real(imag)+self.conv_imag(real)], dim=1)

class ComplexBatchNorm1d(nn.Module):
    def __init__(self, num_features):
        super(ComplexBatchNorm1d, self).__init__()
        self.bn_real = nn.BatchNorm1d(num_features)
        self.bn_imag = nn.BatchNorm1d(num_features)
    def forward(self, x):
        channels = x.shape[1] // 2
        real, imag = x[:, :channels, :], x[:, channels:, :]
        return torch.cat([self.bn_real(real), self.bn_imag(imag)], dim=1)

class ComplexReLU(nn.Module):
    def forward(self, x): return torch.relu(x)

class ComplexEncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(ComplexConv1d(in_ch, out_ch, 3, 2, 1), ComplexBatchNorm1d(out_ch), ComplexReLU(), ComplexConv1d(out_ch, out_ch, 3, 1, 1), ComplexBatchNorm1d(out_ch), ComplexReLU())
    def forward(self, x): return self.net(x)

# 为了加载权重，我们需要定义完整的 DCUNet，但在 forward 中只用 Encoder
class DCUNet_Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = nn.Sequential(ComplexConv1d(1, 32, 3, 1, 1), ComplexBatchNorm1d(32), ComplexReLU())
        self.down1 = ComplexEncoderBlock(32, 64)
        self.down2 = ComplexEncoderBlock(64, 128)
        self.down3 = ComplexEncoderBlock(128, 256)
        self.bot = nn.Sequential(ComplexConv1d(256, 256, 3, 1, 1), ComplexBatchNorm1d(256), ComplexReLU())
        # Decoder 部分定义了但不会用到，为了匹配 state_dict 键值
        # 这里为了省事可以不定义 Decoder，但在加载权重时需要 strict=False
        # 为了严谨，我们还是完整定义，但在 forward 里截断
    
    def forward_features(self, x):
        # x: [Batch, 440, 2] -> [Batch, 2, 440]
        x = x.permute(0, 2, 1)
        x_in = torch.cat([x[:, 0:1, :], x[:, 1:2, :]], dim=1)
        
        x1 = self.inc(x_in)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        feat = self.bot(x4) # [Batch, 512, 55] (256real+256imag)
        return feat

# ==========================================
# 2. 分类模型 (BackBone + Head)
# ==========================================
class MeteorClassifier(nn.Module):
    def __init__(self, pretrained_path, num_classes=2):
        super().__init__()
        
        # 1. BackBone
        self.backbone = DCUNet_Backbone()
        
        # 加载预训练权重 (strict=False 允许我们忽略 Decoder 部分的权重)
        if os.path.exists(pretrained_path):
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            # 过滤掉 decoder 的权重，只加载 encoder
            encoder_dict = {k: v for k, v in checkpoint.items() if 'up' not in k and 'outc' not in k}
            self.backbone.load_state_dict(encoder_dict, strict=False)
            print(">>> 预训练权重加载成功 (Encoder Only)")
        else:
            print("!!! 警告：未找到预训练权重，将随机初始化 !!!")

        # 2. 冻结 BackBone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 3. Classification Head
        # Bottleneck 输出通道是 512 (256 real + 256 imag)
        self.pool = nn.AdaptiveAvgPool1d(1) # [B, 512, 55] -> [B, 512, 1]
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # 提取特征 (No Grad)
        with torch.no_grad():
            feat = self.backbone.forward_features(x)
        
        # 分类
        pooled = self.pool(feat)
        logits = self.head(pooled)
        return logits

# ==========================================
# 3. 分类数据集
# ==========================================
class ClassificationDataset(Dataset):
    def __init__(self, h5_folder):
        super().__init__()
        self.samples = []
        self.labels = []
        
        print(">>> 构建分类数据集...")
        
        # --- 类 0: 纯噪声 ---
        noise_path = os.path.join(h5_folder, 'noise_only.h5')
        if os.path.exists(noise_path):
            with h5py.File(noise_path, 'r') as f:
                n_real = f['signals_real'][:]
                n_imag = f['signals_imag'][:]
                # [N, 440, 2]
                data = np.stack([n_real, n_imag], axis=2)
                
                self.samples.append(torch.from_numpy(data).float())
                self.labels.append(torch.zeros(len(data)).long()) # Label 0
                print(f"  Load Noise (Class 0): {len(data)} samples")
        
        # --- 类 1: 流星信号 (5dB - 30dB) ---
        # 我们不使用 0dB 训练分类器，因为它太模糊，可能导致模型困惑
        # 或者你可以包含 0dB 增加难度
        snr_list = [5, 10, 15, 20, 25, 30]
        meteor_data_list = []
        
        for snr in snr_list:
            path = os.path.join(h5_folder, f'snr_{snr}.h5')
            if os.path.exists(path):
                with h5py.File(path, 'r') as f:
                    # 使用 Noisy 数据作为输入 (因为实际场景只有 Noisy)
                    n_real = f['signals_real'][:]
                    n_imag = f['signals_imag'][:]
                    data = np.stack([n_real, n_imag], axis=2)
                    meteor_data_list.append(data)
        
        if meteor_data_list:
            all_meteors = np.concatenate(meteor_data_list, axis=0)
            
            # 为了数据平衡，如果流星太多，随机采样到和噪声差不多数量
            # 或者噪声太少，就多采一些噪声
            # 这里假设直接全部使用
            self.samples.append(torch.from_numpy(all_meteors).float())
            self.labels.append(torch.ones(len(all_meteors)).long()) # Label 1
            print(f"  Load Meteor (Class 1): {len(all_meteors)} samples")
            
        self.x = torch.cat(self.samples, dim=0)
        self.y = torch.cat(self.labels, dim=0)
        
        print(f">>> 数据集构建完成: 总计 {len(self.x)} 样本")

    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

# ==========================================
# 4. 训练流程
# ==========================================
def main():
    h5_folder = r'F:\denoise'
    pretrained_path = './checkpoints/dcunet/best_dcunet_finetuned.pth'
    save_dir = './checkpoints/classifier'
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 数据
    full_dataset = ClassificationDataset(h5_folder)
    
    # 划分 8:2
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. 模型
    model = MeteorClassifier(pretrained_path, num_classes=2).to(device)
    
    # 3. 优化器 (只优化 Head)
    optimizer = optim.AdamW(model.head.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    print("\n>>> 开始训练分类头 (BackBone 已冻结)...")
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        # --- Train ---
        model.head.train() # 只开启 Head 的 Dropout/BN 更新
        train_loss = 0
        correct = 0
        total = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, pred = torch.max(logits, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            
        train_acc = correct / total
        
        # --- Validation ---
        model.eval()
        val_correct = 0
        val_total = 0
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                
                _, pred = torch.max(logits, 1)
                val_correct += (pred == y).sum().item()
                val_total += y.size(0)
                
                # 收集概率用于计算 AUC
                probs = torch.softmax(logits, dim=1)[:, 1] # Class 1 probability
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                
        val_acc = val_correct / val_total
        try:
            val_auc = roc_auc_score(all_labels, all_probs)
        except:
            val_auc = 0.0
            
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Loss: {train_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Val AUC: {val_auc:.4f}")
        
        scheduler.step(val_acc)
        
        # 保存最优
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_classifier.pth'))
            print("  -> Best Model Saved!")

    print(f"\n训练完成。最佳验证准确率: {best_acc:.4f}")
    
    # === 推理示例 ===
    print("\n>>> 推理示例 (输出概率):")
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_classifier.pth')))
    model.eval()
    
    # 拿几个样本试试
    x_sample, y_sample = next(iter(val_loader))
    x_sample = x_sample[:5].to(device)
    with torch.no_grad():
        logits = model(x_sample)
        probs = torch.softmax(logits, dim=1) # [N, 2]
    
    for i in range(5):
        p_noise = probs[i][0].item()
        p_meteor = probs[i][1].item()
        true_label = "Meteor" if y_sample[i].item() == 1 else "Noise"
        print(f"Sample {i}: True={true_label:<7} | Pred: Noise={p_noise:.2f}, Meteor={p_meteor:.2f}")

if __name__ == '__main__':
    main()