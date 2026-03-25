# -*- coding: utf-8 -*-
"""
流星雷达去噪测试脚本 - DCU-Net 专用版
测试重点：相位连续性、台阶效应是否消除、低信噪比恢复能力
"""

import torch
import torch.nn as nn
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# 绘图设置
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ==========================================
# 1. 模型定义 (必须与 train_dcunet.py 完全一致)
# ==========================================
class ComplexConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConv1d, self).__init__()
        self.conv_real = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_imag = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        
    def forward(self, x):
        channels = x.shape[1] // 2
        real = x[:, :channels, :]
        imag = x[:, channels:, :]
        out_real = self.conv_real(real) - self.conv_imag(imag)
        out_imag = self.conv_real(imag) + self.conv_imag(real)
        return torch.cat([out_real, out_imag], dim=1)

class ComplexBatchNorm1d(nn.Module):
    def __init__(self, num_features):
        super(ComplexBatchNorm1d, self).__init__()
        self.bn_real = nn.BatchNorm1d(num_features)
        self.bn_imag = nn.BatchNorm1d(num_features)

    def forward(self, x):
        channels = x.shape[1] // 2
        real = x[:, :channels, :]
        imag = x[:, channels:, :]
        return torch.cat([self.bn_real(real), self.bn_imag(imag)], dim=1)

class ComplexReLU(nn.Module):
    def forward(self, x): return torch.relu(x)

class ComplexEncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            ComplexConv1d(in_ch, out_ch, kernel_size=3, padding=1, stride=2),
            ComplexBatchNorm1d(out_ch),
            ComplexReLU(),
            ComplexConv1d(out_ch, out_ch, kernel_size=3, padding=1),
            ComplexBatchNorm1d(out_ch),
            ComplexReLU()
        )
    def forward(self, x): return self.net(x)

class ComplexDecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.trans = nn.ConvTranspose1d(in_ch*2, in_ch*2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            ComplexConv1d(in_ch + out_ch, out_ch, kernel_size=3, padding=1),
            ComplexBatchNorm1d(out_ch),
            ComplexReLU()
        )
    def forward(self, x, skip):
        upsampled = self.trans(x)
        if upsampled.shape[2] != skip.shape[2]:
            upsampled = torch.nn.functional.interpolate(upsampled, size=skip.shape[2])
        uc = upsampled.shape[1] // 2
        sc = skip.shape[1] // 2
        u_real, u_imag = upsampled[:, :uc], upsampled[:, uc:]
        s_real, s_imag = skip[:, :sc], skip[:, sc:]
        combined = torch.cat([torch.cat([u_real, s_real], 1), torch.cat([u_imag, s_imag], 1)], 1)
        return self.conv(combined)

class DCUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = nn.Sequential(ComplexConv1d(1, 32, 3, padding=1), ComplexBatchNorm1d(32), ComplexReLU())
        self.down1 = ComplexEncoderBlock(32, 64)
        self.down2 = ComplexEncoderBlock(64, 128)
        self.down3 = ComplexEncoderBlock(128, 256)
        self.bot = nn.Sequential(ComplexConv1d(256, 256, 3, padding=1), ComplexBatchNorm1d(256), ComplexReLU())
        self.up1 = ComplexDecoderBlock(256, 128)
        self.up2 = ComplexDecoderBlock(128, 64)
        self.up3 = ComplexDecoderBlock(64, 32)
        self.outc = ComplexConv1d(32, 1, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x_in = torch.cat([x[:, 0:1, :], x[:, 1:2, :]], dim=1)
        x1 = self.inc(x_in)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bot(x4)
        x = self.up1(x5, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x).permute(0, 2, 1).contiguous()

# ==========================================
# 2. 物理工具 (速度与相位计算)
# ==========================================
PRF = 430.0
c = 2.99792458e8
f = 35e6
lamda = c / f

def estimate_velocity(signal):
    """鲁棒的速度估计"""
    try:
        phase = np.unwrap(np.angle(signal))
        t = np.arange(len(signal)) / PRF
        
        # 能量检测: 只拟合有信号的部分
        mag = np.abs(signal)
        threshold = 0.15 * np.max(mag)
        valid_mask = mag > threshold
        
        # 如果有效点太少，回退到中间 80%
        if np.sum(valid_mask) < 20:
            start = int(len(signal) * 0.1)
            end = int(len(signal) * 0.9)
            valid_indices = np.arange(start, end)
        else:
            valid_indices = np.where(valid_mask)[0]
            
        t_fit = t[valid_indices]
        p_fit = phase[valid_indices]
        
        if len(t_fit) < 2: return 0.0, t, phase, [0,0]
        
        # 线性拟合
        p = np.polyfit(t_fit, p_fit, 1)
        v = -lamda * p[0] / (4 * np.pi)
        return v, t, phase, p
    except:
        return 0.0, np.arange(len(signal))/PRF, np.zeros_like(signal), [0,0]

def calculate_metrics(clean, denoised, noisy):
    """计算核心指标"""
    # 1. SNR
    noise_in = np.mean(np.abs(noisy - clean)**2)
    noise_out = np.mean(np.abs(denoised - clean)**2)
    noise_in = max(noise_in, 1e-12)
    noise_out = max(noise_out, 1e-12)
    snr_imp = 10 * np.log10(np.mean(np.abs(clean)**2) / noise_out) - \
              10 * np.log10(np.mean(np.abs(clean)**2) / noise_in)

    # 2. Velocity
    v_clean, _, _, _ = estimate_velocity(clean)
    v_denoised, _, _, _ = estimate_velocity(denoised)
    v_err = np.abs(v_clean - v_denoised)

    # 3. Phase MAE (Weighted)
    mag = np.abs(clean)
    weight = mag / (np.max(mag) + 1e-9)
    weight[weight < 0.1] = 0 # 忽略底噪
    
    ph_clean = np.angle(clean)
    ph_denoised = np.angle(denoised)
    
    # 最小相位差 (-pi ~ pi)
    diff = np.arctan2(np.sin(ph_clean - ph_denoised), np.cos(ph_clean - ph_denoised))
    
    if np.sum(weight) > 0:
        ph_mae = np.sum(np.abs(diff) * weight) / np.sum(weight)
    else:
        ph_mae = 0.0

    return snr_imp, v_err, ph_mae

# ==========================================
# 3. 可视化函数 (重点看相位是否平滑)
# ==========================================
def visualize_sample(noisy, clean, denoised, snr_label, idx, save_dir):
    t = np.arange(len(noisy)) / PRF
    v_c, _, ph_c, p_c = estimate_velocity(clean)
    v_d, _, ph_d, p_d = estimate_velocity(denoised)
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    
    # 幅度
    axs[0,0].plot(t, np.abs(noisy), 'silver', label='Noisy')
    axs[0,0].plot(t, np.abs(clean), 'g--', linewidth=1.5, label='Clean')
    axs[0,0].plot(t, np.abs(denoised), 'b', alpha=0.8, linewidth=1.5, label='DCU-Net')
    axs[0,0].set_title(f'Magnitude (Input SNR={snr_label}dB)')
    axs[0,0].legend()
    axs[0,0].grid(True, alpha=0.3)
    
    # 相位 (Unwrapped) - 检查是否有台阶
    axs[0,1].plot(t, ph_c, 'g.', markersize=2, label='Clean')
    axs[0,1].plot(t, ph_d, 'b-', alpha=0.6, linewidth=1.5, label='DCU-Net')
    axs[0,1].set_title('Phase (Unwrapped) - Check Smoothness')
    axs[0,1].legend()
    axs[0,1].grid(True, alpha=0.3)
    
    # 频谱
    axs[1,0].plot(np.abs(np.fft.fft(noisy)), 'silver')
    axs[1,0].plot(np.abs(np.fft.fft(clean)), 'g--')
    axs[1,0].plot(np.abs(np.fft.fft(denoised)), 'b')
    axs[1,0].set_yscale('log')
    axs[1,0].set_title('Spectrum')
    axs[1,0].grid(True, alpha=0.3)
    
    # 速度拟合线
    valid_mask = np.abs(clean) > 0.15 * np.max(np.abs(clean))
    t_v = t[valid_mask]
    if len(t_v) > 0:
        fit_c = np.polyval(p_c, t)
        fit_d = np.polyval(p_d, t)
        axs[1,1].plot(t, ph_c, 'g.', markersize=1, alpha=0.3)
        axs[1,1].plot(t, fit_c, 'k--', label=f'True V={v_c:.1f}')
        axs[1,1].plot(t, fit_d, 'r-', linewidth=1.5, label=f'Pred V={v_d:.1f}')
    axs[1,1].set_title(f'Velocity Error: {abs(v_c-v_d):.2f} m/s')
    axs[1,1].legend()
    axs[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'vis_{snr_label}dB_sample{idx}.png'))
    plt.close()

# ==========================================
# 4. 主程序
# ==========================================
def main():
    h5_folder = r'F:\denoise'
    model_path = './checkpoints/dcunet/best_dcunet_finetuned.pth'
    save_dir = './test_results_dcunet_finetune'
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Loading DCU-Net from: {model_path}")
    model = DCUNet().to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully.")
    else:
        print("Error: Model file not found.")
        return

    model.eval()
    
    # 测试所有 SNR (包括训练没用过的 0dB)
    snr_list = [0, 5, 10, 15, 20, 25, 30]
    target_file = os.path.join(h5_folder, 'snr_inf.h5')
    
    with h5py.File(target_file, 'r') as f:
        clean_signals = f['signals_real'][:] + 1j * f['signals_imag'][:]

    results_summary = {'snr': [], 'snr_imp': [], 'vel_err': []}

    print("-" * 65)
    print(f"{'SNR':<8} | {'SNR Imp (dB)':<15} | {'Vel Err (m/s)':<15} | {'Phase MAE':<15}")
    print("-" * 65)

    for snr in snr_list:
        noisy_file = os.path.join(h5_folder, f'snr_{snr}.h5')
        if not os.path.exists(noisy_file): continue
            
        with h5py.File(noisy_file, 'r') as f:
            noisy_signals = f['signals_real'][:] + 1j * f['signals_imag'][:]
        
        n_samples = min(200, len(noisy_signals))
        curr_imp, curr_vel, curr_pha = [], [], []
        
        for i in range(n_samples):
            noisy = noisy_signals[i]
            clean = clean_signals[i]
            
            # === 相对归一化 ===
            max_val = np.max(np.abs(noisy))
            if max_val < 1e-9: max_val = 1.0
            
            # Input: [1, 440, 2]
            inp = torch.stack([
                torch.from_numpy((noisy/max_val).real), 
                torch.from_numpy((noisy/max_val).imag)
            ], dim=1).float().unsqueeze(0).to(device)
            
            # Inference
            with torch.no_grad():
                out = model(inp)
                out_np = out.cpu().numpy().squeeze(0)
                pred_norm = out_np[:, 0] + 1j * out_np[:, 1]
                
                # === 反归一化 ===
                denoised = pred_norm * max_val
            
            # Metrics
            s_imp, v_err, p_mae = calculate_metrics(clean, denoised, noisy)
            
            if v_err < 500: 
                curr_vel.append(v_err)
            curr_imp.append(s_imp)
            curr_pha.append(p_mae)
            
            # Visualize
            if i < 3:
                visualize_sample(noisy, clean, denoised, str(snr), i, save_dir)
        
        mean_imp = np.mean(curr_imp)
        mean_vel = np.mean(curr_vel)
        mean_pha = np.mean(curr_pha)
        
        print(f"{snr:<8} | {mean_imp:<15.2f} | {mean_vel:<15.2f} | {mean_pha:<15.4f}")
        
        results_summary['snr'].append(snr)
        results_summary['snr_imp'].append(mean_imp)
        results_summary['vel_err'].append(mean_vel)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(results_summary['snr'], results_summary['snr_imp'], 'o-')
    plt.xlabel('Input SNR (dB)')
    plt.ylabel('SNR Improvement (dB)')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(results_summary['snr'], results_summary['vel_err'], 'r^-')
    plt.xlabel('Input SNR (dB)')
    plt.ylabel('Velocity Error (m/s)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'summary_metrics.png'))
    print(f"\nResults saved to: {save_dir}")

if __name__ == '__main__':
    main()